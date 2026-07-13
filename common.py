"""
common.py — shared plumbing for the multi-stream deepfake detector.

Conventions used everywhere in this repo:
  * label 1 == FAKE  (positive class)
  * label 0 == REAL
  * "recall" therefore means fake-detection rate — the exact metric that
    collapsed to 0.29 in the cross-generator study. It is the number to watch.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)

# Canonical generator vocabulary. "real" is id 0 by convention; the rest are
# fake sub-types. Keep this stable so checkpoints/manifests stay comparable.
GENERATORS = ["real", "faceapp", "stargan", "pggan", "stylegan"]
GEN2ID = {g: i for i, g in enumerate(GENERATORS)}
ID2GEN = {i: g for g, i in GEN2ID.items()}


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark=True is faster on fixed input sizes (224x224 here).
    torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Discriminative learning rates
# --------------------------------------------------------------------------- #
def build_param_groups(model, base_lr, rgb_lr_mult=0.1, weight_decay=0.05):
    """Give the pretrained RGB ViT a smaller LR than the fresh streams/fusion.

    The RGB backbone is already fine-tuned on DFFD, so we nudge it gently while
    the frequency / noise / fusion parameters (randomly or lightly initialised)
    learn faster. No weight decay on biases and norm params (standard trick).
    """
    rgb_decay, rgb_nodecay, new_decay, new_nodecay = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_rgb = name.startswith("rgb.")
        no_decay = p.ndim == 1 or name.endswith(".bias")
        if is_rgb:
            (rgb_nodecay if no_decay else rgb_decay).append(p)
        else:
            (new_nodecay if no_decay else new_decay).append(p)
    groups = [
        {"params": rgb_decay,    "lr": base_lr * rgb_lr_mult, "weight_decay": weight_decay},
        {"params": rgb_nodecay,  "lr": base_lr * rgb_lr_mult, "weight_decay": 0.0},
        {"params": new_decay,    "lr": base_lr,               "weight_decay": weight_decay},
        {"params": new_nodecay,  "lr": base_lr,               "weight_decay": 0.0},
    ]
    # Drop empty groups so the optimiser doesn't choke.
    return [g for g in groups if len(g["params"]) > 0]


def set_rgb_trainable(model, trainable: bool):
    """Freeze / unfreeze the RGB backbone for staged training."""
    for name, p in model.named_parameters():
        if name.startswith("rgb."):
            p.requires_grad = trainable


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
class AvgMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, val, k=1):
        self.sum += float(val) * k
        self.n += k

    @property
    def avg(self):
        return self.sum / max(self.n, 1)


@torch.no_grad()
def compute_metrics(probs_fake, labels):
    """probs_fake: 1-D array of P(fake). labels: 1-D array in {0,1}."""
    probs_fake = np.asarray(probs_fake)
    labels = np.asarray(labels)
    preds = (probs_fake >= 0.5).astype(int)
    out = {
        "acc":       accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),   # fake-detection rate
        "f1":        f1_score(labels, preds, zero_division=0),
    }
    # AUC needs both classes present.
    try:
        out["auc"] = roc_auc_score(labels, probs_fake)
    except ValueError:
        out["auc"] = float("nan")
    return out


@torch.no_grad()
def per_generator_report(probs_fake, labels, gen_ids):
    """Detection rate per generator — the headline breakdown for this project.

    For fake generators this is recall (fraction flagged as fake).
    For 'real' it is the false-positive rate (fraction wrongly flagged fake);
    lower is better there.
    """
    probs_fake = np.asarray(probs_fake)
    labels = np.asarray(labels)
    gen_ids = np.asarray(gen_ids)
    preds = (probs_fake >= 0.5).astype(int)
    rows = {}
    for gid in np.unique(gen_ids):
        m = gen_ids == gid
        name = ID2GEN.get(int(gid), str(gid))
        flagged_fake = preds[m].mean() if m.sum() else float("nan")
        rows[name] = {
            "n": int(m.sum()),
            "flagged_fake_rate": float(flagged_fake),
        }
    return rows


def format_report(metrics, per_gen=None):
    lines = [
        f"  acc={metrics['acc']:.4f}  P={metrics['precision']:.4f}  "
        f"R(fake-recall)={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  "
        f"AUC={metrics.get('auc', float('nan')):.4f}"
    ]
    if per_gen:
        lines.append("  per-generator flagged-as-fake rate:")
        for name in GENERATORS:
            if name in per_gen:
                r = per_gen[name]
                tag = "(want LOW )" if name == "real" else "(want HIGH)"
                lines.append(f"    {name:<9} n={r['n']:>6}  {r['flagged_fake_rate']:.4f} {tag}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Checkpoint IO
# --------------------------------------------------------------------------- #
def save_checkpoint(path, model, optimizer=None, scaler=None, extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_state_dict_flexible(module, state_dict, drop_prefixes=("head.", "fc.", "classifier."),
                             strip_prefixes=("module.", "model.")):
    """Load a possibly-mismatched state dict into `module`.

    Handles the common cases that come up when loading a fine-tuned ViT:
      * checkpoint wrapped as {'model': ...} or {'state_dict': ...}
      * 'module.' prefix from DataParallel
      * a classification head with a different shape than the feature-only model
    Returns (missing, unexpected) key lists for inspection.
    """
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for sp in strip_prefixes:
            if nk.startswith(sp):
                nk = nk[len(sp):]
        if any(nk.startswith(dp) for dp in drop_prefixes):
            continue  # skip old classifier head
        cleaned[nk] = v

    missing, unexpected = module.load_state_dict(cleaned, strict=False)
    return list(missing), list(unexpected)
