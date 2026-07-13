"""
train.py — Stage 2 + 3: train the fusion model.

Strategy (staged, single command):
  * Stage 2 (epochs 0 .. freeze_rgb_epochs-1): RGB ViT FROZEN. The fresh
    frequency/noise/fusion params learn to be useful without being steamrolled
    by (or corrupting) the already-good RGB backbone.
  * Stage 3 (remaining epochs): RGB UNFROZEN at a small LR (discriminative LR),
    everything fine-tunes end-to-end.

Loss = CE(fusion logits) + aux_weight * mean(CE per-stream aux logits).

Checkpoints: best-by-val-metric (default macro-F1) saved to --out.

Example (T4-safe)
-----------------
  python train.py \
    --manifest manifest.csv \
    --rgb-ckpt /content/vit_dffd.pt \
    --freq-init runs/freq_pretrain/best.pt \
    --noise-init runs/noise_pretrain/best.pt \
    --epochs 12 --freeze-rgb-epochs 4 \
    --batch-size 16 --accum 2 --base-lr 3e-4 \
    --out runs/fusion/best.pt
"""
import argparse, math, os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from common import (seed_everything, get_device, build_param_groups,
                    set_rgb_trainable, AvgMeter, compute_metrics,
                    per_generator_report, format_report, save_checkpoint)
from data import make_loaders
from models.detector import build_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rgb-ckpt", default=None, help="DFFD-fine-tuned ViT checkpoint")
    ap.add_argument("--rgb-backend", default="timm", choices=["timm", "hf"])
    ap.add_argument("--rgb-model", default="vit_base_patch16_224")
    ap.add_argument("--freq-init", default=None, help="stage-1 freq encoder ckpt")
    ap.add_argument("--noise-init", default=None, help="stage-1 noise encoder ckpt")
    ap.add_argument("--cnn-backbone", default="resnet18")
    ap.add_argument("--fusion-dim", type=int, default=512)
    ap.add_argument("--fusion-depth", type=int, default=3)
    ap.add_argument("--stream-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--freeze-rgb-epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2, help="grad accumulation steps")
    ap.add_argument("--base-lr", type=float, default=3e-4)
    ap.add_argument("--rgb-lr-mult", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-robust-aug", action="store_true")
    ap.add_argument("--monitor", default="f1", choices=["f1", "recall", "acc", "auc"])
    ap.add_argument("--out", default="runs/fusion/best.pt")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def cosine_lr(step, total, warmup, base_scale=1.0):
    if step < warmup:
        return base_scale * step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return base_scale * 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels, gens = [], [], []
    for x, y, g in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            p = model.predict_proba(x)
        probs.append(p.float().cpu().numpy())
        labels.append(y.numpy())
        gens.append(g.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    gens = np.concatenate(gens)
    m = compute_metrics(probs, labels)
    pg = per_generator_report(probs, labels, gens)
    return m, pg


def maybe_load_encoder(model, attr, ckpt):
    if not ckpt:
        return
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    # Stage-1 saved a SingleStreamClassifier; keep only encoder.* keys.
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
              if k.startswith("encoder.")}
    if not enc_sd:  # maybe already an encoder state dict
        enc_sd = sd
    missing, unexpected = getattr(model, attr).load_state_dict(enc_sd, strict=False)
    print(f"[warm-start] {attr}: missing={len(missing)} unexpected={len(unexpected)}")


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    cfg = dict(
        rgb_model=args.rgb_model, rgb_ckpt=args.rgb_ckpt, rgb_backend=args.rgb_backend,
        cnn_backbone=args.cnn_backbone, stream_dim=args.stream_dim,
        fusion_dim=args.fusion_dim, fusion_depth=args.fusion_depth, use_bayar=True,
    )
    model = build_model(cfg).to(device)
    maybe_load_encoder(model, "freq", args.freq_init)
    maybe_load_encoder(model, "noise", args.noise_init)

    train_ld, val_ld = make_loaders(
        args.manifest, img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, robust_aug=not args.no_robust_aug,
    )

    # Start in Stage 2: RGB frozen.
    set_rgb_trainable(model, False)
    optim = torch.optim.AdamW(
        build_param_groups(model, args.base_lr, args.rgb_lr_mult, args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    ce = nn.CrossEntropyLoss()

    steps_per_epoch = math.ceil(len(train_ld) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * args.warmup_frac)
    base_lrs = [g["lr"] for g in optim.param_groups]

    best_val, gstep = -1.0, 0
    for epoch in range(args.epochs):
        if epoch == args.freeze_rgb_epochs:
            print(f"\n=== Stage 3: unfreezing RGB backbone at epoch {epoch} ===")
            set_rgb_trainable(model, True)
            # Rebuild optimiser so newly-trainable RGB params get their group.
            optim = torch.optim.AdamW(
                build_param_groups(model, args.base_lr, args.rgb_lr_mult,
                                   args.weight_decay))
            base_lrs = [g["lr"] for g in optim.param_groups]

        model.train()
        loss_m, main_m, aux_m = AvgMeter(), AvgMeter(), AvgMeter()
        optim.zero_grad(set_to_none=True)
        pbar = tqdm(train_ld, desc=f"epoch {epoch+1}/{args.epochs}")
        for i, (x, y, g) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits, aux = model(x, return_aux=True)
                loss_main = ce(logits, y)
                loss_aux = sum(ce(a, y) for a in aux.values()) / max(len(aux), 1)
                loss = loss_main + args.aux_weight * loss_aux
            scaler.scale(loss / args.accum).backward()

            if (i + 1) % args.accum == 0:
                scale = cosine_lr(gstep, total_steps, warmup)
                for pg_, blr in zip(optim.param_groups, base_lrs):
                    pg_["lr"] = blr * scale
                # guard against loss spikes NaN-ing a long run
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                gstep += 1

            loss_m.update(loss.item()); main_m.update(loss_main.item())
            aux_m.update(loss_aux.item())
            pbar.set_postfix(loss=f"{loss_m.avg:.3f}", main=f"{main_m.avg:.3f}",
                             aux=f"{aux_m.avg:.3f}",
                             lr=f"{optim.param_groups[-1]['lr']:.2e}")

        m, pg = evaluate(model, val_ld, device)
        print(f"[val epoch {epoch+1}]")
        print(format_report(m, pg))
        score = m[args.monitor]
        if score > best_val:
            best_val = score
            save_checkpoint(args.out, model, optim, scaler,
                            extra={"epoch": epoch, "val_metrics": m,
                                   "monitor": args.monitor, "score": score})
            print(f"  ** new best {args.monitor}={score:.4f} -> saved {args.out}")

    print(f"\nDone. Best val {args.monitor} = {best_val:.4f}")


if __name__ == "__main__":
    main()
