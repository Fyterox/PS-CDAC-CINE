"""
pretrain_stream.py — Stage 1 (optional but recommended).

Warm-start ONE fresh stream (frequency or noise) as a standalone real/fake
classifier before fusion. This gives that stream useful features so it can hold
its own against the strong RGB ViT during fusion, instead of being ignored.

Run this twice: once for --stream freq, once for --stream noise.

Example
-------
  python pretrain_stream.py --stream freq  --manifest manifest.csv \
      --epochs 5 --batch-size 64 --out runs/freq_pretrain/best.pt
  python pretrain_stream.py --stream noise --manifest manifest.csv \
      --epochs 5 --batch-size 64 --out runs/noise_pretrain/best.pt

The saved checkpoint is consumed by train.py via --freq-init / --noise-init.
"""
import argparse, math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from common import (seed_everything, get_device, AvgMeter, compute_metrics,
                    per_generator_report, format_report, save_checkpoint)
from data import make_loaders
from models.freq_encoder import FrequencyEncoder
from models.noise_encoder import NoiseResidualEncoder
from models.detector import SingleStreamClassifier


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True, choices=["freq", "noise"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cnn-backbone", default="resnet18")
    ap.add_argument("--stream-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--base-lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels, gens = [], [], []
    for x, y, g in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            p = torch.softmax(model(x), dim=1)[:, 1]
        probs.append(p.float().cpu().numpy()); labels.append(y.numpy()); gens.append(g.numpy())
    probs = np.concatenate(probs); labels = np.concatenate(labels); gens = np.concatenate(gens)
    return compute_metrics(probs, labels), per_generator_report(probs, labels, gens)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    if args.stream == "freq":
        enc = FrequencyEncoder(out_dim=args.stream_dim, backbone=args.cnn_backbone)
    else:
        enc = NoiseResidualEncoder(out_dim=args.stream_dim, backbone=args.cnn_backbone,
                                   use_bayar=True)
    model = SingleStreamClassifier(enc, in_dim=args.stream_dim).to(device)

    train_ld, val_ld = make_loaders(args.manifest, img_size=args.img_size,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers, robust_aug=True)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_lr,
                              weight_decay=args.weight_decay)
    total = len(train_ld) * args.epochs
    warmup = int(0.1 * total)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    ce = nn.CrossEntropyLoss()

    best, step = -1.0, 0
    for epoch in range(args.epochs):
        model.train(); lm = AvgMeter()
        pbar = tqdm(train_ld, desc=f"[{args.stream}] epoch {epoch+1}/{args.epochs}")
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            # cosine LR w/ warmup
            if step < warmup:
                lr = args.base_lr * step / max(warmup, 1)
            else:
                prog = (step - warmup) / max(total - warmup, 1)
                lr = args.base_lr * 0.5 * (1 + math.cos(math.pi * prog))
            for pg in optim.param_groups:
                pg["lr"] = lr
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                loss = ce(model(x), y)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()
            step += 1; lm.update(loss.item())
            pbar.set_postfix(loss=f"{lm.avg:.3f}", lr=f"{lr:.2e}")

        m, pg = evaluate(model, val_ld, device)
        print(f"[val {args.stream} epoch {epoch+1}]"); print(format_report(m, pg))
        if m["f1"] > best:
            best = m["f1"]
            save_checkpoint(args.out, model, extra={"val_metrics": m})
            print(f"  ** saved {args.out} (f1={best:.4f})")

    print(f"Done {args.stream}. best val f1={best:.4f}")


if __name__ == "__main__":
    main()
