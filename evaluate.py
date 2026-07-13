"""
evaluate.py — test-set evaluation of the trained fusion model.

Prints the overall metrics AND the per-generator flagged-as-fake table, which
gives the per-generator breakdown, which is the main robustness result: it shows
exactly which generators the detector still misses.

Example
-------
  python evaluate.py --manifest manifest.csv --ckpt runs/fusion/best.pt
"""
import argparse
import numpy as np
import torch

from common import (seed_everything, get_device, compute_metrics,
                    per_generator_report, format_report, load_state_dict_flexible)
from data import make_test_loader
from models.detector import build_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--rgb-model", default="vit_base_patch16_224")
    ap.add_argument("--rgb-backend", default="timm", choices=["timm", "hf"])
    ap.add_argument("--cnn-backbone", default="resnet18")
    ap.add_argument("--fusion-dim", type=int, default=512)
    ap.add_argument("--fusion-depth", type=int, default=3)
    ap.add_argument("--stream-dim", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(42)
    device = get_device()

    cfg = dict(
        rgb_model=args.rgb_model, rgb_ckpt=None, rgb_backend=args.rgb_backend,
        rgb_pretrained=False,  # weights come from the fusion ckpt below
        cnn_pretrained=False,
        cnn_backbone=args.cnn_backbone, stream_dim=args.stream_dim,
        fusion_dim=args.fusion_dim, fusion_depth=args.fusion_depth, use_bayar=True,
    )
    model = build_model(cfg).to(device)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = load_state_dict_flexible(model, sd, drop_prefixes=())
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    loader = make_test_loader(args.manifest, split=args.split,
                              batch_size=args.batch_size, num_workers=args.num_workers)
    probs, labels, gens = [], [], []
    for x, y, g in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            p = model.predict_proba(x)
        probs.append(p.float().cpu().numpy()); labels.append(y.numpy()); gens.append(g.numpy())
    probs = np.concatenate(probs); labels = np.concatenate(labels); gens = np.concatenate(gens)

    m = compute_metrics(probs, labels)
    pg = per_generator_report(probs, labels, gens)
    print(f"\n=== TEST ({args.split}) — multi-stream fusion ===")
    print(format_report(m, pg))
    print("\nCompare fake-recall against the single-stream ViT baseline "
          "(0.29 cross-generator, 0.79 after DFFD fine-tuning). "
          "FaceApp is the row that matters.")


if __name__ == "__main__":
    main()
