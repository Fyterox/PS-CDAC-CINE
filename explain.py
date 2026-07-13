"""
explain.py — explainability maps for the fused detector.

For a given image it produces a panel with:
  1. the input face
  2. a gradient-saliency heatmap for P(fake) overlaid on the face
     (where the fused model is "looking")
  3. the FFT log-magnitude spectrum (what the frequency stream sees)
  4. the SRM noise residual (what the noise stream sees)
and prints the fused P(fake) plus each stream's auxiliary P(fake), which shows
WHICH stream drove a given decision — most useful on the FaceApp cases.

Example
-------
  python explain.py --ckpt runs/fusion/best.pt --images img1.png img2.png \
      --out-dir runs/explain
  # or sample N test images straight from the manifest:
  python explain.py --ckpt runs/fusion/best.pt --manifest manifest.csv --n 8 \
      --out-dir runs/explain
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import get_device, load_state_dict_flexible, ID2GEN
from models.detector import build_model
from models.freq_encoder import FrequencyEncoder
from models.noise_encoder import SRMConv


def load_image(path, size=224):
    img = Image.open(path).convert("RGB").resize((size, size))
    x = torch.from_numpy(np.asarray(img).astype("float32") / 255.0)
    return x.permute(2, 0, 1)  # [3,H,W] in [0,1]


def saliency(model, x, device):
    """abs input-gradient of P(fake), max over channels, min-max normalised."""
    x = x.unsqueeze(0).to(device).requires_grad_(True)
    logits, _ = model(x, return_aux=False)
    p_fake = torch.softmax(logits, dim=1)[0, 1]
    model.zero_grad(set_to_none=True)
    p_fake.backward()
    g = x.grad.detach().abs().amax(dim=1)[0]        # [H,W]
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g.cpu().numpy(), float(p_fake.detach())


@torch.no_grad()
def stream_scores(model, x, device):
    _, aux = model(x.unsqueeze(0).to(device), return_aux=True)
    return {k: float(torch.softmax(v, dim=1)[0, 1]) for k, v in aux.items()}


def make_panel(model, x, device, title, out_path):
    sal, p_fake = saliency(model, x, device)
    scores = stream_scores(model, x, device)

    with torch.no_grad():
        spec = FrequencyEncoder._to_spectrum(x.unsqueeze(0).to(device))[0]
        spec = spec.mean(0).cpu().numpy()
        srm = SRMConv().to(device)(x.unsqueeze(0).to(device))[0]
        srm = srm.abs().mean(0).cpu().numpy()

    img = x.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, 4, figsize=(14, 4))
    ax[0].imshow(img); ax[0].set_title("input")
    ax[1].imshow(img); ax[1].imshow(sal, cmap="jet", alpha=0.5)
    ax[1].set_title(f"saliency  P(fake)={p_fake:.2f}")
    ax[2].imshow(spec, cmap="magma"); ax[2].set_title("FFT log-mag (freq)")
    ax[3].imshow(srm, cmap="viridis"); ax[3].set_title("SRM residual (noise)")
    for a in ax:
        a.axis("off")
    sub = "  ".join(f"{k}:{v:.2f}" for k, v in scores.items())
    fig.suptitle(f"{title}   fused P(fake)={p_fake:.2f}   [per-stream {sub}]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}  fused={p_fake:.3f}  streams={scores}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--n", type=int, default=8, help="sample N from manifest test")
    ap.add_argument("--out-dir", default="runs/explain")
    ap.add_argument("--cnn-backbone", default="resnet18")
    ap.add_argument("--fusion-dim", type=int, default=512)
    ap.add_argument("--fusion-depth", type=int, default=3)
    ap.add_argument("--stream-dim", type=int, default=512)
    return ap.parse_args()


def main():
    args = parse_args()
    device = get_device()
    cfg = dict(rgb_ckpt=None, rgb_pretrained=False, cnn_pretrained=False,
               cnn_backbone=args.cnn_backbone, stream_dim=args.stream_dim,
               fusion_dim=args.fusion_dim, fusion_depth=args.fusion_depth,
               use_bayar=True)
    model = build_model(cfg).to(device)
    load_state_dict_flexible(model, torch.load(args.ckpt, map_location="cpu", weights_only=False),
                             drop_prefixes=())
    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    samples = []  # (path, title)
    if args.images:
        samples = [(p, os.path.basename(p)) for p in args.images]
    elif args.manifest:
        import pandas as pd
        df = pd.read_csv(args.manifest)
        df = df[df["split"] == "test"].sample(min(args.n, len(df)), random_state=0)
        samples = [(r["path"], f'{r["generator"]}/{"fake" if r["label"] else "real"}')
                   for _, r in df.iterrows()]
    else:
        raise SystemExit("Pass --images or --manifest")

    for i, (path, title) in enumerate(samples):
        x = load_image(path)
        make_panel(model, x, device, title, os.path.join(args.out_dir, f"explain_{i:02d}.png"))


if __name__ == "__main__":
    main()
