"""
app.py — interactive UI for the multi-stream deepfake detector.

Upload a face image; get back:
  * a real/fake verdict with confidence
  * per-stream confidences (RGB / frequency / noise), so the decision can be
    attributed to a stream rather than just asserted
  * a saliency map over P(fake)  — where the fused model is looking
  * the FFT log-magnitude spectrum — what the frequency stream reads
  * the SRM noise residual        — what the noise stream reads

The inference path deliberately mirrors data.py: images go in as RAW [0,1]
tensors resized to 224x224, and each encoder does its own preprocessing
internally. Normalising here would corrupt the FFT and SRM streams.

Run:
    python app/app.py --ckpt runs/fusion/best.pt
    python app/app.py --ckpt runs/fusion/best.pt --share     # public link (Colab)
"""
import argparse
import os
import sys

import numpy as np
import torch
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# allow running as `python app/app.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import get_device, load_state_dict_flexible
from models.detector import build_model
from models.freq_encoder import FrequencyEncoder
from models.noise_encoder import SRMConv

IMG_SIZE = 224
MODEL = None
SRM = None
DEVICE = None


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(ckpt, cnn_backbone="resnet18", fusion_dim=512, fusion_depth=3,
               stream_dim=512):
    global MODEL, SRM, DEVICE
    DEVICE = get_device()
    cfg = dict(
        rgb_ckpt=None, rgb_pretrained=False, cnn_pretrained=False,
        cnn_backbone=cnn_backbone, stream_dim=stream_dim,
        fusion_dim=fusion_dim, fusion_depth=fusion_depth, use_bayar=True,
    )
    model = build_model(cfg).to(DEVICE)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = load_state_dict_flexible(model, sd, drop_prefixes=())
    print(f"[load] {ckpt}")
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 10:
        print("[load] WARNING: many missing keys — check --cnn-backbone / "
              "--fusion-dim match the trained model.")
    model.eval()
    MODEL = model
    SRM = SRMConv().to(DEVICE)
    return model


def to_tensor(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = torch.from_numpy(np.asarray(img).astype("float32") / 255.0)
    return x.permute(2, 0, 1)  # [3,H,W] in [0,1] — RAW, not normalised


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def saliency_map(x):
    """abs input-gradient of P(fake), max over channels, min-max normalised."""
    xb = x.unsqueeze(0).to(DEVICE).requires_grad_(True)
    logits, _ = MODEL(xb, return_aux=False)
    p_fake = torch.softmax(logits, dim=1)[0, 1]
    MODEL.zero_grad(set_to_none=True)
    p_fake.backward()
    g = xb.grad.detach().abs().amax(dim=1)[0]
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g.cpu().numpy(), float(p_fake.detach())


@torch.no_grad()
def per_stream_scores(x):
    _, aux = MODEL(x.unsqueeze(0).to(DEVICE), return_aux=True)
    return {k: float(torch.softmax(v, dim=1)[0, 1]) for k, v in aux.items()}


@torch.no_grad()
def stream_views(x):
    xb = x.unsqueeze(0).to(DEVICE)
    spec = FrequencyEncoder._to_spectrum(xb)[0].mean(0).cpu().numpy()
    srm = SRM(xb)[0].abs().mean(0).cpu().numpy()
    return spec, srm


def colorize(arr, cmap="magma"):
    a = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    rgb = (plt.get_cmap(cmap)(a)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def overlay_saliency(x, sal, alpha=0.5):
    base = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    heat = (plt.get_cmap("jet")(sal)[..., :3] * 255).astype(np.uint8)
    blend = (base * (1 - alpha) + heat * alpha).astype(np.uint8)
    return Image.fromarray(blend)


# --------------------------------------------------------------------------- #
# Inference callback
# --------------------------------------------------------------------------- #
def analyze(pil_img):
    if pil_img is None:
        return "Upload an image to begin.", None, None, None, None

    x = to_tensor(pil_img)
    sal, p_fake = saliency_map(x)
    scores = per_stream_scores(x)
    spec, srm = stream_views(x)

    verdict = "FAKE" if p_fake >= 0.5 else "REAL"
    conf = p_fake if p_fake >= 0.5 else 1 - p_fake
    color = "#E8746B" if verdict == "FAKE" else "#5FD08A"

    # Which stream is most confident that it's fake — useful attribution signal.
    lead = max(scores, key=scores.get)
    pretty = {"rgb": "RGB (semantic)", "freq": "Frequency (spectral)",
              "noise": "Noise residual (local)"}

    rows = "".join(
        f"<tr><td style='padding:6px 14px 6px 0'>{pretty.get(k, k)}</td>"
        f"<td style='padding:6px 0'>"
        f"<div style='background:#1B3560;border-radius:4px;width:200px;height:10px;display:inline-block;vertical-align:middle'>"
        f"<div style='background:#3ED8E8;width:{v*200:.0f}px;height:10px;border-radius:4px'></div></div>"
        f"<span style='margin-left:10px;font-family:monospace'>{v:.3f}</span></td></tr>"
        for k, v in scores.items()
    )

    html = f"""
    <div style="font-family:system-ui,sans-serif">
      <div style="font-size:38px;font-weight:700;color:{color};line-height:1.1">{verdict}</div>
      <div style="font-size:15px;color:#9DB2CE;margin-bottom:4px">
        confidence <b style="color:#fff">{conf:.1%}</b>
        &nbsp;·&nbsp; P(fake) = <span style="font-family:monospace">{p_fake:.3f}</span>
      </div>
      <hr style="border:none;border-top:1px solid #27446E;margin:14px 0">
      <div style="font-size:13px;color:#9DB2CE;margin-bottom:8px">
        Per-stream P(fake) — each stream's own auxiliary head:
      </div>
      <table style="font-size:13px;color:#fff">{rows}</table>
      <div style="font-size:12px;color:#9DB2CE;margin-top:12px">
        Most confident stream: <b style="color:#3ED8E8">{pretty.get(lead, lead)}</b>
      </div>
    </div>
    """

    return (
        html,
        pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
        overlay_saliency(x, sal),
        colorize(spec, "magma"),
        colorize(srm, "viridis"),
    )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
CSS = """
.gradio-container {max-width: 1180px !important}
footer {display:none !important}
"""

def build_ui():
    with gr.Blocks(title="Multi-Stream Deepfake Detector",
                   theme=gr.themes.Soft(primary_hue="cyan"), css=CSS) as demo:
        gr.Markdown(
            "# Multi-Stream Deepfake Detector\n"
            "Upload a face image. The model reads it three ways — semantic (RGB ViT), "
            "spectral (FFT), and local noise residual (SRM) — and fuses them with "
            "cross-attention. The panels below show what each stream sees, so the "
            "verdict can be attributed rather than just asserted."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Face image", height=320)
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=1):
                out_html = gr.HTML(label="Verdict")

        gr.Markdown("### What the model looked at")
        with gr.Row():
            im_in = gr.Image(label="Input (224×224)", height=240)
            im_sal = gr.Image(label="Saliency over P(fake)", height=240)
            im_fft = gr.Image(label="FFT log-magnitude — frequency stream", height=240)
            im_srm = gr.Image(label="SRM residual — noise stream", height=240)

        gr.Markdown(
            "<sub>A bright saliency region marks pixels that most changed the fake "
            "probability. The FFT panel exposes upsampling grids and GAN fingerprints; "
            "the SRM panel exposes local noise inconsistency, which is what betrays "
            "edited (rather than fully generated) faces.</sub>"
        )

        btn.click(analyze, inputs=inp,
                  outputs=[out_html, im_in, im_sal, im_fft, im_srm])
        inp.change(analyze, inputs=inp,
                   outputs=[out_html, im_in, im_sal, im_fft, im_srm])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained fusion checkpoint")
    ap.add_argument("--cnn-backbone", default="resnet18")
    ap.add_argument("--fusion-dim", type=int, default=512)
    ap.add_argument("--fusion-depth", type=int, default=3)
    ap.add_argument("--stream-dim", type=int, default=512)
    ap.add_argument("--share", action="store_true",
                    help="public gradio link (needed in Colab)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    load_model(args.ckpt, args.cnn_backbone, args.fusion_dim,
               args.fusion_depth, args.stream_dim)
    build_ui().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
