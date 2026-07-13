# Multi-Stream Deepfake Detector (RGB + Frequency + Noise-Residual, Cross-Attention Fusion)

The Week-3+ model from your PS-I framework slide, built end-to-end. It targets
the exact failure your cross-generator study exposed: a single RGB ViT misses
**FaceApp-style local attribute edits** because they leave weak *global*
artifacts. Two extra streams attack that directly:

| Stream | Sees | Catches |
|---|---|---|
| **RGB** (your fine-tuned ViT-B/16) | semantics / global appearance | full-face synthesis (StyleGAN, PGGAN) |
| **Frequency** (FFT log-mag → CNN) | spectral / grid artifacts from up-sampling | GAN & diffusion fingerprints |
| **Noise-residual** (SRM + Bayar → CNN) | local noise/sensor inconsistency | **FaceApp** local edits, splices, re-renders |

The three streams are fused by a **cross-attention transformer**: every stream's
tokens attend to every other stream's tokens, and a learnable `[FUSION]` token is
read into the classifier.

---

## Files

```
common.py            seeds, metrics, per-generator report, ckpt IO, discriminative LR groups
data.py              DFFDDataset (returns RAW [0,1] images), robustness aug
build_manifest.py    build manifest.csv from your DFFD folders
models/
  rgb_encoder.py     ViT-B/16, loads your DFFD checkpoint (timm or HF)
  freq_encoder.py    FFT log-magnitude spectrum -> CNN -> tokens
  noise_encoder.py   SRM + Bayar high-pass -> CNN -> tokens
  fusion.py          cross-attention transformer + deep-supervision aux heads
  detector.py        assembles the full model + single-stream wrapper for Stage 1
pretrain_stream.py   Stage 1: warm-start freq / noise streams
train.py             Stage 2+3: staged fusion training (freeze RGB, then unfreeze)
evaluate.py          test metrics + per-generator table (maps to your midsem chart)
explain.py           saliency + FFT/SRM visualisations + per-stream scores
```

---

## The plan (why it's staged)

The danger with multi-stream models is that a strong pretrained stream (your ViT)
dominates and the fresh streams never learn — you get "just the ViT" with extra
parameters. Three defences, all baked in:

1. **Stage 1 — warm-start the weak streams.** Train frequency and noise encoders
   *alone* as real/fake classifiers first, so they enter fusion already useful.
2. **Stage 2 — freeze RGB, train fusion.** New params learn to combine with a
   frozen, already-good ViT. No corruption of your DFFD backbone.
3. **Stage 3 — unfreeze end-to-end at discriminative LRs.** RGB gets a small LR
   (`base_lr * 0.1`), everything else the full LR. Cosine schedule.
4. **Deep supervision.** Each stream keeps its own aux head during training
   (`aux_weight=0.3`), so no stream is allowed to go dead.

---

## Run order

### 0. Install
```bash
pip install -r requirements.txt   # Colab already has torch/torchvision
```

### 1. Build the manifest (reuse your existing DFFD split!)
```bash
# If your DFFD is already split into train/val/test folders:
python build_manifest.py --mode presplit --root /content/DFFD --out manifest.csv
# If your folder names differ from the vocab, remap them:
#   --map ffhq=real
```
Vocab: `real, faceapp, stargan, pggan, stylegan`. The per-generator report needs
these tags to line up with your midsem chart.

### 2. Stage 1 — warm-start freq + noise (≈5 epochs each)
```bash
python pretrain_stream.py --stream freq  --manifest manifest.csv \
    --epochs 5 --batch-size 64 --out runs/freq_pretrain/best.pt
python pretrain_stream.py --stream noise --manifest manifest.csv \
    --epochs 5 --batch-size 64 --out runs/noise_pretrain/best.pt
```

### 3. Stage 2+3 — train the fusion model
```bash
python train.py \
  --manifest manifest.csv \
  --rgb-ckpt /content/vit_dffd.pt \
  --freq-init runs/freq_pretrain/best.pt \
  --noise-init runs/noise_pretrain/best.pt \
  --epochs 12 --freeze-rgb-epochs 4 \
  --batch-size 16 --accum 2 --base-lr 3e-4 \
  --monitor f1 --out runs/fusion/best.pt
```
`--rgb-ckpt` is **your** DFFD-fine-tuned ViT. See "RGB checkpoint" below if you
trained with HuggingFace instead of timm.

### 4. Evaluate — get the numbers for your report
```bash
python evaluate.py --manifest manifest.csv --ckpt runs/fusion/best.pt
```
Prints overall metrics + the per-generator flagged-as-fake table. Read the
**FaceApp** row against your baseline: the whole point is to lift it above the
0.79-era number.

### 5. Explainability maps
```bash
python explain.py --ckpt runs/fusion/best.pt --manifest manifest.csv --n 8 \
    --out-dir runs/explain
```

---

## RGB checkpoint (read this once)

`--rgb-ckpt` accepts your fine-tuned weights and the loader auto-strips the old
2-class head. It prints how many backbone keys were missing — **that number
should be ~0**. If it's large, the backend/model name is wrong:

- **timm** (default): `--rgb-backend timm --rgb-model vit_base_patch16_224`.
  Works with a raw `state_dict`, a `{'model': ...}` wrapper, or `module.` prefixes.
- **HuggingFace**: convert once, then use `--rgb-backend hf`:
  ```python
  from transformers import ViTForImageClassification
  m = ViTForImageClassification.from_pretrained("your/ckpt")
  torch.save(m.vit.state_dict(), "vit_backbone.pt")
  ```
  `--rgb-backend hf --rgb-model google/vit-base-patch16-224-in21k --rgb-ckpt vit_backbone.pt`

---

## T4 / Colab memory notes (16 GB)

Defaults are tuned for a T4. Three encoders + ViT-B is heavier than your baseline:

- `--batch-size 16 --accum 2` → effective batch 32 (same as your baseline) but
  fits in 16 GB. If you OOM: drop to `--batch-size 8 --accum 4`.
- Stage 2 (RGB frozen) is the cheap phase — you can use a bigger batch there;
  the script keeps one batch size for simplicity, so size for Stage 3.
- Mixed precision (AMP) is on everywhere. FFT and SRM are forced to fp32 inside
  their modules (fp16 FFT is numerically unreliable) — that's intentional.
- `resnet18` streams keep the extra cost small. `efficientnet_b0` is a drop-in
  via `--cnn-backbone efficientnet_b0` if you want more capacity.

---

## Design choices worth defending in your report

- **Raw [0,1] images to the model, per-stream preprocessing inside each encoder.**
  Normalising before FFT/SRM would distort the very signals those streams exist to
  read. The RGB stream normalises internally with ImageNet stats.
- **SRM (fixed) + Bayar (learnable) high-pass.** Fixed SRM gives a strong prior
  from the steganalysis/forensics literature (Zhou et al., RGB-N); Bayar adapts a
  data-driven high-pass on top. Together they surface local tamper residuals.
- **Cross-attention via a joint transformer over multi-stream tokens.** Simpler
  and more expressive than late-fusing three pooled vectors; the `[FUSION]` token
  learns which stream to trust per-image (visible in `explain.py`'s per-stream scores).
- **Recall is the headline metric.** Your study's story is "missed fakes", so
  `evaluate.py` foregrounds fake-recall and the per-generator breakdown.

## Natural next steps (your slide's future-work list)
- Add diffusion faces as new fake sub-types in the manifest (the pipeline is
  generator-agnostic — just add folders + tags).
- Video benchmarks (FaceForensics++, Celeb-DF): extract frames, tag per source,
  same manifest schema.
- Continual learning: the frozen-RGB + trainable-fusion split is already a good
  substrate for adding streams/generators without full retraining.
```
