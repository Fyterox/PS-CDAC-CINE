# Deepfake Detection — Face Image Classification

Detecting AI-generated and manipulated face images (real vs. fake) with a fine-tuned
Vision Transformer. This repo tracks a multi-week project: a **Week 1** baseline on a
single fake generator, and a **Week 2** study of cross-dataset generalization.

## Overview

The detector is a **ViT-Base/16** (ImageNet-pretrained, via [`timm`](https://github.com/huggingface/pytorch-image-models))
fine-tuned as a binary real/fake classifier on 224×224 face images.

- **Week 1 — Baseline.** Trained on the *140k Real and Fake Faces* dataset (FFHQ real +
  StyleGAN fake) and evaluated on its held-out test set.
- **Week 2 — Cross-dataset generalization.** Tested the Week 1 model on the *Diverse Fake
  Face Dataset (DFFD)*, which contains fake types it had never seen, then fine-tuned on
  DFFD to close the resulting generalization gap.

## Results

### Week 1 — 140k test set (StyleGAN fakes)

| Metric | Value |
|---|---|
| Accuracy | 99.62% |
| Precision | 99.41% |
| Recall | 99.84% |
| F1 | 99.63% |
| ROC-AUC | 1.000 |

### Week 2 — DFFD test set (diverse fakes): before vs. after fine-tuning

| Metric | Before (StyleGAN-only) | After (DFFD fine-tuned) | Δ |
|---|---|---|---|
| Accuracy | 86.25% | 98.90% | +12.65 |
| Precision | 100.0% | 96.10% | −3.90 |
| Recall | 31.25% | 98.50% | +67.25 |
| F1 | 47.62% | 97.28% | +49.66 |
| ROC-AUC | 93.56% | 99.97% | +6.41 |

**Headline:** the StyleGAN-only detector caught only ~31% of unseen fake types; cross-training
on DFFD raised detection to ~98.5%, at a small cost in precision. This demonstrates — and then
closes — the cross-generator generalization gap.

## Resources

- **Slide decks & model checkpoints (Google Drive):**
  https://drive.google.com/drive/folders/1amhEce0aIVq-DzevKJ5L9bCc9O0ch7u3?usp=sharing
  - `Deepfake_Detection_Week1.pptx`, `Deepfake_Detection_Week2.pptx`
  - `best_model.pt` — Week 1 model (trained on 140k)
  - `best_model_dffd.pt` — Week 2 model (fine-tuned on DFFD)
- **Colab notebook (full pipeline):**
  https://colab.research.google.com/drive/19tMWfnxbN_tG36yjfj2rDDGAlxqrH0bP?usp=sharing

## Datasets

- **140k Real and Fake Faces** (Kaggle: `xhlulu/140k-real-and-fake-faces`) — 70k real (FFHQ)
  + 70k fake (StyleGAN). Pre-split into train/valid/test. Used for Week 1.
- **DFFD — Diverse Fake Face Dataset** (MSU) — real faces from FFHQ plus fakes from FaceApp
  and StarGAN (attribute manipulation) and PGGAN and StyleGAN (entire synthesis). Used for
  Week 2. Download requires the dataset's HTTP credentials (username `dffd_dataset` + the
  password issued by MSU).

## Models

Both checkpoints are PyTorch dictionaries: `{model_name, state_dict, img_size, val_acc}`.

```python
import torch, timm

ck = torch.load("best_model_dffd.pt", map_location="cpu")   # or best_model.pt
model = timm.create_model(ck["model_name"], pretrained=False, num_classes=2)
model.load_state_dict(ck["state_dict"])
model.eval()
# class index: fake = 0, real = 1
# preprocessing: resize to ck["img_size"], normalize with mean/std = 0.5
```

## How to run

Open the Colab notebook (T4 GPU) and run the cells in order:

1. Install dependencies, mount Google Drive, load the model checkpoint.
2. Download and prepare the dataset into a `train/valid/test` ImageFolder layout
   (`real/` and `fake/` subfolders).
3. Evaluate the baseline -> fine-tune -> re-evaluate, printing a before/after comparison.

Training config: AdamW, cross-entropy, mixed precision. Week 1 fine-tuning at lr 2e-5 (3 epochs);
Week 2 fine-tuning continues from `best_model.pt` at lr 1e-5 (3 epochs).

## Status

- [x] Week 1 — baseline ViT on 140k (StyleGAN)
- [x] Week 2 — cross-dataset evaluation + DFFD fine-tuning

## Notes

- High in-domain accuracy (Week 1) does **not** imply real-world robustness — a detector trained
  on one generator can silently miss fakes from another. Cross-dataset testing is essential.
- This is a research/coursework project; the models are not production-hardened.
