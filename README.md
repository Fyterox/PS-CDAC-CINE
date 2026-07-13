# Multi-Stream Deepfake Detection

Deepfake detector that combines three views of an image (RGB, frequency spectrum, and noise
residuals) instead of relying on a single backbone. Built during my Practice School-I
project at C-DAC.

## Why I built this

I started with the obvious thing: fine-tune a ViT-Base/16 to classify real vs fake faces.
On a 140K-image dataset it hit over 99% accuracy and a ROC-AUC around 1.0. Looked solved.

Then I tested it on generators it hadn't seen, using the Diverse Fake Face Dataset (DFFD).
Recall on fakes dropped to about 30%. The model had learned to recognize *those particular
fakes*, not fakes in general.

Fine-tuning on DFFD fixed most of it, but one category kept failing: FaceApp. It took me a
while to see why. FaceApp doesn't generate a face from scratch, it edits a real one — it
changes a smile or an age or a hairline and leaves the rest of the photo alone. So there's
no global weirdness for a semantic model to latch onto. The image basically *is* a real
photo, with a small region that isn't.

That's what pushed me toward multiple streams. If the global view can't see the edit, give
the model views that can:

| Stream | What it looks at | What it's good at |
|---|---|---|
| RGB (ViT-B/16) | Semantics, overall appearance | Fully synthesized faces (StyleGAN, PGGAN) |
| Frequency (FFT → CNN) | Log-magnitude spectrum | Upsampling grids and GAN fingerprints |
| Noise residual (SRM + Bayar → CNN) | Local high-pass residuals | FaceApp-style local edits, splices |

The three encoders each produce a sequence of tokens, and a cross-attention transformer
lets them attend to each other before the classifier sees anything. There's also an
auxiliary head on each stream during training, which I added after watching the pretrained
RGB stream steamroll the two randomly-initialized ones.

## Results

Held-out test set (53,381 images), DFFD's official identity-disjoint split:

| Metric | |
|---|---|
| Fake recall | 0.992 |
| Precision | 0.995 |
| F1 | 0.993 |
| ROC-AUC | 0.999 |
| Accuracy | 0.989 |

Broken down by generator:

| Generator | n | Caught as fake |
|---|---|---|
| StarGAN | 21,913 | 1.000 |
| PGGAN | 8,970 | 1.000 |
| StyleGAN | 8,997 | 0.997 |
| FaceApp | 4,501 | 0.929 |
| real faces | 9,000 | 0.026 wrongly flagged |

The number I care about is fake recall, because the whole problem was fakes slipping
through. It went 0.29 (ViT on unseen generators) → 0.79 (same ViT, fine-tuned on DFFD) →
**0.99** with the three streams fused. FaceApp, the category that motivated the whole
design, went from being the model's blind spot to 0.93.

Two things worth writing down anyway.

FaceApp is still the weakest row, and that's not surprising — it's the only manipulation here
that edits a real photo instead of generating one, so there's simply less signal to find. The
other three generators are essentially saturated.

The other is a training result that surprised me: **unfreezing the RGB backbone made things
worse.** My first run used a staged schedule that froze the ViT for four epochs then
fine-tuned everything end-to-end. The model peaked at epoch 1, while frozen. Once I unfroze
it, calibration fell apart — at one point it was flagging 46% of *real* faces as fake — and
it never recovered its earlier score. So the recipe below keeps RGB frozen the whole way,
which is both better and about three times faster.

## What's in here

```
common.py               metrics, seeding, checkpoint loading, per-generator reporting
data.py                 dataset and transforms
build_manifest_dffd.py  builds a manifest from DFFD's folder layout
models/
  rgb_encoder.py        ViT-B/16, loads a fine-tuned checkpoint
  freq_encoder.py       FFT spectrum → CNN → tokens
  noise_encoder.py      SRM + Bayar high-pass → CNN → tokens
  fusion.py             cross-attention transformer, aux heads
  detector.py           puts the whole thing together
pretrain_stream.py      warm-starts the frequency / noise streams
train.py                trains the fusion model
evaluate.py             metrics + the per-generator table
explain.py              saliency maps, FFT/SRM views, per-stream scores
app/                    small UI — drop in an image, get a verdict
```

## Getting the data

DFFD isn't a public download. You request access from
[MSU's CVLab page](https://cse.msu.edu/computervision/dffd_dataset) and they send you
credentials. I'm not redistributing it here — the licenses on the underlying datasets
(FFHQ, CelebA, FaceForensics++, and the GAN sources) don't allow that.

Once you have it, the folders look like this — generator first, then split:

```
DFFD/
  ffhq/{train,validation,test}/            real
  faceapp/{train,validation,test}/         fake
  stargan/{train,validation,test}/
  pggan_v1/{train,validation,test}/
  stylegan_ffhq/{train,validation,test}/
```

Use those train/validation/test folders as they are. They're DFFD's official split and,
importantly, the identities don't overlap between them. If you shuffle the images yourself
and make your own split, the same person's face can land in both train and test and your
recall numbers will look better than they deserve to.

One practical note: MSU's server throttles each connection to something like 85 KB/s. I
wasted the better part of an hour on `wget` before switching to `aria2c -x16 -s16 -c`, which
pulled the same file in under a minute.

## Running it

```bash
pip install -r requirements.txt

# manifest
python build_manifest_dffd.py --root /path/to/DFFD --out manifest.csv

# warm-start the two new streams (skip if you already have the checkpoints)
python pretrain_stream.py --stream freq  --manifest manifest.csv \
    --epochs 5 --batch-size 64 --out runs/freq_pretrain/best.pt
python pretrain_stream.py --stream noise --manifest manifest.csv \
    --epochs 5 --batch-size 64 --out runs/noise_pretrain/best.pt

# fusion, RGB frozen throughout
python train.py \
  --manifest manifest.csv \
  --rgb-ckpt   ckpts/best_model_dffd.pt \
  --freq-init  runs/freq_pretrain/best.pt \
  --noise-init runs/noise_pretrain/best.pt \
  --epochs 4 --freeze-rgb-epochs 4 \
  --batch-size 16 --accum 2 --base-lr 3e-4 \
  --monitor recall --out runs/fusion/best.pt

python evaluate.py --manifest manifest.csv --ckpt runs/fusion/best.pt
python explain.py  --manifest manifest.csv --ckpt runs/fusion/best.pt --n 8 --out-dir runs/explain
```

`--rgb-ckpt` is your DFFD-fine-tuned ViT. When it loads, it prints how many backbone keys
were missing. That should be 0. If it's a big number, your checkpoint is probably in
HuggingFace format, in which case add
`--rgb-backend hf --rgb-model google/vit-base-patch16-224-in21k`.

I haven't committed the weights (the ViT alone is 343 MB, which GitHub won't take). Grab
them from Releases or train your own.

## A few decisions I'd defend

**The dataset hands the model raw `[0,1]` images, and each encoder does its own
preprocessing.** This looks like an oversight but isn't. If you ImageNet-normalize before
the FFT or the SRM filters, you distort the exact signal those streams exist to read. The
RGB stream normalizes internally.

**SRM filters are fixed, Bayar is learnable.** The SRM kernels come from the steganalysis
literature and carry a strong prior about what noise residuals look like; the Bayar conv
learns a data-driven high-pass on top of that. Using both beat using either alone.

**Cross-attention rather than late fusion.** I could have pooled each stream to a vector and
concatenated. Letting the streams attend to each other's tokens before the decision is more
expressive, and it means `explain.py` can tell you which stream actually drove a given
prediction.

**Recall is the headline number, not accuracy.** The entire problem here is fakes that slip
through. Accuracy hides that behind a large real-image class.

## Things that bit me

The Bayar convolution NaN'd on the first batch, every time. The constraint step divides the
kernel by the sum of its surrounding weights, and at default init that sum can be near zero,
so the weights explode. Fixed with a positive init and a guarded denominator. Gradient
clipping is on now too.

FFT and SRM run in fp32 even with AMP enabled. fp16 FFT is unreliable and this is
deliberate, so don't "optimize" it away.

PyTorch 2.6 flipped `torch.load` to `weights_only=True` by default, which rejects any
checkpoint carrying NumPy scalars in its metadata. Everything here passes
`weights_only=False`.

And the one that actually cost me a day: Colab's `/content` is wiped when the runtime
recycles. I lost a finished 12-epoch run to an idle timeout. Write your checkpoints to
Drive.

## What's next

FaceApp at 0.93 is the obvious place to keep pushing, since it's the only generator that
isn't saturated. The lever I'd try first is the auxiliary loss weight (0.3 → around 0.6),
which should stop the pretrained RGB stream from dominating the fused representation and let
the noise-residual stream — the one actually built for local edits — carry more of the
decision.

After that: extend to diffusion-generated faces, which should just be a new folder and a tag
since nothing in the pipeline is generator-specific, and then video benchmarks
(FaceForensics++, Celeb-DF) by extracting frames into the same manifest schema.

## Credit

DFFD comes from Dang et al., *On the Detection of Digital Face Manipulation*, CVPR 2020. If
you use it, cite them and the underlying dataset sources.
