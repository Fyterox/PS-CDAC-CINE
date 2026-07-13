"""
data.py — DFFD dataset + transforms.

Design decision that matters: the dataset returns the image in RAW [0,1] space,
NOT ImageNet-normalised. Each encoder then applies the preprocessing it needs:

  * RGB (ViT)      -> ImageNet mean/std normalisation (done inside RGBEncoder)
  * Frequency      -> FFT log-magnitude on the raw signal (normalising first
                      would distort the spectrum we care about)
  * Noise-residual -> SRM high-pass filters on the raw signal

Manifest format (CSV): columns = path,label,generator,split
  label: 0 real / 1 fake
  generator: one of common.GENERATORS
  split: train / val / test
"""
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

from common import GEN2ID


class RandomJPEG:
    """Robustness augmentation: re-encode as JPEG at a random quality.

    Deepfake fingerprints partly live in high frequencies that JPEG attacks,
    so training with mild JPEG teaches the frequency/noise streams to rely on
    compression-robust cues. Keep quality high-ish to avoid erasing the signal.
    """
    def __init__(self, p=0.5, qmin=60, qmax=95):
        self.p, self.qmin, self.qmax = p, qmin, qmax

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        import io
        q = random.randint(self.qmin, self.qmax)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def build_transforms(img_size=224, train=True, robust_aug=True):
    ops = [T.Resize((img_size, img_size))]
    if train:
        ops.append(T.RandomHorizontalFlip(p=0.5))
        if robust_aug:
            # Mild real-world perturbations. Deliberately gentle — aggressive
            # blur/compression destroys generator fingerprints.
            ops.append(RandomJPEG(p=0.5, qmin=60, qmax=95))
            ops.append(T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2))
    ops.append(T.ToTensor())  # -> [0,1], shape [3,H,W]
    return T.Compose(ops)


class DFFDDataset(Dataset):
    def __init__(self, manifest_csv, split, img_size=224, train=None, robust_aug=True):
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split='{split}' in {manifest_csv}")
        is_train = (split == "train") if train is None else train
        self.tf = build_transforms(img_size, train=is_train, robust_aug=robust_aug)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        x = self.tf(img)                          # raw [0,1]
        y = int(row["label"])                     # 0 real / 1 fake
        g = GEN2ID.get(str(row["generator"]).lower(), 0)
        return x, y, g


def make_loaders(manifest_csv, img_size=224, batch_size=32, num_workers=2,
                 robust_aug=True):
    train_ds = DFFDDataset(manifest_csv, "train", img_size, robust_aug=robust_aug)
    val_ds   = DFFDDataset(manifest_csv, "val",   img_size, robust_aug=False)
    common = dict(num_workers=num_workers, pin_memory=True)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          drop_last=True, **common)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    return train_ld, val_ld


def make_test_loader(manifest_csv, split="test", img_size=224, batch_size=64, num_workers=2):
    ds = DFFDDataset(manifest_csv, split, img_size, train=False, robust_aug=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
