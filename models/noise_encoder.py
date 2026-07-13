"""
noise_encoder.py — noise-residual / local-tamper stream.

Why: FaceApp-style edits change only PART of a real face. They break the local
noise/sensor consistency of the image (splicing boundaries, re-rendered patches)
even when global appearance is convincing. This is the single hardest case for a
semantic/global model, and the reason this stream exists.
Steganalysis-style high-pass residuals expose those local inconsistencies.

Front-end:
  * 3 fixed SRM high-pass kernels (from Zhou et al., RGB-N) suppress image
    content and surface noise residuals.
  * an optional LEARNABLE Bayar constrained conv adapts a data-driven high-pass.
Both feed a small CNN that produces tokens.

Input:  raw [0,1] RGB image, [B,3,224,224]
Output: token sequence [B, Nn, D_out]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _srm_kernels():
    """The three canonical SRM high-pass filters (5x5), normalised."""
    k1 = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32) / 4.0
    k2 = torch.tensor([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ], dtype=torch.float32) / 12.0
    k3 = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32) / 2.0
    return torch.stack([k1, k2, k3], dim=0)  # [3,5,5]


class SRMConv(nn.Module):
    """Fixed 3-kernel SRM front-end. Produces a 3-channel residual map."""
    def __init__(self):
        super().__init__()
        ker = _srm_kernels()                      # [3,5,5]
        # out_ch=3, in_ch=3: kernel k applied to each RGB channel then averaged.
        w = ker.unsqueeze(1).repeat(1, 3, 1, 1) / 3.0   # [3,3,5,5]
        self.register_buffer("weight", w)

    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            r = F.conv2d(x.float(), self.weight, padding=2)
        return r.to(x.dtype)


class BayarConv2d(nn.Module):
    """Bayar constrained conv: learnable high-pass whose kernel centre is -1 and
    the surrounding weights sum to +1 (re-projected after each step in training).

    NUMERICAL STABILITY (this bit matters — the naive version NaNs out):
    the projection divides by the sum of the surrounding weights. At default
    (zero-mean) init that sum can be ~0 or negative, so the division explodes the
    kernel, the conv output overflows, and the loss goes NaN on the very first
    batch. Two guards:
      1. positive uniform init  -> the denominator is well-conditioned from step 0
      2. a guarded denominator  -> never divide by a near-zero sum
    """
    def __init__(self, in_ch=3, out_ch=3, k=5):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False)
        with torch.no_grad():
            self.conv.weight.uniform_(0.0, 1.0)   # positive init
        self._constrain()

    def _constrain(self):
        with torch.no_grad():
            w = self.conv.weight.data
            c = self.k // 2
            w[:, :, c, c] = 0.0
            s = w.sum(dim=(2, 3), keepdim=True)
            s = torch.where(s.abs() < 1e-4, torch.ones_like(s), s)   # guard
            w = w / s
            w[:, :, c, c] = -1.0
            self.conv.weight.data = w.contiguous()

    def forward(self, x):
        if self.training:
            self._constrain()
        return self.conv(x)


class NoiseResidualEncoder(nn.Module):
    def __init__(self, out_dim=512, backbone="resnet18", pretrained=True,
                 use_bayar=True):
        super().__init__()
        self.srm = SRMConv()
        self.use_bayar = use_bayar
        in_ch = 3
        if use_bayar:
            self.bayar = BayarConv2d(3, 3, 5)
            in_ch = 6  # concat SRM(3) + Bayar(3)
        # A 1x1 to map residual channels to 3 so we can use a pretrained CNN stem.
        self.stem = nn.Conv2d(in_ch, 3, kernel_size=1)
        self.cnn = timm.create_model(backbone, pretrained=pretrained,
                                     num_classes=0, global_pool="")
        self.feat_dim = self.cnn.num_features
        self.proj = nn.Linear(self.feat_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        r = self.srm(x)
        if self.use_bayar:
            r = torch.cat([r, self.bayar(x)], dim=1)
        r = self.stem(r)
        fmap = self.cnn(r)                               # [B,C,h,w]
        B, C, h, w = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)          # [B, h*w, C]
        tokens = self.norm(self.proj(tokens))
        return tokens
