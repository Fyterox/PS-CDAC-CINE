"""
freq_encoder.py — spectral-artifact stream.

Why: GAN/diffusion generators leave periodic grid artifacts and spectral
signatures (from up-sampling layers) that are invisible in RGB but obvious in
the Fourier domain. This stream converts the image to a log-magnitude FFT
spectrum and learns over it with a small CNN.

Input:  raw [0,1] RGB image, [B,3,224,224]
Output: token sequence [B, Nf, D_out]

Implementation notes
--------------------
* FFT is computed in float32 with autocast disabled — torch.fft is unreliable
  under fp16 AMP. This is a correctness fix, not an optimisation.
* We use fftshift so the DC term sits in the centre (standard spectrum layout).
* Per-channel spectrum (3 channels) keeps the CNN's default in_chans=3.
"""
import torch
import torch.nn as nn
import timm


class FrequencyEncoder(nn.Module):
    def __init__(self, out_dim=512, backbone="resnet18", pretrained=True):
        super().__init__()
        # features_only gives us the last spatial feature map to tokenise.
        self.cnn = timm.create_model(backbone, pretrained=pretrained,
                                     num_classes=0, global_pool="")
        self.feat_dim = self.cnn.num_features           # 512 for resnet18
        self.proj = nn.Linear(self.feat_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    @staticmethod
    def _to_spectrum(x):
        """x: [B,3,H,W] in [0,1]. Returns log-magnitude spectrum, same shape."""
        # Do the transform in float32 regardless of AMP context.
        with torch.autocast(device_type="cuda", enabled=False):
            xf = x.float()
            fft = torch.fft.fft2(xf, norm="ortho")
            fft = torch.fft.fftshift(fft, dim=(-2, -1))
            mag = torch.log1p(torch.abs(fft))
            # Per-sample standardisation stabilises training across images.
            mu = mag.mean(dim=(-2, -1), keepdim=True)
            sd = mag.std(dim=(-2, -1), keepdim=True) + 1e-6
            mag = (mag - mu) / sd
        return mag.to(x.dtype)

    def forward(self, x):
        spec = self._to_spectrum(x)                     # [B,3,H,W]
        fmap = self.cnn(spec)                            # [B,C,h,w]
        B, C, h, w = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)         # [B, h*w, C]
        tokens = self.norm(self.proj(tokens))            # [B, h*w, out_dim]
        return tokens
