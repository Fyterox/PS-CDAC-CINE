"""
detector.py — assemble the three encoders + fusion into one model.

forward(x) -> (logits [B,2], aux_logits {stream: [B,2]})
The aux dict is used only for the deep-supervision loss during training; at
inference only `logits` is used.
"""
import torch
import torch.nn as nn

from .rgb_encoder import RGBEncoder
from .freq_encoder import FrequencyEncoder
from .noise_encoder import NoiseResidualEncoder
from .fusion import CrossAttentionFusion


class MultiStreamDeepfakeDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.rgb = RGBEncoder(
            model_name=cfg.get("rgb_model", "vit_base_patch16_224"),
            pretrained=cfg.get("rgb_pretrained", True),
            ckpt_path=cfg.get("rgb_ckpt", None),
            backend=cfg.get("rgb_backend", "timm"),
        )
        self.freq = FrequencyEncoder(
            out_dim=cfg.get("stream_dim", 512),
            backbone=cfg.get("cnn_backbone", "resnet18"),
            pretrained=cfg.get("cnn_pretrained", True),
        )
        self.noise = NoiseResidualEncoder(
            out_dim=cfg.get("stream_dim", 512),
            backbone=cfg.get("cnn_backbone", "resnet18"),
            pretrained=cfg.get("cnn_pretrained", True),
            use_bayar=cfg.get("use_bayar", True),
        )
        dims = {
            "rgb":   self.rgb.embed_dim,
            "freq":  cfg.get("stream_dim", 512),
            "noise": cfg.get("stream_dim", 512),
        }
        self.fusion = CrossAttentionFusion(
            dims=dims,
            fusion_dim=cfg.get("fusion_dim", 512),
            depth=cfg.get("fusion_depth", 3),
            heads=cfg.get("fusion_heads", 8),
            num_classes=2,
            dropout=cfg.get("dropout", 0.1),
        )

    def forward(self, x, return_aux=True):
        streams = {
            "rgb":   self.rgb(x),
            "freq":  self.freq(x),
            "noise": self.noise(x),
        }
        logits, aux = self.fusion(streams, return_aux=return_aux)
        return logits, aux

    @torch.no_grad()
    def predict_proba(self, x):
        logits, _ = self.forward(x, return_aux=False)
        return torch.softmax(logits, dim=1)[:, 1]   # P(fake)


def build_model(cfg):
    return MultiStreamDeepfakeDetector(cfg)


# --------------------------------------------------------------------------- #
# Single-stream classifier — used only for Stage-1 warm-starting the freq /
# noise encoders before fusion. Wraps one encoder + a mean-pool linear head.
# --------------------------------------------------------------------------- #
class SingleStreamClassifier(nn.Module):
    def __init__(self, encoder, in_dim, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.norm = nn.LayerNorm(in_dim)
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        tok = self.encoder(x)              # [B,N,D]
        feat = self.norm(tok.mean(dim=1))  # mean-pool tokens
        return self.head(feat)
