"""
rgb_encoder.py — semantic / global stream.

A ViT-Base/16 initialised from a DFFD-fine-tuned checkpoint. It provides the
global "does this face look synthetic" view. The frequency and noise streams
exist to catch what this one misses: FaceApp-style local edits, which leave the
global appearance largely intact.

Input:  raw [0,1] RGB image, [B,3,224,224]
Output: token sequence [B, N+1, 768]  (CLS at index 0 + 196 patch tokens)

Checkpoint loading
------------------
--rgb-ckpt takes the fine-tuned weights. The flexible loader strips any old
classification head and reports missing/unexpected keys, so a clean load can be
verified: the missing-key count should be ~0. Works with a raw state_dict, a
{'model': ...} wrapper, or DataParallel 'module.' prefixes.

For a checkpoint trained with HuggingFace `transformers` rather than timm, set
backend='hf' (see the note at the bottom of this file).
"""
import torch
import torch.nn as nn
import timm

from common import load_state_dict_flexible

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RGBEncoder(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True,
                 ckpt_path=None, backend="timm"):
        super().__init__()
        self.backend = backend
        if backend == "timm":
            # num_classes=0 -> feature mode; forward_features returns tokens.
            self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                              num_classes=0)
            self.embed_dim = self.backbone.num_features
        elif backend == "hf":
            from transformers import ViTModel
            self.backbone = ViTModel.from_pretrained(model_name)
            self.embed_dim = self.backbone.config.hidden_size
        else:
            raise ValueError(backend)

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        if ckpt_path:
            self.load_finetuned(ckpt_path)

    def load_finetuned(self, ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = load_state_dict_flexible(self.backbone, sd)
        # Missing-key count is the check that the backbone actually loaded.
        n_missing = len([m for m in missing if "head" not in m])
        print(f"[RGBEncoder] loaded {ckpt_path}")
        print(f"[RGBEncoder]   backbone missing (non-head): {n_missing}")
        print(f"[RGBEncoder]   unexpected: {len(unexpected)}")
        if n_missing > 20:
            print("[RGBEncoder]   WARNING: many missing keys — is the backend/"
                  "model_name right? (timm vs hf naming differs)")

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        """Return token sequence [B, N+1, D]."""
        x = self._normalize(x)
        if self.backend == "timm":
            feats = self.backbone.forward_features(x)  # [B, N+1, D] for ViT
            if feats.ndim == 2:                        # some configs pool
                feats = feats.unsqueeze(1)
            return feats
        else:  # hf
            out = self.backbone(pixel_values=x)
            return out.last_hidden_state                # [B, N+1, D]


# --- Note on HuggingFace checkpoints ------------------------------------------
# In a checkpoint from transformers.ViTForImageClassification the encoder weights
# live under `vit.*`. Convert once:
#
#   from transformers import ViTForImageClassification
#   m = ViTForImageClassification.from_pretrained("path/to/ckpt")
#   torch.save(m.vit.state_dict(), "vit_backbone.pt")
#
# then pass backend='hf', model_name='google/vit-base-patch16-224-in21k',
# ckpt_path='vit_backbone.pt'. The timm path is the default and is simplest.
