"""
fusion.py — cross-attention fusion of the three streams.

Each encoder emits a token sequence. We:
  1. project every stream to a common fusion dimension,
  2. tag each token with a learnable STREAM-TYPE embedding (so the transformer
     knows which modality a token came from),
  3. prepend a learnable [FUSION] token,
  4. run L transformer layers. Self-attention over the concatenated multi-stream
     sequence lets every RGB token attend to every frequency/noise token and
     vice-versa. This is the cross-attention fusion step.
  5. read the final [FUSION] token into the classifier.

We also attach lightweight per-stream auxiliary heads. Training them with a
small weight (deep supervision) keeps each stream individually informative and
stops the strong RGB stream from dominating early — the failure mode that makes
multi-stream models collapse back to "just the ViT".
"""
import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, dims, fusion_dim=512, depth=3, heads=8, mlp_ratio=4.0,
                 num_classes=2, dropout=0.1):
        """dims: dict with keys 'rgb','freq','noise' -> incoming token dims."""
        super().__init__()
        self.proj = nn.ModuleDict({
            name: nn.Linear(d, fusion_dim) for name, d in dims.items()
        })
        self.stream_embed = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, 1, fusion_dim)) for name in dims
        })
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.trunc_normal_(self.fusion_token, std=0.02)
        for p in self.stream_embed.values():
            nn.init.trunc_normal_(p, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=heads,
            dim_feedforward=int(fusion_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(fusion_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(fusion_dim, num_classes)
        )
        # Auxiliary per-stream heads (deep supervision).
        self.aux_heads = nn.ModuleDict({
            name: nn.Linear(fusion_dim, num_classes) for name in dims
        })

    def forward(self, stream_tokens: dict, return_aux=True):
        B = next(iter(stream_tokens.values())).size(0)
        seq = [self.fusion_token.expand(B, -1, -1)]
        aux_logits = {}
        for name, tok in stream_tokens.items():
            t = self.proj[name](tok) + self.stream_embed[name]   # [B,N,D]
            seq.append(t)
            if return_aux:
                aux_logits[name] = self.aux_heads[name](t.mean(dim=1))
        x = torch.cat(seq, dim=1)                                # [B, 1+sum(N), D]
        x = self.encoder(x)
        fusion_feat = self.norm(x[:, 0])                         # [FUSION] token
        logits = self.head(fusion_feat)
        return logits, aux_logits
