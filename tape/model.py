"""The causal transformer (a small GPT-style decoder).

Design (from the design notes): 6 layers, width 192, 6 heads, learned positions,
pre-norm blocks, GELU, tied input and output embeddings, no dropout, context 512.
With a vocabulary of up to 224 tokens this has about 2.8 million parameters.

The only unusual feature is ``last_k`` in ``forward``: when scoring we need the
next-token distribution at only the final few positions of each window, so we can skip
the output layer everywhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab: int
    n_layer: int = 6
    d_model: int = 192
    n_head: int = 6
    context: int = 512
    mlp_ratio: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


class Block(nn.Module):
    """One pre-norm transformer block: attention then MLP, each with a residual."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        assert c.d_model % c.n_head == 0
        self.n_head = c.n_head
        self.ln1 = nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.fc = nn.Linear(c.d_model, c.mlp_ratio * c.d_model)
        self.fc_out = nn.Linear(c.mlp_ratio * c.d_model, c.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        # Reshape to (batch, heads, time, head_dim) for scaled dot-product attention.
        q, k, v = (z.view(b, t, self.n_head, d // self.n_head).transpose(1, 2) for z in (q, k, v))
        # is_causal=True masks out future positions: this is what makes the model causal.
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.fc_out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d_model)
        self.pos = nn.Embedding(c.context, c.d_model)       # learned positions
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        self.head.weight = self.tok.weight                    # tied embeddings
        self.apply(self._init)
        # Scale the output projection of each residual branch down, so the variance of
        # the residual stream does not grow with depth (standard GPT-2 practice).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, last_k: int | None = None) -> torch.Tensor:
        """Logits for the next token at each position (or at the last ``last_k``)."""
        t = idx.shape[1]
        assert t <= self.c.context, f"sequence of {t} exceeds context {self.c.context}"
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        if last_k is not None:
            x = x[:, -last_k:]
        return self.head(x)

    def n_params(self) -> int:
        # The tied output weight is the same tensor as the embedding, counted once.
        return sum(p.numel() for p in self.parameters())
