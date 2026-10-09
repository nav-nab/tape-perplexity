"""Autoregressive generation: continue a sequence of events with the trained model.

Used by the GUI, and importable on its own. Given a prompt of token ids, the model's
next-token distribution is computed, one token is sampled and appended, and the process
repeats. When the sequence outgrows the model's context only the most recent ``context``
events are shown to it (a sliding window), exactly as in scoring.

Nothing here touches the study pipeline: it reads only the files in ``models/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .model import GPT, ModelConfig
from .tokeniser import Tokeniser


@dataclass
class Bundle:
    """Everything the GUI needs, loaded from the ``models`` directory."""
    model: GPT
    tokeniser: Tokeniser
    excerpts: dict           # contents of prompt_excerpts.json
    lot: float = 0.00001     # BTC per lot, to label size bins


def load_bundle(models_dir: str | Path) -> Bundle:
    d = Path(models_dir)
    blob = torch.load(d / "model.pt", weights_only=False)
    model = GPT(ModelConfig(**blob["model_config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    tok = Tokeniser.load(d / "tokeniser.json")
    assert tok.vocab == model.c.vocab, "tokeniser and weights do not belong together"
    excerpts = json.loads((d / "prompt_excerpts.json").read_text())
    return Bundle(model, tok, excerpts)


@torch.inference_mode()
def generate(model: GPT, prompt: list[int], n_new: int, temperature: float = 1.0,
             n_samples: int = 1, seed: int = 0) -> np.ndarray:
    """Sample ``n_new`` continuation tokens for each of ``n_samples`` independent samples.

    ``temperature`` 1 samples from the model's own distribution. Lower values sharpen it
    (towards the most likely token) and higher values flatten it. Temperature 0 is greedy.
    Returns an array of shape ``(n_samples, n_new)``.
    """
    if len(prompt) == 0:
        raise ValueError("the prompt is empty")
    if max(prompt) >= model.c.vocab or min(prompt) < 0:
        raise ValueError("prompt contains a token id outside the vocabulary")
    gen = torch.Generator().manual_seed(int(seed))
    seq = torch.tensor(prompt, dtype=torch.long).unsqueeze(0).repeat(n_samples, 1)
    for _ in range(n_new):
        window = seq[:, -model.c.context:]                   # sliding window over the latest events
        logits = model(window, last_k=1)[:, -1].float()      # next-token logits, shape (samples, vocab)
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen)
        seq = torch.cat((seq, nxt), dim=1)
    return seq[:, -n_new:].numpy()


def size_labels(tok: Tokeniser, lot: float = 0.00001) -> list[str]:
    """Human-readable BTC range of each final size bin, e.g. '0.001 to 0.0023 BTC'."""
    def fmt(x: float) -> str:
        return f"{x:.3g}"
    labels = []
    for j in range(tok.n_size):
        raw = tok.lo + j
        low = 0.0 if j == 0 else tok.edges[raw - 1] * lot
        high = None if j == tok.n_size - 1 else tok.edges[raw] * lot
        if j == 0:
            labels.append(f"under {fmt(high)} BTC")
        elif high is None:
            labels.append(f"over {fmt(low)} BTC")
        else:
            labels.append(f"{fmt(low)} to {fmt(high)} BTC")
    return labels
