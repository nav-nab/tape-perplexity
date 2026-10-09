"""Stage 7: score the test month with the selected, frozen model.

Every test event gets a surprisal  s_t = -log p(x_t | previous events)  in nats.

Why sliding windows (docs/project-design.md, section 6): the model reads at most
``context`` (512) events. If the test stream were cut into plain 512-event chunks, the
first event of a chunk would have no history and the last would have 511, and surprisal
falls as history grows, so the series would show a spurious sawtooth. Instead a window of
``context`` events slides over the stream with a stride of ``stride`` (64) events, and
only the last ``stride`` events of each window are scored. Every scored event then has
between ``context - stride`` (448) and ``context - 1`` (511) events of history.

Layout. Let ``H = context - 1`` be the number of context events taken from the end of
the training period (they are history only and are never scored). The extended stream is

    ext = [last H training events] + [test events]

Window ``k`` ends (exclusive) at ``end_k = H + stride * (k + 1)`` (the last window is cut
at the end of the stream) and starts ``context`` events earlier. It scores the events
``[end_k - stride, end_k)``. The model is fed the window without its last event and
predicts the window shifted by one, so the event at window index ``i`` is predicted from
``i`` events of history.

Resuming: every 10 minutes the surprisals computed so far and the next window number are
saved.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from ..model import GPT, ModelConfig
from ..runtime import RunContext, read_json, save_npz
from .tokenise import data_path
from .train import final_path


def surprisal_path(ctx: RunContext):
    return ctx.cfg.path("data", "surprisal.npy")


def partial_path(ctx: RunContext):
    return ctx.cfg.path("score", "partial.npz")


def load_model(path) -> GPT:
    """Rebuild a model from a saved ``final.pt`` (state dict plus its settings)."""
    blob = torch.load(path, weights_only=False)
    model = GPT(ModelConfig(**blob["model_config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def score_stream(model: GPT, ext: np.ndarray, n_ctx: int, stride: int, batch_windows: int,
                 start_window: int = 0, out: np.ndarray | None = None, on_batch=None) -> np.ndarray:
    """Surprisal of ``ext[n_ctx:]`` using sliding windows. Pure function of its inputs.

    ``on_batch(next_window, out)`` is called after each batch (for progress and checkpoints).
    """
    L = model.c.context
    n_test = len(ext) - n_ctx
    n_windows = -(-n_test // stride)                       # ceiling division
    if out is None:
        out = np.zeros(n_test, dtype=np.float32)
    offs = np.arange(L - 1)                                # the window minus its last event
    with torch.inference_mode():
        k = start_window
        while k < n_windows:
            ks = np.arange(k, min(k + batch_windows, n_windows))
            ends = np.minimum(n_ctx + stride * (ks + 1), len(ext))
            starts = ends - L
            idx = starts[:, None] + offs[None, :]
            x = torch.from_numpy(ext[idx].astype(np.int64))
            # Window event i (i = L - stride .. L - 1) is predicted at position i - 1.
            logits = model(x, last_k=stride)               # positions L-1-stride .. L-2
            logp = F.log_softmax(logits.float(), dim=-1)
            # The targets are the last `stride` events of each window. For the final window
            # `ends` is cut at the end of the stream, and only its new events are kept below.
            tidx = ends[:, None] - stride + np.arange(stride)[None, :]
            targets = torch.from_numpy(ext[tidx].astype(np.int64))
            ll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).numpy()   # (W, stride)
            for w, kk in enumerate(ks):
                a, e = n_ctx + stride * kk, ends[w]         # new events [a, e) of this window
                out[a - n_ctx:e - n_ctx] = ll[w, stride - (e - a):]
            k = int(ks[-1]) + 1
            if on_batch is not None:
                on_batch(k, out)
    return out


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    sel = read_json(cfg.path("selected.json"))["seed"]
    model = load_model(final_path(ctx, sel))
    train = np.load(data_path(ctx, "train", "tokens"), mmap_mode="r")
    test = np.load(data_path(ctx, "test", "tokens"), mmap_mode="r")
    n_ctx = cfg.context - 1
    ext = np.concatenate((np.asarray(train[-n_ctx:]), np.asarray(test)))
    n_test = len(test)
    n_windows = -(-n_test // cfg.stride)
    ctx.log(f"score: seed {sel}, {n_test:,} test events, {n_windows:,} windows of {cfg.context} "
            f"(stride {cfg.stride})")

    k0, out = 0, np.zeros(n_test, dtype=np.float32)
    if partial_path(ctx).exists():                         # ---- resume ----
        s = np.load(partial_path(ctx))
        if str(s["fingerprint"]) == cfg.fingerprint():
            k0, out = int(s["next_window"]), s["surprisal"].copy()
            ctx.log(f"score: resumed at window {k0:,}/{n_windows:,}")
    t0, w0 = time.monotonic(), k0

    def on_batch(next_window: int, buf: np.ndarray) -> None:
        rate = (next_window - w0) / max(time.monotonic() - t0, 1e-9)
        ctx.progress.update(next_window / n_windows,
                            f"window {next_window:,}/{n_windows:,} {rate * cfg.context / 1000:.1f}k tok/s")
        ctx.maybe_checkpoint(lambda: save_npz(partial_path(ctx), surprisal=buf, next_window=next_window,
                                              fingerprint=np.array(cfg.fingerprint())))

    score_stream(model, ext, n_ctx, cfg.stride, cfg.score_batch_windows, k0, out, on_batch)
    np.save(surprisal_path(ctx), out)
    if partial_path(ctx).exists():
        partial_path(ctx).unlink()
    ctx.log(f"score: mean surprisal {out.mean():.4f} nats over {n_test:,} test events")
