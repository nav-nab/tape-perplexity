"""Stage 5: train one seed, and Stage 6: choose the seed.

Training recipe (docs/project-design.md, section 5):

* One pass over the training stream, no validation set. Each batch is new data, so the
  running loss on a batch is an honest held-out estimate.
* The stream is cut into non-overlapping chunks of ``context`` events and the chunks are
  shuffled (the order is fixed by the seed). Chunk ``c`` feeds the model tokens
  ``[c*L, c*L + L)`` and asks it to predict ``[c*L + 1, c*L + L + 1)``, so every event
  in a used chunk is a prediction target once. The last partial batch of chunks (random
  chunks, because the order is shuffled) and the events after the last full chunk are
  left out.
* AdamW, peak learning rate 1e-3, short linear warm-up, cosine decay to 10% of the peak,
  weight decay 0.1 (on weight matrices and embeddings, not on biases or layer norms),
  gradient clipping at 1.0, about 16,000 tokens per step, no dropout.
* The *final training loss* is the mean batch loss over the last 10% of steps. The
  seed with the lowest value is the one used for every result. The test month plays no
  part in the choice.

Resuming: a checkpoint holds the model, the optimiser, the step number and the loss
history. Because the data order comes from the seed and there is no dropout, a resumed
run follows the same trajectory as an uninterrupted one (up to floating-point noise).
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..model import GPT, ModelConfig
from ..runtime import RunContext, read_json, torch_save, write_json
from ..tokeniser import Tokeniser
from .tokenise import data_path, tokeniser_path


def model_config(ctx: RunContext, tok: Tokeniser) -> ModelConfig:
    c = ctx.cfg
    return ModelConfig(vocab=tok.vocab, n_layer=c.n_layer, d_model=c.d_model, n_head=c.n_head,
                       context=c.context, mlp_ratio=c.mlp_ratio)


def seed_dir(ctx: RunContext, seed: int):
    return ctx.cfg.path("train", f"seed{seed}")


def final_path(ctx: RunContext, seed: int):
    return seed_dir(ctx, seed) / "final.pt"


def make_optimiser(model: GPT, cfg) -> torch.optim.AdamW:
    # Weight decay applies to weight matrices and embeddings, not to biases or layer norms.
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=cfg.peak_lr, betas=(0.9, 0.95))


def lr_at(step: int, steps: int, cfg) -> float:
    """Linear warm-up, then cosine decay down to ``min_lr_frac`` of the peak."""
    warm = min(cfg.warmup_steps, max(1, steps // 10))
    if step < warm:
        return cfg.peak_lr * (step + 1) / warm
    progress = (step - warm) / max(1, steps - warm)
    floor = cfg.peak_lr * cfg.min_lr_frac
    return floor + 0.5 * (cfg.peak_lr - floor) * (1 + math.cos(math.pi * progress))


def run_seed(ctx: RunContext, seed: int) -> None:
    cfg = ctx.cfg
    tok = Tokeniser.load(tokeniser_path(ctx))
    mcfg = model_config(ctx, tok)
    tokens = np.load(data_path(ctx, "train", "tokens"), mmap_mode="r")
    L = cfg.context
    n_chunks = (len(tokens) - 1) // L
    B = max(1, cfg.tokens_per_step // L)           # chunks per optimiser step
    steps = n_chunks // B                           # the last partial batch is dropped
    if steps < 1:
        raise RuntimeError("training stream too short for one step")
    order = np.random.default_rng(seed).permutation(n_chunks)   # fixed by the seed

    d = seed_dir(ctx, seed)
    ckpt = d / "ckpt.pt"
    torch.manual_seed(seed)                         # fixes the initial weights
    model = GPT(mcfg)
    opt = make_optimiser(model, cfg)
    step, train_seconds = 0, 0.0
    loss_hist = np.zeros(steps, dtype=np.float64)
    pos_sum = np.zeros(L, dtype=np.float64)         # summed loss at each position in the chunk

    if ckpt.exists():                               # ---- resume ----
        s = torch.load(ckpt, weights_only=False)
        if s["fingerprint"] != cfg.fingerprint() or s["steps"] != steps:
            raise SystemExit(f"checkpoint {ckpt} was made with different settings; run with --reset")
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        step, train_seconds = s["step"], s["train_seconds"]
        loss_hist, pos_sum = s["loss_hist"], s["pos_sum"]
        ctx.log(f"seed {seed}: resumed from checkpoint at step {step}/{steps}")

    def save() -> None:
        torch_save(ckpt, dict(model=model.state_dict(), opt=opt.state_dict(), step=step, steps=steps,
                              train_seconds=train_seconds + (time.monotonic() - t_session),
                              loss_hist=loss_hist, pos_sum=pos_sum, fingerprint=cfg.fingerprint()))

    ctx.log(f"seed {seed}: {model.n_params():,} parameters, {n_chunks:,} chunks, "
            f"{steps} steps of {B} chunks ({B * L:,} tokens)")
    t_session, tok_session = time.monotonic(), 0
    model.train()
    while step < steps:
        # ---- one optimiser step ----
        idx = order[step * B:(step + 1) * B]
        batch = np.stack([tokens[c * L: c * L + L + 1] for c in idx]).astype(np.int64)
        x, y = torch.from_numpy(batch[:, :-1]), torch.from_numpy(batch[:, 1:])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, cfg)
        logits = model(x)
        per_token = F.cross_entropy(logits.reshape(-1, mcfg.vocab), y.reshape(-1),
                                    reduction="none").view(B, L)
        loss = per_token.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        # ---- bookkeeping ----
        loss_hist[step] = loss.item()
        pos_sum += per_token.detach().mean(0).double().numpy()
        step += 1
        tok_session += B * L
        rate = tok_session / max(time.monotonic() - t_session, 1e-9)
        recent = loss_hist[max(0, step - 20):step].mean()
        ctx.progress.update(step / steps, f"seed {seed} step {step}/{steps} loss {recent:.4f} {rate / 1000:.1f}k tok/s")
        ctx.maybe_checkpoint(save)                 # every 10 minutes, or at once on interrupt

    # ---- finished: final loss, weights and summary ----
    tail = max(1, int(round(cfg.final_loss_tail * steps)))
    final_loss = float(loss_hist[-tail:].mean())
    train_seconds += time.monotonic() - t_session
    torch_save(final_path(ctx, seed), dict(state_dict=model.state_dict(), model_config=mcfg.to_dict(),
                                           seed=seed, final_loss=final_loss, steps=steps))
    write_json(d / "summary.json", dict(
        seed=seed, final_loss=final_loss, steps=steps, tokens=steps * B * L, train_seconds=train_seconds,
        parameters=model.n_params(),
        # Loss at each position within a chunk. Position 0 has no history, and the loss
        # should flatten once there is enough: this checks that scored events (which have
        # 448 to 511 events of history) sit on the flat part of the curve.
        loss_by_position=(pos_sum / steps).tolist(),
        loss_first_steps=loss_hist[:5].tolist()))
    if ckpt.exists():
        ckpt.unlink()                               # the final weights supersede it
    ctx.log(f"seed {seed}: final training loss {final_loss:.4f} nats after {steps} steps "
            f"({train_seconds / 60:.1f} min of training)")


def make_train_stage(seed: int):
    def stage(ctx: RunContext) -> None:
        run_seed(ctx, seed)
    return stage


def select_seed(ctx: RunContext) -> None:
    """Choose the seed with the lowest final training loss (the test month plays no part)."""
    cfg = ctx.cfg
    losses = {s: read_json(seed_dir(ctx, s) / "summary.json")["final_loss"] for s in cfg.seeds}
    best = min(losses, key=lambda s: (losses[s], s))
    spread = max(losses.values()) - min(losses.values())
    write_json(cfg.path("selected.json"), dict(seed=best, final_losses={str(k): v for k, v in losses.items()},
                                               spread=spread))
    ctx.log("seed selection: " + ", ".join(f"seed {s}: {v:.4f}" for s, v in losses.items())
            + f" -> using seed {best}")
    if spread > cfg.seed_loss_warn:
        ctx.log(f"WARNING: final losses differ by {spread:.4f} nats (more than {cfg.seed_loss_warn}). "
                f"The design says to discuss this before relying on a single seed.")
