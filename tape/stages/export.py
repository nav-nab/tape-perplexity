"""Stage 9: export the small files that are committed to the repo.

The generation GUI needs three things, and none of them needs the downloaded data:

  models/model.pt             weights of the selected seed (about 11 MB), with the model
                              settings so the network can be rebuilt from the file alone
  models/tokeniser.json       the fitted tokeniser (size edges, merged bins, median)
  models/prompt_excerpts.json ten real test-month stretches (prompt plus what followed)

Excerpts are drawn at random with a fixed seed, half from volatile moments and half from
calm ones. "Volatile" means the realised volatility at the last prompt event exceeds the
training-period threshold. They are drawn by rule, not picked by eye, so the GUI does not
show only favourable examples. Excerpts never overlap, and every one lies entirely
inside the test month.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from ..runtime import RunContext, read_json, torch_save
from .tokenise import data_path, tokeniser_path
from .train import final_path
from .volatility import regime_path, rv_path


def _utc(us: int) -> str:
    return dt.datetime.fromtimestamp(us / 1e6, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"


def draw_excerpts(test: np.ndarray, t0: np.ndarray, flag: np.ndarray, rv_test: np.ndarray,
                  prompt_len: int, cont_len: int, per_kind: int, seed: int) -> list[dict]:
    """Choose non-overlapping excerpts: ``per_kind`` volatile and ``per_kind`` calm."""
    rng = np.random.default_rng(seed)
    n = len(test)
    ends = np.arange(prompt_len, n - cont_len + 1)             # prompt = test[i-P:i], continuation = test[i:i+C]
    taken: list[tuple[int, int]] = []
    out: list[dict] = []
    for kind, label in ((True, "volatile"), (False, "calm")):
        cand = ends[flag[ends - 1] == kind]                    # kind judged at the last prompt event
        found = 0
        for i in rng.permutation(cand)[:200_000]:
            lo, hi = int(i) - prompt_len, int(i) + cont_len
            if any(lo < b and a < hi for a, b in taken):       # overlaps an earlier excerpt
                continue
            taken.append((lo, hi))
            found += 1
            out.append(dict(id=f"{label}-{found}", kind=label, volatile=bool(kind),
                            rv_ticks=float(rv_test[i - 1]),
                            t_start=_utc(int(t0[lo])), t_split=_utc(int(t0[i])),
                            prompt=test[lo:i].astype(int).tolist(),
                            continuation=test[i:hi].astype(int).tolist()))
            if found == per_kind:
                break
    return out


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    out = Path(cfg.models_dir)
    out.mkdir(parents=True, exist_ok=True)
    sel = read_json(cfg.path("selected.json"))["seed"]
    blob = torch.load(final_path(ctx, sel), weights_only=False)
    torch_save(out / "model.pt", blob)
    shutil.copyfile(tokeniser_path(ctx), out / "tokeniser.json")
    ctx.progress.update(0.4, "weights and tokeniser written")

    regime = read_json(regime_path(ctx))
    test = np.load(data_path(ctx, "test", "tokens"))
    t0 = np.load(data_path(ctx, "test", "t0"), mmap_mode="r")
    rv_test = np.load(rv_path(ctx), mmap_mode="r")[regime["n_train"]:]
    flag = np.asarray(rv_test) > regime["threshold_ticks"]
    ex = draw_excerpts(test, t0, flag, np.asarray(rv_test), cfg.context - 1, cfg.excerpt_continuation,
                       cfg.excerpts_per_kind, cfg.excerpt_seed)
    want = 2 * cfg.excerpts_per_kind
    if len(ex) < want:
        ctx.log(f"WARNING: only {len(ex)} of {want} excerpts could be drawn (too few volatile or calm stretches)")
    (out / "prompt_excerpts.json").write_text(json.dumps(
        dict(context=cfg.context, prompt_len=cfg.context - 1, continuation_len=cfg.excerpt_continuation,
             regime_threshold_ticks=regime["threshold_ticks"], seed=sel, excerpts=ex),
        separators=(",", ":")))
    size_mb = (out / "model.pt").stat().st_size / 1e6
    ctx.log(f"export: wrote {out}/model.pt ({size_mb:.1f} MB), tokeniser.json and {len(ex)} excerpts")
