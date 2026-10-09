"""Stage 11: the dependence check on the training period.

The block bootstrap needs blocks long enough to contain the dependence in the data
(docs/block-bootstrap.md). The block length of 1,000,000 events was chosen from the
autocorrelation of log RV alone. This stage repeats the check on the series the headline
statistic is actually built from: the rolling mean surprisal and log RV over the whole
training period, scored by the selected model (stage ``score_train``).

Two things are reported (``results/dependence_check.json`` and ``dependence_check.png``):

* the autocorrelation of both series at several lags, and
* the bootstrap standard error of the correlation for a range of block lengths.

If the standard error is still rising at 1,000,000 events, the chosen block is too
short; if it has levelled off before that, it is long enough. A rising curve does not
change the headline result by itself. It tells us how much to trust the interval.

Caveat: the model was trained on these events, so the surprisal here is in-sample. The
check is about how dependent the series are, not about the size of the correlation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..runtime import RunContext, read_json, write_json
from ..stats import acf, block_bootstrap_corr, rolling_mean
from .score import surprisal_path
from .volatility import regime_path, rv_path


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    res_dir = Path(cfg.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    n_ctx = cfg.context - 1
    n_train = read_json(regime_path(ctx))["n_train"]

    s = np.load(surprisal_path(ctx, "train"))[n_ctx:]              # drop the unscored first events
    rv = np.load(rv_path(ctx), mmap_mode="r")[:n_train][n_ctx:]
    S = rolling_mean(s, cfg.N)
    start = max(cfg.N, cfg.M) - 1
    x = S[start:]
    y = np.log(np.maximum(np.asarray(rv[start:], dtype=np.float64), 1e-3)).astype(np.float32)
    n = len(x)
    ctx.log(f"dependence: {n:,} training events with both windows full")

    lags = [cfg.N, 10 * cfg.N, 100 * cfg.N, cfg.block_len]
    result = dict(n_events=int(n), block_len_used=cfg.block_len,
                  acf_log_rv={str(k): v for k, v in acf(y, lags).items()},
                  acf_surprisal_mean={str(k): v for k, v in acf(x, lags).items()},
                  blocks=[])
    lens = [b for b in cfg.dependence_block_lens if b <= n // 4]    # need several blocks to resample
    for i, b in enumerate(lens):
        ctx.check_stop()
        ctx.progress.update(i / len(lens), f"bootstrap with block length {b:,}")
        r = block_bootstrap_corr(x, y, b, cfg.n_boot, cfg.boot_seed)
        result["blocks"].append(dict(block_len=b, correlation=r["rho"], se=float(np.std(r["reps"], ddof=1)),
                                     ci_low=r["lo"], ci_high=r["hi"]))
        ctx.log(f"dependence: block {b:>9,}: correlation {r['rho']:.3f}, bootstrap SE {np.std(r['reps'], ddof=1):.4f}")
    write_json(res_dir / "dependence_check.json", result)

    # ---- figure: standard error against block length --------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([b["block_len"] for b in result["blocks"]], [b["se"] for b in result["blocks"]], "o-", color="#1f4e9c")
    ax.axvline(cfg.block_len, color="grey", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("block length (events)")
    ax.set_ylabel("bootstrap standard error of the correlation")
    ax.set_title("Training period: does the standard error level off?\n(dashed line: block length used)")
    fig.tight_layout()
    fig.savefig(res_dir / "dependence_check.png", dpi=130)
    plt.close(fig)
