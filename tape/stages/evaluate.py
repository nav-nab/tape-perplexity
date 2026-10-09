"""Stage 8: the headline test.

For each test event t (once a full window of N events exists) we have

    x_t = S_t        rolling mean surprisal over the last N events (nats)
    y_t = log RV_t   log of the realised volatility over the last M events

The statistic is the Pearson correlation of x and y on the test month, with a 95%
moving-block bootstrap interval (block length 1,000,000 events, 2,000 replicates;
docs/block-bootstrap.md). The figures show the two series over time with the
high-volatility regime shaded, and a scatter of one against the other.

Outputs (in the results directory): ``results.json``, ``timeseries.png``, ``scatter.png``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # no display needed
import matplotlib.pyplot as plt
import numpy as np

from ..runtime import RunContext, read_json, write_json
from ..stats import acf, block_bootstrap_corr, rolling_mean
from .score import surprisal_path
from .tokenise import data_path
from .volatility import regime_path, rv_path


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    res_dir = Path(cfg.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    surprisal = np.load(surprisal_path(ctx))
    regime = read_json(regime_path(ctx))
    n_train, n_test = regime["n_train"], regime["n_test"]
    rv = np.load(rv_path(ctx), mmap_mode="r")[n_train:]            # test-month RV
    t0 = np.load(data_path(ctx, "test", "t0"), mmap_mode="r")
    assert len(surprisal) == n_test == len(rv)

    ctx.progress.update(0.05, "rolling mean surprisal")
    S = rolling_mean(surprisal, cfg.N)
    start = max(cfg.N, cfg.M) - 1                                   # first event with both windows full
    x = S[start:]
    rv_valid = np.asarray(rv[start:], dtype=np.float64)
    if np.any(rv_valid <= 0):
        ctx.log("WARNING: some windows have zero volatility; clipping before the log")
    y = np.log(np.maximum(rv_valid, 1e-3)).astype(np.float32)

    ctx.progress.update(0.2, "block bootstrap")
    boot = block_bootstrap_corr(x, y, cfg.block_len, cfg.n_boot, cfg.boot_seed)
    reps = boot.pop("reps")
    ctx.log(f"evaluate: correlation {boot['rho']:.3f}, 95% block-bootstrap interval "
            f"[{boot['lo']:.3f}, {boot['hi']:.3f}] ({boot['n']:,} events, block {boot['block_len']:,})")

    ctx.progress.update(0.5, "diagnostics")
    lags = [cfg.N, 10 * cfg.N, 100 * cfg.N, cfg.block_len]
    flagged = rv_valid > regime["threshold_ticks"]
    results = dict(
        correlation=boot["rho"], ci_low=boot["lo"], ci_high=boot["hi"], ci_level=0.95,
        n_events=boot["n"], block_len=boot["block_len"], n_boot=boot["n_boot"],
        N=cfg.N, M=cfg.M, regime_threshold_ticks=regime["threshold_ticks"],
        flagged_share_test=float(flagged.mean()),
        mean_surprisal_nats=float(np.mean(surprisal)),
        # Autocorrelation of both series at several lags (how fast dependence fades).
        acf_log_rv={str(k): v for k, v in acf(y, lags).items()},
        acf_surprisal_mean={str(k): v for k, v in acf(x, lags).items()},
    )
    write_json(res_dir / "results.json", results)

    # ---- figure 1: the two series over time, regime shaded -------------------------
    ctx.progress.update(0.7, "figures")
    step = max(1, len(x) // cfg.plot_points)
    sel = np.arange(0, len(x), step)
    when = [dt.datetime.fromtimestamp(float(t0[start + i]) / 1e6, dt.timezone.utc) for i in sel]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    a1.plot(when, x[sel], lw=0.8, color="#1f4e9c")
    a1.set_ylabel(f"mean surprisal over {cfg.N} events (nats)")
    a2.plot(when, rv_valid[sel], lw=0.8, color="#c2410c")
    a2.set_yscale("log")
    a2.set_ylabel(f"realised volatility over {cfg.M} events (ticks)")
    a2.axhline(regime["threshold_ticks"], color="grey", ls="--", lw=0.8)
    # Shade runs of flagged points on both panels.
    fl = flagged[sel]
    edges = np.flatnonzero(np.diff(np.concatenate(([0], fl.astype(int), [0]))))
    for s_, e_ in zip(edges[::2], edges[1::2]):
        for ax in (a1, a2):
            ax.axvspan(when[s_], when[min(e_, len(when) - 1)], color="#f59e0b", alpha=0.18, lw=0)
    a1.set_title(f"Test month: correlation {boot['rho']:.2f} (95% interval {boot['lo']:.2f} to {boot['hi']:.2f}); "
                 f"shading = volatility above the training 90th percentile")
    fig.tight_layout()
    fig.savefig(res_dir / "timeseries.png", dpi=130)
    plt.close(fig)

    # ---- figure 2: scatter -------------------------------------------------------------
    rng = np.random.default_rng(0)
    pick = rng.choice(len(x), size=min(200_000, len(x)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hexbin(y[pick], x[pick], gridsize=60, bins="log", cmap="viridis", mincnt=1)
    ax.set_xlabel("log realised volatility")
    ax.set_ylabel("mean surprisal (nats)")
    fig.tight_layout()
    fig.savefig(res_dir / "scatter.png", dpi=130)
    plt.close(fig)
    ctx.progress.update(1.0, "done")
