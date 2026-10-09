"""Stage 4: realised volatility and the regime threshold.

    RV_t = sqrt( mean of the squared price moves (ticks) over the last M events )

computed for every event of the continuous stream (training then test). The first
M-1 events have no full window and get NaN. Windows are in event time, so a window near
the start of the test month reaches back into the end of the training period; that is
legitimate, because it only uses events that precede the one being measured.

The high-volatility regime flag is RV above the 90th percentile of RV over all training
events. The threshold is fitted on the training period only and applied unchanged to the
test month. The flag is used for shading in plots and for choosing excerpts; it does
not enter the headline correlation.

Outputs: ``data/rv.npy`` (float32, whole stream) and ``regime.json``.
"""

from __future__ import annotations

import numpy as np

from ..runtime import RunContext, save_npy, write_json
from ..stats import rolling_rms
from .tokenise import data_path


def rv_path(ctx: RunContext):
    return ctx.cfg.path("data", "rv.npy")


def regime_path(ctx: RunContext):
    return ctx.cfg.path("regime.json")


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    d_train = np.load(data_path(ctx, "train", "delta"), mmap_mode="r")
    d_test = np.load(data_path(ctx, "test", "delta"), mmap_mode="r")
    n_train, n_test = len(d_train), len(d_test)
    ctx.progress.update(0.1, "rolling RMS")
    rv = rolling_rms(np.concatenate((d_train, d_test)), cfg.M)
    save_npy(rv_path(ctx), rv)

    train_rv = rv[:n_train][cfg.M - 1:]
    threshold = float(np.percentile(train_rv, cfg.regime_percentile))
    flagged_train = float(np.mean(train_rv > threshold))
    test_rv = rv[n_train:]
    flagged_test = float(np.mean(test_rv > threshold))
    write_json(regime_path(ctx), dict(threshold_ticks=threshold, percentile=cfg.regime_percentile,
                                      M=cfg.M, n_train=int(n_train), n_test=int(n_test),
                                      flagged_share_train=flagged_train, flagged_share_test=flagged_test))
    ctx.log(f"volatility: regime threshold {threshold:.1f} ticks "
            f"(flags {100 * flagged_train:.1f}% of training and {100 * flagged_test:.1f}% of test events)")
