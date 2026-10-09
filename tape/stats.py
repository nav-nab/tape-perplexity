"""Rolling statistics and the block bootstrap.

Everything here is a plain function of arrays, so it is easy to test.

* ``rolling_mean`` / ``rolling_rms``: trailing-window means over events.
* ``block_bootstrap_corr``: the moving-block bootstrap for a correlation, computed from
  prefix sums of five sufficient statistics (method: docs/block-bootstrap.md).
* ``acf``: sample autocorrelation at chosen lags (a diagnostic).
"""

from __future__ import annotations

import numpy as np

CHUNK = 4_000_000   # rows processed at a time, to keep memory modest


# ------------------------------------------------------------------ rolling windows
def rolling_mean(x: np.ndarray, window: int, square: bool = False) -> np.ndarray:
    """Trailing mean of ``x`` (or of ``x**2``) over the last ``window`` entries.

    Entry ``t`` of the result averages ``x[t-window+1 .. t]``. The first ``window-1``
    entries have no full window and are NaN. Sums are taken in float64 over chunks, so
    rounding error stays negligible even for tens of millions of events.
    """
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)
    for b0 in range(0, n, CHUNK):
        b1 = min(n, b0 + CHUNK)
        lo = max(0, b0 - window + 1)             # include the window-1 events before the chunk
        seg = np.asarray(x[lo:b1], dtype=np.float64)
        if square:
            seg = seg * seg
        cs = np.concatenate(([0.0], np.cumsum(seg)))   # cs[j] = sum of the first j entries
        t = np.arange(max(b0, window - 1), b1)          # events with a full window
        if len(t) == 0:
            continue
        j = t - lo
        out[t] = (cs[j + 1] - cs[j + 1 - window]) / window
    return out


def rolling_rms(delta: np.ndarray, window: int) -> np.ndarray:
    """Realised volatility: root mean square of the last ``window`` price moves (ticks)."""
    return np.sqrt(rolling_mean(delta, window, square=True))


# --------------------------------------------------------------------- correlation
def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Sample correlation, computed in chunks in float64."""
    n = len(x)
    mx, my = _mean(x), _mean(y)
    sxx = syy = sxy = 0.0
    for b0 in range(0, n, CHUNK):
        a = np.asarray(x[b0:b0 + CHUNK], dtype=np.float64) - mx
        b = np.asarray(y[b0:b0 + CHUNK], dtype=np.float64) - my
        sxx += float(a @ a)
        syy += float(b @ b)
        sxy += float(a @ b)
    return sxy / np.sqrt(sxx * syy)


def _mean(x: np.ndarray) -> float:
    n, s = len(x), 0.0
    for b0 in range(0, n, CHUNK):
        s += float(np.sum(x[b0:b0 + CHUNK], dtype=np.float64))
    return s / n


# -------------------------------------------------------------------- block bootstrap
def _prefix_at(cols: list[np.ndarray], positions: np.ndarray) -> np.ndarray:
    """Prefix sums of several columns, evaluated only at the requested positions.

    ``result[i, c]`` is the sum of the first ``positions[i]`` entries of column ``c``
    (so position 0 gives 0 and position n gives the total). ``positions`` must be
    sorted. Only a few thousand positions are needed, so we never store a full
    length-n prefix array; we stream through the data once instead.
    """
    n = len(cols[0])
    out = np.zeros((len(positions), len(cols)))
    carry = np.zeros(len(cols))
    for b0 in range(0, n, CHUNK):
        b1 = min(n, b0 + CHUNK)
        # Positions p with b0 < p <= b1 need the running sum inside this chunk.
        sel = np.nonzero((positions > b0) & (positions <= b1))[0]
        cums = [np.cumsum(c[b0:b1], dtype=np.float64) for c in cols]
        for c, cum in enumerate(cums):
            if len(sel):
                out[sel, c] = carry[c] + cum[positions[sel] - b0 - 1]
            carry[c] += cum[-1]
    out[positions == 0] = 0.0
    return out


def corr_from_sums(n: np.ndarray | float, sx, sy, sxx, syy, sxy):
    """Correlation from the five sums over n pairs. Works on arrays of replicates."""
    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    return cov / np.sqrt(vx * vy)


def block_bootstrap_corr(x: np.ndarray, y: np.ndarray, block_len: int, n_boot: int,
                         seed: int, level: float = 0.95) -> dict:
    """Moving-block bootstrap percentile interval for ``corr(x, y)``.

    Steps (see docs/block-bootstrap.md):
      1. Blocks are every window of ``b`` consecutive pairs, starting at 0 .. n-b.
      2. Draw ``k = ceil(n / b)`` start positions uniformly with replacement.
      3. Glue the blocks and cut to length n (the last block is used only in part).
      4. Recompute the correlation. Repeat ``n_boot`` times.
      5. The interval is the (1-level)/2 and (1+level)/2 percentiles of the replicates.

    Each replicate needs only the five sums over its blocks. Those are differences of
    prefix sums, which we evaluate at the few thousand positions that are needed.
    """
    n = len(x)
    b = min(block_len, n)
    k = int(np.ceil(n / b))
    rem = n - (k - 1) * b                       # length used from the last block (1..b)
    # Centre the data first: correlation is unchanged, and the sums become better behaved.
    xm, ym = _mean(x), _mean(y)

    class _C:   # lazily centred column, so we never hold a second copy of the data
        def __init__(self, f):
            self.f = f
        def __len__(self):
            return n
        def __getitem__(self, s):
            return self.f(s)
    cx = _C(lambda s: np.asarray(x[s], dtype=np.float64) - xm)
    cy = _C(lambda s: np.asarray(y[s], dtype=np.float64) - ym)
    cols = [cx, cy, _C(lambda s: cx[s] ** 2), _C(lambda s: cy[s] ** 2), _C(lambda s: cx[s] * cy[s])]

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - b + 1, size=(n_boot, k))        # step 2
    # Positions where prefix sums are needed: each start, each start + b, and the end
    # of the partial last block.
    pos_full = np.concatenate([starts.ravel(), (starts + b).ravel(), starts[:, -1] + rem])
    uniq = np.unique(pos_full)
    pref = _prefix_at(cols, uniq)                                # shape (len(uniq), 5)
    look = lambda p: pref[np.searchsorted(uniq, p)]
    # Sums over full blocks 0..k-2, plus the partial last block.
    tot = np.zeros((n_boot, 5))
    for i in range(k - 1):
        tot += look(starts[:, i] + b) - look(starts[:, i])
    tot += look(starts[:, -1] + rem) - look(starts[:, -1])
    reps = corr_from_sums(float(n), tot[:, 0], tot[:, 1], tot[:, 2], tot[:, 3], tot[:, 4])  # step 4
    a = (1 - level) / 2
    lo, hi = np.quantile(reps, [a, 1 - a])
    return dict(rho=pearson(x, y), lo=float(lo), hi=float(hi), n=int(n), block_len=int(b),
                n_blocks_drawn=int(k), n_boot=int(n_boot), reps=reps)


# ---------------------------------------------------------------------------- acf
def acf(x: np.ndarray, lags: list[int]) -> dict[int, float]:
    """Sample autocorrelation of ``x`` at the given lags (chunked, float64)."""
    n = len(x)
    m = _mean(x)
    var = 0.0
    for b0 in range(0, n, CHUNK):
        a = np.asarray(x[b0:b0 + CHUNK], dtype=np.float64) - m
        var += float(a @ a)
    var /= n
    out = {}
    for lag in lags:
        if lag >= n:
            continue
        s = 0.0
        for b0 in range(0, n - lag, CHUNK):
            b1 = min(n - lag, b0 + CHUNK)
            a = np.asarray(x[b0:b1], dtype=np.float64) - m
            c = np.asarray(x[b0 + lag:b1 + lag], dtype=np.float64) - m
            s += float(a @ c)
        out[int(lag)] = s / (n - lag) / var
    return out
