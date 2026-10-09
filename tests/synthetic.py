"""Synthetic Binance-format daily files, so the whole pipeline can run offline.

``write_synthetic_days`` writes ``<symbol>-aggTrades-<day>.zip`` files that look like the
real ones (headerless CSV, columns a,p,q,f,l,T,m,M, microsecond timestamps) together
with the ``.verified`` marker the download stage looks for, so no network is needed.

The data has the features the pipeline cares about: orders that sweep several price
levels (multi-record events), common round sizes, and a volatile stretch in the middle of
the last day.
"""

from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import numpy as np


def make_day(day: str, n_events: int, seed: int, a0: int, volatile: tuple[float, float] | None = None):
    """Return (csv_text, next_a). Prices random-walk; a volatile window scales the moves up."""
    rng = np.random.default_rng(seed)
    t = dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp() * 1e6
    price = 82000.0 + rng.normal(0, 50)
    rows, a = [], a0
    gap = 86400e6 / n_events
    for i in range(n_events):
        frac = i / n_events
        vol = 1.0 + (6.0 if volatile and volatile[0] <= frac < volatile[1] else 0.0)
        t += rng.exponential(gap)
        ts = int(t)
        buyer_aggr = rng.random() < 0.5
        # Number of price levels the order sweeps: mostly 1, sometimes several.
        levels = 1 + int(rng.random() < 0.35) * rng.integers(1, 5)
        size = float(rng.choice([0.00012, 0.001, 0.002, 0.005, 0.02, 0.1, 0.5, 2.0],
                                p=[.1, .15, .3, .2, .12, .08, .04, .01]))
        price += rng.normal(0, 0.4 * vol) * (1 if rng.random() < 0.5 else -1)
        for lev in range(levels):
            step = 0.01 * (1 + int(rng.random() * 2 * vol))
            price += step if buyer_aggr else -step
            p = round(max(price, 1000.0), 2)
            q = round(size / levels + 0.00001 * rng.integers(0, 3), 5)
            m = not buyer_aggr          # m True = buyer was the maker = seller aggressor
            rows.append(f"{a},{p:.2f},{q:.5f},{a},{a},{ts},{'True' if m else 'False'},True")
            a += 1
    return "\n".join(rows) + "\n", a


def write_synthetic_days(raw_dir: Path, days: list[str], n_events: int = 6000, symbol: str = "BTCUSDT") -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    a = 1
    for i, day in enumerate(days):
        vol = (0.4, 0.6) if i == len(days) - 1 else None
        text, a = make_day(day, n_events, seed=i + 1, a0=a, volatile=vol)
        zp = raw_dir / f"{symbol}-aggTrades-{day}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{symbol}-aggTrades-{day}.csv", text)
        (raw_dir / (zp.name + ".verified")).write_text("synthetic")
