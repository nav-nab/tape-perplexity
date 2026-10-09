"""Stage 2: clean the raw records and merge them into events.

An *event* is one taker order. Binance's aggTrade record sums the fills of one order at
one price and time, so an order that sweeps several price levels appears as several
consecutive records. We merge those records back together (rule in
docs/tokeniser-design.md, section 3):

    a record joins the current event when
      1. it has the same side as the previous record,
      2. it has the same timestamp (at native resolution), and
      3. its price continues strictly in the aggressor's direction
         (higher for a buy, lower for a sell).

Each day is processed on its own and saved as ``events/<day>.npz`` with arrays

    side   int8   1 = buy aggressor, 0 = sell aggressor
    size   int64  total quantity in lots (1 lot = 0.00001 BTC), an exact integer
    last   int64  price of the event's last fill, in ticks (1 tick = 0.01 USDT)
    t0     int64  timestamp of the event's first record, in microseconds

Prices and quantities are held as integers (ticks and lots) from here on, so there is
no floating-point rounding in later comparisons.

Approximation at day boundaries: events are merged within a day. An order that straddles
midnight within the same microsecond would be split in two. This is negligible.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

from ..runtime import RunContext, save_npz
from .download import raw_path

# Column order of the Binance CSV files (no header row).
COLUMNS = ["a", "p", "q", "f", "l", "T", "m", "M"]


def read_day(zip_path) -> pd.DataFrame:
    """Read one daily zip into a DataFrame with the columns we use."""
    with zipfile.ZipFile(zip_path) as z:
        (name,) = z.namelist()   # each zip holds exactly one CSV
        with z.open(name) as fh:
            return pd.read_csv(fh, header=None, names=COLUMNS,
                               usecols=["a", "p", "q", "T", "m"],
                               dtype={"a": np.int64, "p": np.float64, "q": np.float64,
                                      "T": np.int64, "m": bool})


def records_to_events(a: np.ndarray, p: np.ndarray, q: np.ndarray, T: np.ndarray,
                      m: np.ndarray, tick: float, lot: float) -> dict[str, np.ndarray]:
    """Clean raw records and merge them into events. Pure function (easy to test)."""
    # --- Cleaning -------------------------------------------------------------
    # Binance flags audit duplicates with price 0 or quantity 0: drop them.
    keep = (p > 0) & (q > 0)
    a, p, q, T, m = a[keep], p[keep], q[keep], T[keep], m[keep]
    # Order by aggregate trade ID (files are normally already sorted).
    if len(a) > 1 and np.any(np.diff(a) < 0):
        order = np.argsort(a, kind="stable")
        a, p, q, T, m = a[order], p[order], q[order], T[order], m[order]
    # Files from 2025 on are in microseconds; older ones are in milliseconds.
    if len(T) and T.max() < 10**14:
        T = T * 1000

    # --- Integers instead of floats -------------------------------------------
    ticks = np.rint(p / tick).astype(np.int64)
    lots = np.rint(q / lot).astype(np.int64)
    # m is True when the buyer was the maker, i.e. the SELLER took liquidity.
    side = np.where(m, 0, 1).astype(np.int8)

    n = len(ticks)
    if n == 0:
        z = np.zeros(0, dtype=np.int64)
        return dict(side=np.zeros(0, dtype=np.int8), size=z, last=z.copy(), t0=z.copy())

    # --- Merge rule: a record joins the previous record's event if all three hold ---
    same_side = side[1:] == side[:-1]
    same_time = T[1:] == T[:-1]
    # Price continues in the aggressor's direction: higher for buys, lower for sells.
    continues = np.where(side[1:] == 1, ticks[1:] > ticks[:-1], ticks[1:] < ticks[:-1])
    joins = same_side & same_time & continues
    starts = np.concatenate(([0], np.nonzero(~joins)[0] + 1))        # first record of each event
    ends = np.concatenate((starts[1:] - 1, [n - 1]))                   # last record of each event

    return dict(side=side[starts],
                size=np.add.reduceat(lots, starts),                    # total quantity of the event
                last=ticks[ends],                                      # price of the last fill
                t0=T[starts])


def events_path(ctx: RunContext, day: str):
    return ctx.cfg.path("events", f"{day}.npz")


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    days = cfg.all_days
    n_records = n_events = 0
    for i, day in enumerate(days):
        ctx.check_stop()
        out = events_path(ctx, day)
        if out.exists():                      # finished on an earlier run
            ctx.progress.update((i + 1) / len(days), f"day {day} present ({i + 1}/{len(days)})")
            continue
        ctx.progress.update(i / len(days), f"merging {day} ({i + 1}/{len(days)})")
        df = read_day(raw_path(ctx, day))
        ev = records_to_events(df["a"].to_numpy(), df["p"].to_numpy(), df["q"].to_numpy(),
                               df["T"].to_numpy(), df["m"].to_numpy(), cfg.tick, cfg.lot)
        save_npz(out, **ev)
        n_records += len(df)
        n_events += len(ev["side"])
        ctx.progress.update((i + 1) / len(days),
                            f"{day}: {len(df):,} records -> {len(ev['side']):,} events ({i + 1}/{len(days)})")
    if n_records:
        ctx.log(f"events: {n_records:,} records merged into {n_events:,} events "
                f"({n_records / max(n_events, 1):.2f} records per event) in this session")
