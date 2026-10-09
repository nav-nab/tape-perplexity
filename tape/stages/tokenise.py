"""Stage 3: fit the tokeniser on the training period, then encode every event.

Three phases:

  A. Fit.  Walk the training days once to count events per raw size bin and collect the
           absolute price moves of 2 or more ticks. Fit the tail rule and the median
           (``tokeniser.fit``). Only the training period is looked at.
  B. Encode. For every day (training and test), compute each event's price move and
           encode it as one token. Output ``tokens/<day>.npz``.
  C. Assemble. Join the days into one continuous stream per split, written as plain
           ``.npy`` arrays that later stages memory-map:

           data/{train,test}_tokens.npy   uint8   one token per event
           data/{train,test}_delta.npy    int32   price move in ticks
           data/{train,test}_t0.npy       int64   event start time (microseconds)

The price move of an event is its last fill minus the previous event's last fill. The
previous event can lie in the previous day, so days are walked in order and the last
price is carried across the boundary. The very first event of the whole stream has no
previous event and is dropped.
"""

from __future__ import annotations

import numpy as np

from .. import tokeniser as tk
from ..runtime import RunContext, save_npy, save_npz
from .events import events_path


def tokeniser_path(ctx: RunContext):
    return ctx.cfg.path("tokeniser.json")


def data_path(ctx: RunContext, split: str, what: str):
    return ctx.cfg.path("data", f"{split}_{what}.npy")


def _walk_days(ctx: RunContext):
    """Yield (day, events, delta) for every day in order, with moves carried across days.

    Events without a previous event (only the first of the whole stream) are removed
    from both ``events`` and ``delta``.
    """
    prev_last = None
    for day in ctx.cfg.all_days:
        ev = dict(np.load(events_path(ctx, day)))
        if len(ev["last"]) == 0:
            yield day, ev, np.zeros(0, dtype=np.int64)
            continue
        last = ev["last"]
        if prev_last is None:
            # First event of the whole stream: it has no previous event, so no price
            # move is defined. Drop it; the others compare with their predecessor.
            delta = np.diff(last)
            ev = {k: v[1:] for k, v in ev.items()}
        else:
            delta = np.diff(np.concatenate(([prev_last], last)))
        prev_last = int(last[-1])
        yield day, ev, delta


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    all_days = cfg.all_days
    edges = tk.size_edges_lots(cfg.size_edge_start_btc, cfg.lot, cfg.size_bins_per_decade, cfg.size_n_edges)

    # ---- A. fit on the training period -------------------------------------------
    if tokeniser_path(ctx).exists():
        tok = tk.Tokeniser.load(tokeniser_path(ctx))
    else:
        counts = np.zeros(len(edges) + 1)
        big_moves = []
        train = set(cfg.train_days)
        for i, (day, ev, delta) in enumerate(_walk_days(ctx)):
            ctx.check_stop()
            if day not in train:
                break
            raw = np.searchsorted(edges, ev["size"], side="right")
            counts += np.bincount(raw, minlength=len(edges) + 1)
            a = np.abs(delta)
            big_moves.append(a[a >= 2].astype(np.int32))
            ctx.progress.update(0.3 * (i + 1) / len(cfg.train_days), f"fitting on {day}")
        tok = tk.fit(edges, counts, np.concatenate(big_moves), cfg.tail_min_share)
        tok.save(tokeniser_path(ctx))
        ctx.log(f"tokeniser: {tok.n_size} size bins (raw bins {tok.lo}..{tok.hi} kept), "
                f"vocabulary {tok.vocab}, median large-move boundary {tok.median_big:.1f} ticks")
        shares = counts / counts.sum()
        ctx.log("tokeniser: raw size-bin shares (%): " + " ".join(f"{100 * s:.2f}" for s in shares))

    # ---- B. encode each day -------------------------------------------------------
    for i, (day, ev, delta) in enumerate(_walk_days(ctx)):
        ctx.check_stop()
        out = cfg.path("tokens", f"{day}.npz")
        if out.exists():
            continue
        assert delta.min(initial=0) > -2**31 and delta.max(initial=0) < 2**31
        save_npz(out, tok=tok.encode(ev["side"], ev["size"], delta), delta=delta.astype(np.int32), t0=ev["t0"])
        ctx.progress.update(0.3 + 0.6 * (i + 1) / len(all_days), f"encoding {day}")

    # ---- C. assemble the two continuous streams -----------------------------------
    for split, days in (("train", cfg.train_days), ("test", cfg.test_days)):
        parts = [np.load(cfg.path("tokens", f"{d}.npz")) for d in days]
        for key, name in (("tok", "tokens"), ("delta", "delta"), ("t0", "t0")):
            save_npy(data_path(ctx, split, name), np.concatenate([p[key] for p in parts]))
        n = sum(len(p["tok"]) for p in parts)
        ctx.log(f"tokenise: {split} stream has {n:,} events")
    ctx.progress.update(1.0, "assembled")
