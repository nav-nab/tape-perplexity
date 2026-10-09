"""The tokeniser: turns (side, size, price move) into one integer token and back.

Full specification: docs/tokeniser-design.md. In short, one token describes one event:

    side   2 values   0 = sell aggressor, 1 = buy aggressor
    size   S values   log-spaced bins on the total quantity (S <= 16 after the tail rule)
    move   7 values   0 down large, 1 down mid, 2 down 1 tick, 3 flat,
                      4 up 1 tick, 5 up mid, 6 up large

    token id = (side * S + size) * 7 + move          (so ids run 0 .. 2*S*7 - 1)

Only two things are fitted to data, and only on the training period:
  * the tail rule (which outer size bins are merged into their neighbours), and
  * the median that separates "mid" from "large" price moves of 2 or more ticks.
The log-spaced size edges themselves are constants.

This module has no dependence on the rest of the pipeline, so the GUI can import it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_MOVE = 7
MOVE_NAMES = ["down large", "down mid", "down 1 tick", "flat", "up 1 tick", "up mid", "up large"]


def size_edges_lots(start_btc: float, lot: float, per_decade: int, n_edges: int) -> np.ndarray:
    """Bin edges in lots: ``start * 10**(k / per_decade)`` for k = 1 .. n_edges.

    Edges are built as an exact power of ten times a fractional power, so that edges
    that are whole numbers of lots (5, 50, 500, ...) are exact in floating point. A size
    equal to an edge then always falls in the same (upper) bin.
    """
    start_lots = round(start_btc / lot)           # 5e-5 BTC / 1e-5 BTC = 5 lots
    assert abs(start_lots * lot - start_btc) < 1e-12, "start is not a whole number of lots"
    k = np.arange(1, n_edges + 1)
    return start_lots * 10.0 ** (k // per_decade) * 10.0 ** ((k % per_decade) / per_decade)


@dataclass(frozen=True)
class Tokeniser:
    edges: np.ndarray        # raw size-bin edges in lots (15 values -> 16 raw bins)
    lo: int                  # first raw size bin kept (bins below are merged into it)
    hi: int                  # last raw size bin kept (bins above are merged into it)
    median_big: float        # training median of |move| among moves of 2 or more ticks

    # ---- derived sizes ----
    @property
    def n_size(self) -> int:
        return self.hi - self.lo + 1

    @property
    def vocab(self) -> int:
        return 2 * self.n_size * N_MOVE

    # ---- encoding ----
    def size_bin(self, size_lots: np.ndarray) -> np.ndarray:
        """Final size bin (0 .. n_size-1) of each event."""
        raw = np.searchsorted(self.edges, size_lots, side="right")   # raw bin 0 .. len(edges)
        return np.clip(raw, self.lo, self.hi) - self.lo

    def move_class(self, delta: np.ndarray) -> np.ndarray:
        """Move class (0 .. 6) of each price move, given in ticks."""
        a = np.abs(delta)
        cls = np.full(delta.shape, 3, dtype=np.int64)                # flat
        big = a >= 2
        mid, large = big & (a <= self.median_big), big & (a > self.median_big)
        up, dn = delta > 0, delta < 0
        cls[up & (a == 1)], cls[up & mid], cls[up & large] = 4, 5, 6
        cls[dn & (a == 1)], cls[dn & mid], cls[dn & large] = 2, 1, 0
        return cls

    def encode(self, side: np.ndarray, size_lots: np.ndarray, delta: np.ndarray) -> np.ndarray:
        ids = (side.astype(np.int64) * self.n_size + self.size_bin(size_lots)) * N_MOVE + self.move_class(delta)
        assert ids.max(initial=0) < 256, "token ids must fit in uint8"
        return ids.astype(np.uint8)

    def decode(self, token_id: int) -> tuple[int, int, int]:
        """(side, size bin, move class) of a token id."""
        rest, move = divmod(int(token_id), N_MOVE)
        side, size = divmod(rest, self.n_size)
        return side, size, move

    # ---- saving ----
    def to_dict(self) -> dict:
        return dict(edges=[float(e) for e in self.edges], lo=int(self.lo), hi=int(self.hi),
                    median_big=float(self.median_big), n_size=int(self.n_size), vocab=int(self.vocab),
                    move_names=MOVE_NAMES,
                    numbering="id=(side*n_size+size)*7+move; side 0=sell 1=buy; "
                              "move 0..6=down large..up large")

    @staticmethod
    def from_dict(d: dict) -> "Tokeniser":
        return Tokeniser(np.array(d["edges"]), int(d["lo"]), int(d["hi"]), float(d["median_big"]))

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: Path | str) -> "Tokeniser":
        return Tokeniser.from_dict(json.loads(Path(path).read_text()))


def fit(edges: np.ndarray, raw_bin_counts: np.ndarray, big_abs_moves: np.ndarray,
        tail_min_share: float) -> Tokeniser:
    """Fit the tokeniser on the training period.

    ``raw_bin_counts[i]`` is the number of training events in raw size bin i, and
    ``big_abs_moves`` holds the absolute price moves (ticks) that are 2 or more.

    Tail rule: from each end, a bin holding less than ``tail_min_share`` of all events is
    merged into its inner neighbour, and the test repeats on the merged bin until the
    outermost bin holds at least that share. Interior bins are never merged.
    """
    total = float(raw_bin_counts.sum())
    # Low end: accumulate bins upward until the merged outer bin is big enough.
    lo, acc = 0, float(raw_bin_counts[0])
    while acc / total < tail_min_share and lo < len(raw_bin_counts) - 1:
        lo += 1
        acc += raw_bin_counts[lo]
    # High end: the same from the top (never crossing the low end).
    hi, acc = len(raw_bin_counts) - 1, float(raw_bin_counts[-1])
    while acc / total < tail_min_share and hi > lo:
        hi -= 1
        acc += raw_bin_counts[hi]
    median_big = float(np.median(big_abs_moves)) if len(big_abs_moves) else 2.0
    return Tokeniser(np.asarray(edges, dtype=np.float64), lo, hi, median_big)
