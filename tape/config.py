"""All settings of the study in one place.

Every number that the design documents fix (study period, bin edges, model size,
training recipe, window lengths, bootstrap settings) lives here, so the rest of the
code contains no hidden constants.

Two presets exist:

* ``Config.full()``  - the real study (91 days of BTCUSDT, medium model, 3 seeds).
* ``Config.smoke()`` - a tiny version (a few days, a toy model) that exercises every
  stage in a minute or two. It is used by the tests and to check a new machine.

The *fingerprint* is a hash of the settings that change results. It is stored with every
checkpoint and completion marker. If the settings change, the script refuses to resume
from old state (see ``runtime.py``).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ paths
    # Everything large or regenerable goes under ``workdir`` (ignored by git).
    workdir: str = "work"
    # Small final artefacts that ARE committed: weights, tokeniser, prompt excerpts.
    models_dir: str = "models"
    # Small result files and figures from the evaluation.
    results_dir: str = "results"

    # ------------------------------------------------------------------- data
    symbol: str = "BTCUSDT"
    base_url: str = "https://data.binance.vision/data/spot/daily/aggTrades"
    # Inclusive date ranges (UTC). Training strictly precedes testing.
    train_start: str = "2025-09-01"
    train_end: str = "2025-10-31"
    test_start: str = "2025-11-01"
    test_end: str = "2025-11-30"
    tick: float = 0.01          # price step in USDT (Binance exchangeInfo)
    lot: float = 0.00001        # quantity step in BTC (Binance exchangeInfo)

    # -------------------------------------------------------------- tokeniser
    # Size bin edges (in BTC): e_k = size_edge_start * 10**(k / bins_per_decade).
    size_edge_start_btc: float = 5e-5
    size_bins_per_decade: int = 3
    size_n_edges: int = 15      # 15 edges -> 16 bins before the tail rule
    tail_min_share: float = 0.01  # outer bins below this share of training events merge inward

    # ------------------------------------------------------------------ model
    n_layer: int = 6
    d_model: int = 192
    n_head: int = 6
    context: int = 512          # events the model can read at once
    mlp_ratio: int = 4

    # --------------------------------------------------------------- training
    seeds: tuple = (0, 1, 2)
    tokens_per_step: int = 16384   # about 16k tokens per optimiser step
    peak_lr: float = 1e-3
    warmup_steps: int = 100
    min_lr_frac: float = 0.1       # cosine decay down to 10% of the peak
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    final_loss_tail: float = 0.1   # "final training loss" = mean over the last 10% of steps
    seed_loss_warn: float = 0.01   # warn if best and worst final losses differ by more than this (nats)

    # ---------------------------------------------------------------- scoring
    stride: int = 64               # sliding-window stride at test time
    score_batch_windows: int = 32  # windows per forward pass

    # ---------------------------------------------------- signal and evaluation
    M: int = 3000                  # realised-volatility window (events)
    N: int = 3000                  # surprisal window (events)
    regime_percentile: float = 90.0
    block_len: int = 1_000_000     # moving-block bootstrap block length (events)
    n_boot: int = 2000
    boot_seed: int = 12345
    plot_points: int = 6000        # points drawn in the time-series figure
    # Block lengths tried in the dependence check on the training period (events).
    dependence_block_lens: tuple = (1_000, 10_000, 100_000, 300_000, 1_000_000, 2_000_000)

    # --------------------------------------------------------------- excerpts
    excerpts_per_kind: int = 5     # 5 volatile + 5 calm
    excerpt_continuation: int = 256
    excerpt_seed: int = 2025

    # ------------------------------------------------------------ run behaviour
    checkpoint_interval_s: float = 600.0   # checkpoint every 10 minutes of wall time
    heartbeat_interval_s: float = 60.0     # status line every minute
    threads: int = 0                       # 0 = PyTorch's own default (physical cores)

    # Planned duration (seconds) of each stage, used for the run ETA until a stage has
    # started and its measured rate takes over. Training and scoring come from the CPU
    # benchmark in the design notes; the others are rough guesses.
    planned_s: dict = field(default_factory=lambda: {
        "download": 600.0, "events": 150.0, "tokenise": 120.0, "volatility": 30.0,
        "train": 3720.0,   # per seed: about 62 minutes
        "select": 5.0, "score": 7200.0, "evaluate": 300.0, "export": 60.0,
        "score_train": 9000.0,   # the training period is about 1.25 times the test month
        "dependence": 120.0,
    })

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def full() -> "Config":
        return Config()

    @staticmethod
    def smoke() -> "Config":
        """A tiny configuration that runs every stage in a minute or two."""
        return Config(
            workdir="work_smoke", models_dir="models_smoke", results_dir="results_smoke",
            train_start="2025-09-01", train_end="2025-09-03",
            test_start="2025-09-04", test_end="2025-09-05",
            n_layer=2, d_model=64, n_head=4, context=128,
            tokens_per_step=4096, warmup_steps=10,
            stride=16, score_batch_windows=64,
            M=300, N=300, block_len=20_000, n_boot=200,
            excerpts_per_kind=2, excerpt_continuation=32,
            dependence_block_lens=(500, 2_000, 5_000, 20_000, 50_000),
            checkpoint_interval_s=20.0, heartbeat_interval_s=5.0,
            plot_points=2000,
            planned_s={"download": 60.0, "events": 60.0, "tokenise": 20.0, "volatility": 5.0,
                       "train": 120.0, "select": 1.0, "score": 120.0, "evaluate": 20.0,
                       "export": 5.0, "score_train": 120.0, "dependence": 10.0},
        )

    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)

    # Days ---------------------------------------------------------------------
    @staticmethod
    def _days(start: str, end: str) -> list[str]:
        d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
        return [(d0 + dt.timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]

    @property
    def train_days(self) -> list[str]:
        return self._days(self.train_start, self.train_end)

    @property
    def test_days(self) -> list[str]:
        return self._days(self.test_start, self.test_end)

    @property
    def all_days(self) -> list[str]:
        return self.train_days + self.test_days

    # Paths --------------------------------------------------------------------
    def path(self, *parts: str) -> Path:
        """A path under the work directory."""
        return Path(self.workdir).joinpath(*parts)

    # Stage list ---------------------------------------------------------------
    @property
    def stage_names(self) -> list[str]:
        return (["download", "events", "tokenise", "volatility"]
                + [f"train_seed{s}" for s in self.seeds]
                + ["select", "score", "evaluate", "export", "score_train", "dependence"])

    def planned_for(self, stage: str) -> float:
        key = "train" if stage.startswith("train_seed") else stage
        return float(self.planned_s.get(key, 60.0))

    # Fingerprint --------------------------------------------------------------
    def fingerprint(self) -> str:
        """Hash of every setting that can change a result.

        Paths, checkpoint and heartbeat intervals, thread count and planned durations
        are excluded: changing them does not invalidate saved state.
        """
        d = dataclasses.asdict(self)
        for k in ("workdir", "models_dir", "results_dir", "checkpoint_interval_s",
                  "heartbeat_interval_s", "threads", "planned_s", "plot_points"):
            d.pop(k)
        blob = json.dumps(d, sort_keys=True, default=list).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
