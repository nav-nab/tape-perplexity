"""The ordered list of stages that make up the whole study."""

from __future__ import annotations

from .config import Config
from .stages import download, evaluate, events, export, score, tokenise, train, volatility


def build_stages(cfg: Config):
    """[(name, function)] in the order they run. Each function takes a RunContext."""
    stages = [
        ("download", download.run),         # daily Binance files, checksum-verified
        ("events", events.run),             # clean records and merge them into events
        ("tokenise", tokenise.run),         # fit the tokeniser (training period) and encode
        ("volatility", volatility.run),     # rolling realised volatility and the regime threshold
    ]
    stages += [(f"train_seed{s}", train.make_train_stage(s)) for s in cfg.seeds]
    stages += [
        ("select", train.select_seed),      # lowest final training loss wins
        ("score", score.run),               # surprisal of every test event
        ("evaluate", evaluate.run),         # correlation, block-bootstrap interval, figures
        ("export", export.run),             # weights, tokeniser and excerpts for the repo
    ]
    assert [n for n, _ in stages] == cfg.stage_names
    return stages
