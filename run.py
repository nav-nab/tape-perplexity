#!/usr/bin/env python3
"""Run the whole study with one command.

    python run.py                 # the full study (91 days of BTCUSDT; several hours on a laptop CPU)
    python run.py --smoke         # a tiny version of every stage (a few minutes); good for checking a machine

The run is safe to interrupt (Ctrl-C, or the process being terminated) and safe to
repeat: finished stages are skipped, and long stages (training, scoring) resume from a
checkpoint that is written every 10 minutes of wall time. A status line with the
progress and a whole-run ETA is printed every minute.

Other options:
    --reset            forget saved state (downloads are kept) and start again
    --stages a,b       run only the named stages (they must follow the usual order)
    --workdir DIR      where large intermediate files live (default: work)
    --list-stages      print the stage names and exit
"""

from __future__ import annotations

import argparse
import ast
import shutil
import signal
import sys

from tape.config import Config
from tape.pipeline import build_stages
from tape.runtime import Heartbeat, RunContext, StopRequested, configure_threads, run_stages


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="run the tiny configuration")
    ap.add_argument("--reset", action="store_true", help="delete saved state (keep downloads) and start over")
    ap.add_argument("--stages", help="comma-separated stage names to run")
    ap.add_argument("--list-stages", action="store_true")
    ap.add_argument("--workdir")
    ap.add_argument("--models-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--checkpoint-seconds", type=float)
    ap.add_argument("--heartbeat-seconds", type=float)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override any setting from tape/config.py, e.g. --set M=1000 (repeatable)")
    args = ap.parse_args(argv)

    cfg = Config.smoke() if args.smoke else Config.full()
    over = {k: v for k, v in dict(workdir=args.workdir, models_dir=args.models_dir,
                                  results_dir=args.results_dir,
                                  checkpoint_interval_s=args.checkpoint_seconds,
                                  heartbeat_interval_s=args.heartbeat_seconds,
                                  threads=args.threads).items() if v is not None}
    cfg = cfg.replace(**over)
    for item in args.set:                      # generic overrides, e.g. --set tokens_per_step=256
        name, _, value = item.partition("=")
        if name not in cfg.__dataclass_fields__:
            sys.exit(f"--set: unknown setting {name!r}")
        cfg = cfg.replace(**{name: ast.literal_eval(value)})

    if args.list_stages:
        print("\n".join(cfg.stage_names))
        return 0

    if args.reset:
        # Keep the downloaded files (slow to fetch); remove everything derived from them.
        for p in cfg.path().glob("*"):
            if p.name != "raw":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        print(f"reset: removed saved state in {cfg.path()} (downloads kept)")

    only = set(args.stages.split(",")) if args.stages else None
    if only and not only <= set(cfg.stage_names):
        sys.exit(f"unknown stage(s): {sorted(only - set(cfg.stage_names))}")

    configure_threads(cfg)
    ctx = RunContext(cfg)

    # Interrupts: do not die at once. Ask the running stage to save a checkpoint and
    # stop at its next safe point. A second Ctrl-C forces an immediate exit.
    def handler(signum, frame):
        if ctx.stop_requested:
            ctx.log("second interrupt: exiting immediately")
            raise KeyboardInterrupt
        ctx.log(f"signal {signum} received: will checkpoint and stop at the next safe point")
        ctx.request_stop()
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    beat = Heartbeat(ctx)
    beat.start()
    code = 0
    try:
        run_stages(ctx, build_stages(cfg), only)
        ctx.log("all requested stages finished")
    except StopRequested:
        ctx.log("interrupted: state saved. Run the same command again to resume.")
        code = 130
    finally:
        beat.stop()
    return code


if __name__ == "__main__":
    sys.exit(main())
