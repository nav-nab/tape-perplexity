"""Run-time machinery shared by every stage: safe file writes, progress, checkpoints,
heartbeat, interrupt handling and the stage runner.

How the pieces fit together
---------------------------
* ``run.py`` builds a ``RunContext`` and hands it to ``run_stages``.
* Each stage is a function ``fn(ctx)``. It reports progress with ``ctx.progress.update``,
  and (if it is long) calls ``ctx.maybe_checkpoint(save_fn)`` once per loop iteration.
* ``maybe_checkpoint`` saves state when 10 wall-clock minutes have passed since the last
  save, or at once when an interrupt (Ctrl-C or SIGTERM) has been requested, and in the
  second case it then raises ``StopRequested`` so the script exits cleanly.
* A ``Heartbeat`` thread prints one status line a minute (stage, progress, ETA).
* A finished stage leaves ``state/<stage>.done.json``. A rerun skips finished stages.

Why every write is atomic
-------------------------
A checkpoint is useless if the process is killed half way through writing it. So files
are written to a temporary name in the same directory and then renamed over the target.
A rename within one directory is atomic on every common file system, so a reader (or a
resumed run) sees either the old complete file or the new complete file, never a mixture.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .config import Config


class StopRequested(Exception):
    """Raised after an interrupt has been handled and state has been saved."""


# --------------------------------------------------------------------------- files
def atomic_write(path: Path | str, writer: Callable[[Path], None]) -> None:
    """Call ``writer(tmp_path)`` to produce a file, then rename it onto ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        writer(tmp)
        # Flush file contents to disk before the rename, so a power cut cannot leave a
        # renamed but empty file.
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json(path: Path | str, obj) -> None:
    atomic_write(path, lambda t: t.write_text(json.dumps(obj, indent=2, sort_keys=True)))


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def save_npy(path: Path | str, arr: np.ndarray) -> None:
    def w(t: Path) -> None:
        with open(t, "wb") as fh:   # open by handle so numpy does not append ".npy"
            np.save(fh, arr)
    atomic_write(path, w)


def save_npz(path: Path | str, **arrays) -> None:
    def w(t: Path) -> None:
        with open(t, "wb") as fh:
            np.savez(fh, **arrays)   # uncompressed: fast to write and to read back
    atomic_write(path, w)


def torch_save(path: Path | str, obj) -> None:
    import torch
    atomic_write(path, lambda t: torch.save(obj, t))


# ------------------------------------------------------------------------ formatting
def fmt_dur(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "?"
    s = int(max(seconds, 0))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# ----------------------------------------------------------------------- progress
class Progress:
    """The shared progress record. Stages write it, the heartbeat thread reads it.

    ``fraction`` is the share of the *current stage* that is done (0 to 1). The ETA
    for the stage comes from the rate measured since the first update of this session,
    so a resumed stage (which starts at, say, 40%) is not judged by its earlier work.
    """

    def __init__(self, stage_names: list[str], planned: Callable[[str], float]):
        self.stage_names = stage_names
        self.planned = planned
        self.done: set[str] = set()
        self.stage: str | None = None
        self.fraction = 0.0
        self.detail = ""
        self._f0: float | None = None   # fraction at the first update of this session
        self._t0: float | None = None   # time of that first update
        self._lock = threading.Lock()

    def begin(self, stage: str) -> None:
        with self._lock:
            self.stage, self.fraction, self.detail = stage, 0.0, ""
            self._f0 = self._t0 = None

    def update(self, fraction: float, detail: str = "") -> None:
        with self._lock:
            self.fraction = float(min(max(fraction, 0.0), 1.0))
            self.detail = detail
            if self._f0 is None:
                self._f0, self._t0 = self.fraction, time.monotonic()

    def finish(self, stage: str) -> None:
        with self._lock:
            self.done.add(stage)

    def snapshot(self) -> dict:
        """Everything the heartbeat prints, computed under the lock."""
        with self._lock:
            stage, frac, detail = self.stage, self.fraction, self.detail
            f0, t0 = self._f0, self._t0
            done = set(self.done)
        # ETA of the current stage: measured rate once enough progress has been seen,
        # otherwise the planned duration scaled by the share still to do.
        stage_eta = None
        if stage is not None:
            planned = self.planned(stage)
            gained = (frac - f0) if f0 is not None else 0.0
            if f0 is not None and gained > 0.01:
                stage_eta = (time.monotonic() - t0) * (1.0 - frac) / gained
            else:
                stage_eta = planned * (1.0 - frac)
        # Later stages: their planned durations.
        later = 0.0
        started = stage is not None
        for name in self.stage_names:
            if name in done or name == stage:
                continue
            if started and self.stage_names.index(name) < self.stage_names.index(stage):
                continue   # skipped earlier stage (e.g. run with --stages)
            later += self.planned(name)
        run_eta = (stage_eta or 0.0) + later
        return dict(stage=stage, index=(self.stage_names.index(stage) + 1) if stage in self.stage_names else 0,
                    n_stages=len(self.stage_names), fraction=frac, detail=detail,
                    stage_eta_s=stage_eta, run_eta_s=run_eta)


# --------------------------------------------------------------------------- context
class RunContext:
    """Everything a stage needs: settings, progress, logging, checkpoint timing, stop flag."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.progress = Progress(cfg.stage_names, cfg.planned_for)
        self.t_start = time.monotonic()
        self._stop = threading.Event()
        self._last_ckpt = time.monotonic()
        cfg.path().mkdir(parents=True, exist_ok=True)
        self._log_file = open(cfg.path("run.log"), "a", buffering=1)

    # Logging -------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self._log_file.write(line + "\n")

    # Interrupts ----------------------------------------------------------------
    def request_stop(self) -> None:
        self._stop.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def check_stop(self) -> None:
        """For stages with nothing to save: exit at the next safe point if asked to."""
        if self._stop.is_set():
            raise StopRequested()

    # Checkpoints ---------------------------------------------------------------
    def checkpoint_due(self) -> bool:
        return time.monotonic() - self._last_ckpt >= self.cfg.checkpoint_interval_s

    def maybe_checkpoint(self, save_fn: Callable[[], None]) -> None:
        """Call once per loop iteration of a long stage.

        Saves when a checkpoint is due (10 minutes of wall time since the last one) or
        when an interrupt is pending. After saving for an interrupt it raises
        ``StopRequested`` so the run ends with all work preserved.
        """
        stopping = self._stop.is_set()
        if stopping or self.checkpoint_due():
            save_fn()
            self._last_ckpt = time.monotonic()
            self.log("checkpoint saved" + (" (interrupt)" if stopping else ""))
        if stopping:
            raise StopRequested()

    def reset_checkpoint_clock(self) -> None:
        self._last_ckpt = time.monotonic()


# -------------------------------------------------------------------------- heartbeat
class Heartbeat(threading.Thread):
    """Prints a status line (and writes ``status.json``) once per heartbeat interval."""

    def __init__(self, ctx: RunContext):
        super().__init__(daemon=True, name="heartbeat")
        self.ctx = ctx
        self._quit = threading.Event()

    def line(self) -> str:
        s = self.ctx.progress.snapshot()
        elapsed = time.monotonic() - self.ctx.t_start
        eta = s["run_eta_s"]
        finish = (dt.datetime.now() + dt.timedelta(seconds=eta)).strftime("%a %H:%M") if eta is not None else "?"
        return (f"HEARTBEAT stage {s['index']}/{s['n_stages']} {s['stage']} | {100 * s['fraction']:5.1f}% | "
                f"{s['detail']} | stage ETA {fmt_dur(s['stage_eta_s'])} | run ETA {fmt_dur(eta)} (about {finish}) | "
                f"elapsed {fmt_dur(elapsed)}")

    def beat(self) -> None:
        self.ctx.log(self.line())
        s = self.ctx.progress.snapshot()
        s["time"] = dt.datetime.now().isoformat(timespec="seconds")
        try:
            write_json(self.ctx.cfg.path("status.json"), s)
        except OSError:
            pass   # a failed status write must never stop the run

    def run(self) -> None:
        self.beat()
        while not self._quit.wait(self.ctx.cfg.heartbeat_interval_s):
            self.beat()

    def stop(self) -> None:
        self._quit.set()


# ----------------------------------------------------------------------- stage runner
def check_fingerprint(ctx: RunContext) -> None:
    """Refuse to continue if the settings differ from those of the saved state."""
    path = ctx.cfg.path("state", "fingerprint.json")
    fp = ctx.cfg.fingerprint()
    if path.exists():
        old = read_json(path)["fingerprint"]
        if old != fp:
            raise SystemExit(
                f"Settings have changed since this work directory was started "
                f"(fingerprint {old} -> {fp}). Resuming would mix incompatible state.\n"
                f"Run again with --reset to delete the saved state (downloads are kept), "
                f"or restore the old settings.")
    else:
        write_json(path, {"fingerprint": fp})


def stage_done_path(ctx: RunContext, stage: str) -> Path:
    return ctx.cfg.path("state", f"{stage}.done.json")


def run_stages(ctx: RunContext, stages: list[tuple[str, Callable[[RunContext], None]]],
               only: set[str] | None = None) -> None:
    """Run the stages in order, skipping those already finished."""
    check_fingerprint(ctx)
    for name, fn in stages:
        if only is not None and name not in only:
            continue
        marker = stage_done_path(ctx, name)
        if marker.exists():
            ctx.progress.finish(name)
            ctx.log(f"stage {name}: already finished, skipping")
            continue
        ctx.log(f"stage {name}: starting")
        ctx.progress.begin(name)
        t0 = time.monotonic()
        ctx.reset_checkpoint_clock()
        fn(ctx)                       # may raise StopRequested; state is then already saved
        ctx.progress.update(1.0, "done")
        write_json(marker, {"fingerprint": ctx.cfg.fingerprint(), "seconds": time.monotonic() - t0,
                            "finished": dt.datetime.now().isoformat(timespec="seconds")})
        ctx.progress.finish(name)
        ctx.log(f"stage {name}: finished in {fmt_dur(time.monotonic() - t0)}")


def configure_threads(cfg: Config) -> None:
    """Set the number of CPU threads PyTorch uses (0 keeps PyTorch's default).

    PyTorch's default is the number of physical cores, which is normally the fastest
    choice; hyper-threads rarely help matrix arithmetic.
    """
    import torch
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
