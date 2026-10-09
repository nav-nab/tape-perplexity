"""Stage 1: download the daily Binance aggTrades files.

One zip per day from data.binance.vision, each next to a ``.CHECKSUM`` file holding its
SHA-256. A file is downloaded to a temporary name, checked against its checksum, and
only then renamed into place and marked verified. So an interrupted download never
leaves a corrupt file that looks complete, and a rerun resumes from the first day that
is not yet verified.
"""

from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path

from ..runtime import RunContext, atomic_write

USER_AGENT = "tape-perplexity/1.0 (research; data.binance.vision downloader)"


def raw_path(ctx: RunContext, day: str) -> Path:
    return ctx.cfg.path("raw", f"{ctx.cfg.symbol}-aggTrades-{day}.zip")


def verified_marker(zip_path: Path) -> Path:
    return zip_path.with_name(zip_path.name + ".verified")


def _fetch(url: str, dest: Path, retries: int = 5) -> None:
    """Download ``url`` to ``dest`` (atomically), retrying with growing pauses."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

            def write(tmp: Path) -> None:
                with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fh:
                    while chunk := r.read(1 << 20):
                        fh.write(chunk)
            atomic_write(dest, write)
            return
        except Exception as e:                     # network errors of any kind: retry
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not download {url}: {last}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    days = cfg.all_days
    for i, day in enumerate(days):
        ctx.check_stop()
        zip_path = raw_path(ctx, day)
        if verified_marker(zip_path).exists() and zip_path.exists():
            ctx.progress.update((i + 1) / len(days), f"day {day} present ({i + 1}/{len(days)})")
            continue
        url = f"{cfg.base_url}/{cfg.symbol}/{cfg.symbol}-aggTrades-{day}.zip"
        ctx.progress.update(i / len(days), f"downloading {day} ({i + 1}/{len(days)})")
        _fetch(url, zip_path)
        # The checksum file reads "<sha256>  <filename>".
        chk_path = zip_path.with_name(zip_path.name + ".CHECKSUM")
        _fetch(url + ".CHECKSUM", chk_path)
        expected = chk_path.read_text().split()[0].lower()
        actual = _sha256(zip_path)
        if expected != actual:
            zip_path.unlink()
            raise RuntimeError(f"checksum mismatch for {day}: expected {expected}, got {actual}")
        verified_marker(zip_path).write_text(actual)
        ctx.progress.update((i + 1) / len(days), f"downloaded {day} ({i + 1}/{len(days)})")
