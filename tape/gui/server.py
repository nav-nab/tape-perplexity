"""A small local web server for the generation GUI (standard library only).

    GET  /                  the page (static/index.html)
    GET  /api/meta          vocabulary, size-bin labels, list of prompt excerpts
    GET  /api/excerpt?id=   one excerpt: its prompt and the events that really followed
    GET  /api/window?time=  (only if the tokenised data is on disk) a prompt at a given
                            UTC time in the test month
    POST /api/generate      sample continuations; body: {prompt, n_new, temperature, n_samples, seed}

The server only reads the files in ``models/`` (and, optionally, the tokenised test month
in the work directory). It cannot change the study pipeline or any of its outputs.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from ..generate import Bundle, generate, size_labels
from ..tokeniser import MOVE_NAMES

STATIC = Path(__file__).parent / "static"
MAX_NEW, MAX_SAMPLES = 512, 8
GEN_LOCK = threading.Lock()          # one generation at a time: the CPU is the bottleneck


class FullData:
    """Optional access to the whole tokenised test month, if it is on disk."""

    def __init__(self, workdir: Path):
        self.tokens = np.load(workdir / "data" / "test_tokens.npy", mmap_mode="r")
        self.t0 = np.load(workdir / "data" / "test_t0.npy", mmap_mode="r")

    def window(self, when: dt.datetime, prompt_len: int, cont_len: int) -> dict:
        us = int(when.replace(tzinfo=dt.timezone.utc).timestamp() * 1e6)
        i = int(np.searchsorted(self.t0, us))                     # first event at or after that time
        i = min(max(i, prompt_len), len(self.tokens) - cont_len)  # keep the whole excerpt inside the month
        stamp = lambda k: dt.datetime.fromtimestamp(int(self.t0[k]) / 1e6, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return dict(prompt=self.tokens[i - prompt_len:i].astype(int).tolist(),
                    continuation=self.tokens[i:i + cont_len].astype(int).tolist(),
                    t_start=stamp(i - prompt_len), t_split=stamp(i))


def make_handler(bundle: Bundle, data: FullData | None):
    tok, ex = bundle.tokeniser, bundle.excerpts
    by_id = {e["id"]: e for e in ex["excerpts"]}
    meta = dict(
        vocab=tok.vocab, n_size=tok.n_size, context=bundle.model.c.context, prompt_len=ex["prompt_len"],
        move_names=MOVE_NAMES, size_labels=size_labels(tok, bundle.lot),
        max_new=MAX_NEW, max_samples=MAX_SAMPLES, has_data=data is not None,
        excerpts=[dict(id=e["id"], kind=e["kind"], rv_ticks=e["rv_ticks"], t_start=e["t_start"],
                       t_split=e["t_split"], n_prompt=len(e["prompt"]), n_cont=len(e["continuation"]))
                  for e in ex["excerpts"]])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):          # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/meta":
                self._json(meta)
            elif u.path == "/api/excerpt":
                e = by_id.get(q.get("id", [""])[0])
                self._json(e if e else {"error": "unknown excerpt"}, 200 if e else 404)
            elif u.path == "/api/window" and data is not None:
                try:
                    when = dt.datetime.fromisoformat(q["time"][0].replace("T", " "))
                except (KeyError, ValueError):
                    return self._json({"error": "give time as YYYY-MM-DD HH:MM (UTC)"}, 400)
                self._json(data.window(when, ex["prompt_len"], ex["continuation_len"]))
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/api/generate":
                return self._json({"error": "not found"}, 404)
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                prompt = [int(t) for t in req["prompt"]][-bundle.model.c.context:]
                n_new = min(max(int(req.get("n_new", 64)), 1), MAX_NEW)
                n_samples = min(max(int(req.get("n_samples", 1)), 1), MAX_SAMPLES)
                with GEN_LOCK:
                    out = generate(bundle.model, prompt, n_new, float(req.get("temperature", 1.0)),
                                   n_samples, int(req.get("seed", 0)))
                self._json({"samples": out.tolist()})
            except (KeyError, ValueError, TypeError) as e:
                self._json({"error": str(e)}, 400)

    return Handler


def serve(bundle: Bundle, data: FullData | None, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(bundle, data))
