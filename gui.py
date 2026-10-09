#!/usr/bin/env python3
"""Start the generation GUI.

    python gui.py                 # uses the committed files in models/
    python gui.py --open          # also opens the page in your browser

The GUI continues a starting sequence of trade events by sampling new tokens from the
trained model, and draws every token as its icon (see docs/token-icons.md). It needs
only the files in ``models/`` (weights, tokeniser, prompt excerpts). If you have run the
study and the tokenised test month is on disk (``--data work``), you can also pick a
prompt at any time in that month.
"""

import argparse
import sys
import webbrowser
from pathlib import Path

from tape.generate import load_bundle
from tape.gui.server import FullData, serve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="models", help="directory with model.pt, tokeniser.json, prompt_excerpts.json")
    ap.add_argument("--data", default=None, help="work directory holding data/test_tokens.npy (optional)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the page in a browser")
    args = ap.parse_args()

    for f in ("model.pt", "tokeniser.json", "prompt_excerpts.json"):
        if not (Path(args.models) / f).exists():
            sys.exit(f"missing {Path(args.models) / f}. Run `python run.py` first, or restore the committed files.")
    bundle = load_bundle(args.models)
    data = None
    if args.data and (Path(args.data) / "data" / "test_tokens.npy").exists():
        data = FullData(Path(args.data))
    server = serve(bundle, data, args.host, args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"Generation GUI running at {url}  (Ctrl-C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
