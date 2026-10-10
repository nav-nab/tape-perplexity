#!/usr/bin/env python3
"""Screenshot of the generation GUI for the paper (paper/figures/gui_example.png).

Start the GUI first (``python gui.py --port 8793``), then run

    python paper/make_gui_screenshot.py [--port 8793] [--excerpt volatile-1] [--events 18]

It needs Playwright with Chromium (``pip install playwright``). The picture shows the
real interface after one generation: the chosen excerpt as the starting sequence, two
sampled continuations (temperature 1, seed 1), and the events that really followed.
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="8793")
ap.add_argument("--excerpt", default="volatile-1")
ap.add_argument("--events", type=int, default=18)
ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "figures" / "gui_example.png"))
a = ap.parse_args()

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1180, "height": 800}, device_scale_factor=2)
    pg.goto(f"http://127.0.0.1:{a.port}/")
    pg.wait_for_selector("#source option", state="attached")
    pg.select_option("#source", a.excerpt)
    pg.wait_for_timeout(2000)
    pg.fill("#nNew", str(a.events))
    pg.fill("#nSamples", "2")
    pg.fill("#seed", "1")
    pg.click("#go")
    pg.wait_for_function("document.querySelector('#status').textContent.startsWith('Done')", timeout=90000)
    # scroll the tape so that the end of the prompt and all generated events are in view
    pg.evaluate("(n)=>{const b=document.getElementById('tapes');b.scrollLeft=511*32+130+n*32-b.clientWidth+12}", a.events)
    pg.wait_for_timeout(500)
    h = pg.evaluate("document.querySelector('main').getBoundingClientRect().bottom")
    pg.screenshot(path=a.out, clip={"x": 0, "y": 0, "width": 1180, "height": h - 20})
    b.close()
