"""End-to-end tests on synthetic Binance-format files (no network needed).

* the full smoke pipeline runs and produces every artefact;
* a run killed with SIGTERM during training resumes and ends with the same weights as an
  uninterrupted run;
* a changed setting refuses to resume.
"""

import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from synthetic import write_synthetic_days  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DAYS = ["2025-09-01", "2025-09-02", "2025-09-03", "2025-09-04", "2025-09-05"]


def run(args, cwd, wait=True):
    cmd = [sys.executable, str(ROOT / "run.py"), "--smoke", "--threads", "2"] + args
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not wait:
        return p
    out, _ = p.communicate(timeout=900)
    return p.returncode, out


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("rawdata")
    write_synthetic_days(d, DAYS, n_events=40_000)
    return d


def make_run_dir(base: Path, raw: Path, name: str) -> Path:
    d = base / name
    (d / "work_smoke").mkdir(parents=True)
    shutil.copytree(raw, d / "work_smoke" / "raw")
    return d


def test_full_smoke_pipeline(tmp_path, raw_dir):
    d = make_run_dir(tmp_path, raw_dir, "full")
    code, out = run(["--heartbeat-seconds", "1"], d)
    assert code == 0, out[-3000:]
    assert "HEARTBEAT" in out and "run ETA" in out          # heartbeat with an ETA was printed
    for f in ("model.pt", "tokeniser.json", "prompt_excerpts.json"):
        assert (d / "models_smoke" / f).exists()
    res = json.loads((d / "results_smoke" / "results.json").read_text())
    assert -1 <= res["correlation"] <= 1 and res["ci_low"] <= res["ci_high"]
    assert (d / "results_smoke" / "timeseries.png").stat().st_size > 5000
    ex = json.loads((d / "models_smoke" / "prompt_excerpts.json").read_text())
    assert len(ex["excerpts"]) >= 2 and all(len(e["prompt"]) == ex["prompt_len"] for e in ex["excerpts"])
    # token ids are valid for the fitted vocabulary
    vocab = json.loads((d / "models_smoke" / "tokeniser.json").read_text())["vocab"]
    assert max(max(e["prompt"]) for e in ex["excerpts"]) < vocab
    # every test event was scored with a finite, positive surprisal
    s = np.load(d / "work_smoke" / "data" / "surprisal.npy")
    assert np.isfinite(s).all() and (s > 0).all()
    # the block-length check ran on the test month and produced its table and figure
    assert len(res["block_length_check"]) >= 3 and all(b["se"] > 0 for b in res["block_length_check"])
    assert (d / "results_smoke" / "block_length.png").exists()
    # a second run does nothing (all stages finished)
    code, out = run([], d)
    assert code == 0 and out.count("already finished") == 11


def test_interrupt_and_resume_matches_uninterrupted(tmp_path, raw_dir):
    ref = make_run_dir(tmp_path, raw_dir, "ref")
    cut = make_run_dir(tmp_path, raw_dir, "cut")
    small = ["--set", "tokens_per_step=256", "--set", "seeds=(0,)"]    # many small steps, one seed
    stages_to_train = ["--stages", "download,events,tokenise,volatility,train_seed0"]
    code, out = run(small + stages_to_train, ref)
    assert code == 0, out[-2000:]

    p = run(small + stages_to_train + ["--checkpoint-seconds", "1"], cut, wait=False)
    log = cut / "work_smoke" / "run.log"
    deadline = time.time() + 300
    while time.time() < deadline:                                      # wait until training has started
        if log.exists() and "parameters" in log.read_text():
            break
        time.sleep(0.2)
    time.sleep(1.5)
    p.send_signal(signal.SIGTERM)
    out, _ = p.communicate(timeout=120)
    assert p.returncode == 130, out[-2000:]
    assert (cut / "work_smoke" / "train" / "seed0" / "ckpt.pt").exists()
    assert not (cut / "work_smoke" / "train" / "seed0" / "final.pt").exists()

    code, out = run(small + stages_to_train, cut)
    assert code == 0 and "resumed from checkpoint" in out, out[-2000:]

    a = torch.load(ref / "work_smoke/train/seed0/final.pt", weights_only=False)["state_dict"]
    b = torch.load(cut / "work_smoke/train/seed0/final.pt", weights_only=False)["state_dict"]
    assert all(torch.allclose(a[k], b[k], atol=1e-4) for k in a)


def test_changed_settings_refuse_to_resume(tmp_path, raw_dir):
    d = make_run_dir(tmp_path, raw_dir, "fp")
    code, _ = run(["--stages", "download,events"], d)
    assert code == 0
    code, out = run(["--stages", "download,events", "--set", "M=301"], d)
    assert code != 0 and "Settings have changed" in out


def test_generation_and_gui_api(tmp_path, raw_dir):
    """Runs after the pipeline has produced models/: generation is deterministic and the API works."""
    import threading
    import urllib.request
    from tape.generate import generate, load_bundle
    from tape.gui.server import serve

    d = make_run_dir(tmp_path, raw_dir, "gui")
    code, out = run([], d)
    assert code == 0, out[-2000:]
    bundle = load_bundle(d / "models_smoke")
    prompt = bundle.excerpts["excerpts"][0]["prompt"]
    a = generate(bundle.model, prompt, 10, 1.0, 3, seed=5)
    b = generate(bundle.model, prompt, 10, 1.0, 3, seed=5)
    assert a.shape == (3, 10) and (a == b).all() and a.max() < bundle.tokeniser.vocab
    greedy = generate(bundle.model, prompt, 5, 0.0, 2, seed=1)
    assert (greedy[0] == greedy[1]).all()                      # temperature 0 is deterministic
    long = generate(bundle.model, prompt, bundle.model.c.context + 5, 1.0, 1, seed=2)   # slides past the context
    assert long.shape == (1, bundle.model.c.context + 5)

    server = serve(bundle, None, "127.0.0.1", 0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    meta = json.loads(urllib.request.urlopen(base + "/api/meta").read())
    assert meta["vocab"] == bundle.tokeniser.vocab and len(meta["size_labels"]) == meta["n_size"]
    ex = json.loads(urllib.request.urlopen(base + "/api/excerpt?id=" + meta["excerpts"][0]["id"]).read())
    assert ex["prompt"] == prompt
    req = urllib.request.Request(base + "/api/generate", method="POST",
                                 data=json.dumps(dict(prompt=prompt, n_new=6, n_samples=2, seed=3)).encode())
    got = json.loads(urllib.request.urlopen(req).read())
    assert len(got["samples"]) == 2 and len(got["samples"][0]) == 6
    assert b"Tape generator" in urllib.request.urlopen(base + "/").read()
    server.shutdown()
