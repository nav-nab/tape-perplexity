"""Unit tests for the pure parts: merge rule, tokeniser, statistics, model."""

import numpy as np
import pytest
import torch

from tape import tokeniser as tk
from tape.config import Config
from tape.model import GPT, ModelConfig
from tape.stages.events import records_to_events
from tape.stats import (block_bootstrap_corr, corr_from_sums, pearson, rolling_mean,
                        rolling_rms)

TICK, LOT = 0.01, 0.00001


# ----------------------------------------------------------------------- events
def test_worked_example_from_the_design_notes():
    """The four records of docs/tokeniser-design.md: three merge into one buy event."""
    a = np.array([5001, 5002, 5003, 5004])
    p = np.array([82000.10, 82000.11, 82000.13, 82000.12])
    q = np.array([0.0020, 0.0150, 0.0400, 0.0010])
    T = np.array([10**15 + 1] * 3 + [10**15 + 2])
    m = np.array([False, False, False, True])         # False = buyer aggressor
    ev = records_to_events(a, p, q, T, m, TICK, LOT)
    assert list(ev["side"]) == [1, 0]
    assert list(ev["size"]) == [5700, 100]             # 0.0570 BTC and 0.0010 BTC in lots
    assert list(ev["last"]) == [8200013, 8200012]
    # price move vs the previous event's last fill at 82,000.10
    assert ev["last"][0] - 8200010 == 3


def test_merge_requires_all_three_tests():
    a = np.arange(6)
    T = np.array([5, 5, 5, 5, 6, 6]) + 10**15
    # buyer, buyer (price up), buyer (price DOWN: not continuing), seller, seller, seller
    p = np.array([100.00, 100.01, 100.00, 100.00, 99.99, 99.99])
    q = np.ones(6) * 0.001
    m = np.array([False, False, False, True, True, True])
    ev = records_to_events(a, p, q, T, m, TICK, LOT)
    # events: [0,1] buy | [2] buy (price fell) | [3] sell | [4] sell at a new time | [5] sell same price (not strictly lower)
    assert list(ev["size"]) == [200, 100, 100, 100, 100]


def test_cleaning_drops_zero_price_and_quantity_and_sorts():
    a = np.array([3, 1, 2, 4])
    p = np.array([10.0, 10.0, 0.0, 10.0])
    q = np.array([1.0, 1.0, 1.0, 0.0])
    ev = records_to_events(a, p, q, np.array([10**15] * 4), np.zeros(4, bool), TICK, LOT)
    assert ev["size"].sum() == 200_000     # 2 BTC in lots: records a=1 and a=3 survive


def test_millisecond_timestamps_are_converted():
    ev = records_to_events(np.array([1]), np.array([1.0]), np.array([1.0]), np.array([1_700_000_000_000]),
                           np.array([False]), TICK, LOT)
    assert ev["t0"][0] == 1_700_000_000_000_000


# --------------------------------------------------------------------- tokeniser
def test_size_edges_are_exact_at_whole_lot_values():
    e = tk.size_edges_lots(5e-5, LOT, 3, 15)
    assert len(e) == 15
    for k, exact in ((3, 50), (6, 500), (9, 5000), (12, 50_000), (15, 500_000)):
        assert e[k - 1] == exact
    assert np.all(np.diff(e) > 0)


def _toy_tokeniser(lo=0, hi=15, median=104.0):
    return tk.Tokeniser(tk.size_edges_lots(5e-5, LOT, 3, 15), lo, hi, median)


def test_size_bin_boundaries():
    t = _toy_tokeniser()
    # 0.0005 BTC = 50 lots sits exactly on an edge and goes to the upper bin (bin 4, 0-based 3 -> 4th is 5e-4..1.08e-3)
    assert t.size_bin(np.array([49, 50, 10, 11, 1_000_000]))[1] == 3
    assert t.size_bin(np.array([49]))[0] == 2
    assert t.size_bin(np.array([10]))[0] == 0 and t.size_bin(np.array([11]))[0] == 1
    assert t.size_bin(np.array([10**9]))[0] == 15


def test_move_classes():
    t = _toy_tokeniser(median=104.0)
    d = np.array([-500, -104, -2, -1, 0, 1, 2, 104, 105])
    assert list(t.move_class(d)) == [0, 1, 1, 2, 3, 4, 5, 5, 6]


def test_encode_decode_round_trip():
    t = _toy_tokeniser(lo=1, hi=13)
    rng = np.random.default_rng(0)
    side = rng.integers(0, 2, 1000)
    size = rng.integers(1, 10**6, 1000)
    delta = rng.integers(-300, 300, 1000)
    ids = t.encode(side, size, delta)
    assert ids.max() < t.vocab == 2 * 13 * 7
    for i in range(0, 1000, 37):
        s, z, m = t.decode(ids[i])
        assert (s, z, m) == (side[i], t.size_bin(size[i:i + 1])[0], t.move_class(delta[i:i + 1])[0])


def test_tail_rule_merges_thin_outer_bins_only():
    edges = tk.size_edges_lots(5e-5, LOT, 3, 15)
    counts = np.array([5, 4, 300, 600, 900, 1000, 700, 500, 300, 100, 50, 30, 8, 2, 1, 0], float)
    t = tk.fit(edges, counts, np.array([2, 3, 10]), 0.01)
    total = counts.sum()
    # total 4500, so 1% is 45.
    # low end: 5+4 = 9 < 45, 9+300 >= 45 -> lo = 2
    # high end: 0, +1, +2, +8, +30 = 41 < 45, +50 = 91 >= 45 -> hi = 10
    assert (t.lo, t.hi) == (2, 10)
    assert t.n_size == 9 and t.vocab == 126
    assert counts[2] / total > 0.01


def test_tokeniser_save_load(tmp_path):
    t = _toy_tokeniser(lo=1, hi=12, median=77.5)
    t.save(tmp_path / "t.json")
    u = tk.Tokeniser.load(tmp_path / "t.json")
    assert (u.lo, u.hi, u.median_big) == (1, 12, 77.5) and np.allclose(u.edges, t.edges)


# ------------------------------------------------------------------------ stats
def test_rolling_mean_and_rms_match_naive():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    r = rolling_mean(x, 50)
    assert np.isnan(r[48]) and not np.isnan(r[49])
    assert np.allclose(r[49:], [x[t - 49:t + 1].mean() for t in range(49, 500)], atol=1e-5)
    rms = rolling_rms(x, 50)
    assert np.allclose(rms[100], np.sqrt((x[51:101] ** 2).mean()), atol=1e-5)


def test_rolling_mean_across_chunk_boundary(monkeypatch):
    import tape.stats as st
    monkeypatch.setattr(st, "CHUNK", 64)
    x = np.random.default_rng(2).normal(size=500)
    assert np.allclose(st.rolling_mean(x, 30)[29:], [x[t - 29:t + 1].mean() for t in range(29, 500)], atol=1e-5)


def test_bootstrap_matches_direct_resampling():
    """The prefix-sum shortcut must give the same replicates as literally resampling."""
    rng = np.random.default_rng(3)
    n, b, B = 1000, 130, 50
    x = np.cumsum(rng.normal(size=n)); y = 0.5 * x + np.cumsum(rng.normal(size=n))
    res = block_bootstrap_corr(x, y, b, B, seed=7)
    # Rebuild the same draws and recompute directly.
    r2 = np.random.default_rng(7)
    k = -(-n // b)
    starts = r2.integers(0, n - b + 1, size=(B, k))
    direct = []
    for row in starts:
        idx = np.concatenate([np.arange(s, s + b) for s in row])[:n]
        direct.append(np.corrcoef(x[idx], y[idx])[0, 1])
    assert np.allclose(res["reps"], direct, atol=1e-9)
    assert res["lo"] < res["rho"] < res["hi"]
    assert res["rho"] == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_bootstrap_chunking_does_not_change_result(monkeypatch):
    import tape.stats as st
    rng = np.random.default_rng(4)
    x = rng.normal(size=3000); y = x + rng.normal(size=3000)
    a = st.block_bootstrap_corr(x, y, 400, 30, 1)["reps"]
    monkeypatch.setattr(st, "CHUNK", 257)
    b = st.block_bootstrap_corr(x, y, 400, 30, 1)["reps"]
    assert np.allclose(a, b, atol=1e-9)


# ------------------------------------------------------------------------ model
def test_parameter_count_matches_design():
    m = GPT(ModelConfig(vocab=224))
    assert 2.7e6 < m.n_params() < 2.9e6


def test_model_is_causal():
    torch.manual_seed(0)
    m = GPT(ModelConfig(vocab=50, n_layer=2, d_model=32, n_head=4, context=64)).eval()
    x = torch.randint(0, 50, (2, 40))
    y = x.clone(); y[:, 25:] = (y[:, 25:] + 1) % 50          # change only the future
    with torch.no_grad():
        a, b = m(x), m(y)
    assert torch.allclose(a[:, :25], b[:, :25], atol=1e-5)    # earlier outputs unchanged
    assert not torch.allclose(a[:, 26:], b[:, 26:], atol=1e-5)


def test_last_k_equals_slice():
    m = GPT(ModelConfig(vocab=50, n_layer=2, d_model=32, n_head=4, context=64)).eval()
    x = torch.randint(0, 50, (3, 30))
    with torch.no_grad():
        assert torch.allclose(m(x)[:, -5:], m(x, last_k=5), atol=1e-6)


def test_config_fingerprint_ignores_paths_but_not_settings():
    c = Config.smoke()
    assert c.fingerprint() == c.replace(workdir="elsewhere", threads=3).fingerprint()
    assert c.fingerprint() != c.replace(M=301).fingerprint()
