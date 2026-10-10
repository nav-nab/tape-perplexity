#!/usr/bin/env python3
"""Make every figure, number and table of the paper from the outputs of ``python run.py``.

    python paper/make_figures.py                  # run from the repository root
    python paper/make_figures.py --root /path     # repository root elsewhere

Inputs (all produced by run.py):  work/data/*.npy, work/regime.json, work/selected.json,
work/tokeniser.json, work/run.log, work/train/seed*/summary.json, results/results.json.

Outputs: paper/figures/*.pdf, paper/generated/numbers.tex (one LaTeX macro per quoted number)
and paper/generated/*.tex (tables). The text of the paper only quotes numbers through these
macros and tables, so the paper cannot disagree with the run.

Quantities that the study itself does not compute (the rarity benchmark, the previous-token
model, the per-decile tables, the entropy given the volatility level) are computed here. They
are descriptive checks on the test month and play no part in the design of the study.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ---------------------------------------------------------------------------- style
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": "#b9b8b3", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#ecebe7", "grid.linewidth": 0.7, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "legend.frameon": False, "pdf.fonttype": 42,
})
W = 6.5                                        # text width in inches


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def rolling_mean(x, n):
    cs = np.cumsum(np.concatenate(([0.0], x.astype(np.float64))))
    return (cs[n:] - cs[:-n]) / n                # element j is the mean of events j .. j+n-1


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--root", default=str(here.parent), help="repository root (holds work/ and results/)")
    ap.add_argument("--out", default=str(here), help="paper directory (figures/ and generated/ go here)")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "generated").mkdir(parents=True, exist_ok=True)
    fig_dir, gen = out / "figures", out / "generated"

    # ------------------------------------------------------------------ load
    R = json.loads((root / "results/results.json").read_text())
    reg = json.loads((root / "work/regime.json").read_text())
    sel = json.loads((root / "work/selected.json").read_text())
    tok = json.loads((root / "work/tokeniser.json").read_text())
    surp = np.load(root / "work/data/surprisal.npy")
    tt = np.load(root / "work/data/test_tokens.npy").astype(np.int64)
    tr = np.load(root / "work/data/train_tokens.npy").astype(np.int64)
    t0 = np.load(root / "work/data/test_t0.npy")
    rv_all = np.load(root / "work/data/rv.npy", mmap_mode="r")
    n_train, n_test = reg["n_train"], reg["n_test"]
    rv = np.asarray(rv_all[n_train:], dtype=np.float64)
    N, M = R["N"], R["M"]
    thr = reg["threshold_ticks"]
    V, ns = tok["vocab"], tok["n_size"]
    assert len(surp) == len(tt) == len(rv) == n_test

    # ------------------------------------------------------------ core series
    S = rolling_mean(surp, N)                                # aligned: S[j] ends at test event j+N-1
    y = np.log(np.maximum(rv[N - 1:], 1e-3))                 # log RV at the same events
    rvw = rv[N - 1:]
    flag = rvw > thr
    t_s = (t0[N - 1:] - 1761955200 * 10**6) / 1e6            # seconds since 2025-11-01 00:00 UTC
    r_model = corr(S, y)
    assert abs(r_model - R["correlation"]) < 1e-6, (r_model, R["correlation"])

    # rarity benchmark: surprisal of each event under the training-period token frequencies
    cnt = np.bincount(tr, minlength=V) + 0.5
    rar_tab = -np.log(cnt / cnt.sum())
    rar = rar_tab[tt].astype(np.float32)
    Rr = rolling_mean(rar, N)
    # previous-token model: counts of (previous, next) pairs over the whole training stream,
    # smoothed towards the token frequencies with strength alpha. alpha is chosen on the last
    # 15% of the training stream with the counts fitted on the first 85%, then the counts are
    # refitted on all of it.
    def pair_counts(x):
        return np.bincount(x[:-1] * V + x[1:], minlength=V * V).reshape(V, V).astype(np.float64)

    def pair_logp(pc, pr, prev_tok, next_tok, a):
        tot_ = pc.sum(1)
        return np.log((pc[prev_tok, next_tok] + a * V * pr[next_tok]) / (tot_[prev_tok] + a * V))

    cut = int(0.85 * len(tr))
    pc_a = pair_counts(tr[:cut])
    pr_a = (np.bincount(tr[:cut], minlength=V) + 0.5)
    pr_a = pr_a / pr_a.sum()
    alphas = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    held = {a_: float(-pair_logp(pc_a, pr_a, tr[cut - 1:-1], tr[cut:], a_).mean()) for a_ in alphas}
    a_s = min(held, key=held.get)
    prior = cnt / cnt.sum()
    pair = pair_counts(tr)
    p_prev = np.empty(n_test)
    p_prev[0] = prior[tt[0]]
    p_prev[1:] = np.exp(pair_logp(pair, prior, tt[:-1], tt[1:], a_s))
    prev = (-np.log(p_prev)).astype(np.float32)
    Rp = rolling_mean(prev, N)

    def ladder_row(series, per_event):
        r = corr(series, y)
        slope = r * series.std() / y.std()
        return dict(mean=float(per_event.mean()), r=r, slope=float(slope), sd=float(series.std()),
                    gap=float(series[flag].mean() - series[~flag].mean()))
    lad = {"rarity": ladder_row(Rr, rar), "prev": ladder_row(Rp, prev), "model": ladder_row(S, surp)}

    # per-field rarity (side, size, move)
    parts_tr = (tr // (ns * 7), (tr // 7) % ns, tr % 7)
    parts_te = (tt // (ns * 7), (tt // 7) % ns, tt % 7)
    field_r = {}
    for name, a, b, k in zip(("side", "size", "move"), parts_tr, parts_te, (2, ns, 7)):
        c = np.bincount(a, minlength=k) + 0.5
        field_r[name] = corr(rolling_mean((-np.log(c / c.sum()))[b].astype(np.float32), N), y)

    # ----------------------------------------------------------- hourly series
    hour = (t_s // 3600).astype(int)
    nh = hour.max() + 1
    hcount = np.bincount(hour, minlength=nh)
    hourly = lambda x: np.bincount(hour, x, nh) / np.maximum(hcount, 1)
    hS, hy, hfl = hourly(S), hourly(y), hourly(flag.astype(float))
    t00 = dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc)
    when = [t00 + dt.timedelta(hours=i + 0.5) for i in range(nh)]

    # ------------------------------------------------------------- deciles
    edges_q = np.quantile(y, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges_q, y, side="right") - 1, 0, 9)
    dS = np.array([S[dec == i].mean() for i in range(10)])
    dSsd = np.array([S[dec == i].std() for i in range(10)])
    dR = np.array([Rr[dec == i].mean() for i in range(10)])
    dRVlo, dRVhi = np.exp(edges_q[:-1]), np.exp(edges_q[1:])
    # entropy of the token distribution given the volatility decile BEFORE the event
    tt_i = np.arange(N, n_test)
    lag_y = np.log(np.maximum(rv[tt_i - 1], 1e-3))
    qd = np.quantile(lag_y, np.linspace(0, 1, 11))
    bdec = np.clip(np.searchsorted(qd, lag_y, side="right") - 1, 0, 9)
    ent = np.empty(10)
    for i in range(10):
        c = np.bincount(tt[tt_i][bdec == i], minlength=V).astype(float)
        p = c[c > 0] / c.sum()
        ent[i] = -(p * np.log(p)).sum()
    ent_all = float(sum(ent[i] * (bdec == i).mean() for i in range(10)))
    rar_ctx = float(rar[tt_i].mean())
    # entropy of the token distribution on the test month without conditioning on volatility
    c0 = np.bincount(tt[tt_i], minlength=V).astype(float)
    p0 = c0[c0 > 0] / c0.sum()
    ent_flat = float(-(p0 * np.log(p0)).sum())

    # --------------------------------------------------------------- daily
    day = (t_s // 86400).astype(int)
    nd = 30
    dayS = np.array([S[day == d].mean() for d in range(nd)])
    dayY = np.array([y[day == d].mean() for d in range(nd)])
    r_day = corr(dayS, dayY)

    # ---------------------------------------------------------- lag correlation
    lags = [-30000, -3000, 0, 3000, 30000]
    lagr = []
    for L in lags:
        a = S[max(L, 0): len(S) + min(L, 0)]
        b = y[max(-L, 0): len(y) + min(-L, 0)]
        lagr.append(corr(a, b))

    # ------------------------------------------------------------- per class
    mv, sz = parts_te[2], parts_te[1]
    cls_model = [float(surp[mv == i].mean()) for i in range(7)]
    cls_rar = [float(rar[mv == i].mean()) for i in range(7)]
    siz_model = [float(surp[sz == i].mean()) for i in range(ns)]
    siz_rar = [float(rar[sz == i].mean()) for i in range(ns)]
    mv_tr = np.bincount(parts_tr[2], minlength=7) / len(tr)
    sz_tr = np.bincount(parts_tr[1], minlength=ns) / len(tr)
    share_big = lambda m: float(((np.abs(parts_te[2] - 3) >= 2)[N - 1:])[m].mean())

    # --------------------------------------------------------------- training
    log = (root / "work/run.log").read_text()
    seeds = [0, 1, 2]
    summ = {s: json.loads((root / f"work/train/seed{s}/summary.json").read_text()) for s in seeds}
    curves = {s: np.array([(int(i), float(l)) for i, l in re.findall(rf"seed {s} step (\d+)/1901 loss ([\d.]+)", log)]) for s in seeds}

    # ================================================================ FIGURES
    def fmt_dates(ax):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))

    def shade(ax):
        on = hfl >= 0.2
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j < len(on) and on[j]:
                    j += 1
                ax.axvspan(t00 + dt.timedelta(hours=i), t00 + dt.timedelta(hours=j), color=ORANGE, alpha=0.28, lw=0)
                i = j
            else:
                i += 1

    # 1. time series
    fig, (a, b) = plt.subplots(2, 1, figsize=(W, 4.0), sharex=True)
    a.plot(when, hS, color=BLUE)
    a.set_ylabel("mean surprisal (nats)")
    b.plot(when, np.exp(hy), color=ORANGE)
    b.set_yscale("log")
    b.set_ylabel("realised volatility (ticks)")
    b.axhline(thr, color=GREY, ls="--", lw=0.9)
    b.text(when[3], thr * 1.07, "training 90th percentile", color=INK2, fontsize=8)
    for ax in (a, b):
        shade(ax)
    fmt_dates(b)
    fig.tight_layout()
    fig.savefig(fig_dir / "series.pdf")
    plt.close(fig)

    # 2. relation: deciles and days
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.9))
    x = np.arange(1, 11)
    a.errorbar(x, dS, yerr=dSsd, color=BLUE, marker="o", ms=4, capsize=2.5, lw=1.2)
    a.set_xticks(x)
    a.set_xlabel("volatility decile (1 = calmest)")
    a.set_ylabel("mean surprisal (nats)")
    a.set_title("(a) By volatility decile")
    b.scatter(np.exp(dayY), dayS, s=22, color=BLUE, edgecolor="white", lw=0.8, zorder=3)
    b.set_xscale("log")
    b.set_xlabel("daily geometric-mean volatility (ticks)")
    b.set_ylabel("daily mean surprisal (nats)")
    b.set_title("(b) One point per test day")
    b.text(0.05, 0.9, f"$r={r_day:.2f}$", transform=b.transAxes)
    fig.tight_layout()
    fig.savefig(fig_dir / "relation.pdf")
    plt.close(fig)

    # 3. benchmark
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
    a.plot(x, dR, color=GREY, marker="o", ms=4, label="rarity benchmark")
    a.plot(x, dS, color=BLUE, marker="o", ms=4, label="transformer")
    mid = 6
    a.annotate("", xy=(mid, dS[mid - 1]), xytext=(mid, dR[mid - 1]), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9))
    a.text(mid + 0.2, (dS[mid - 1] + dR[mid - 1]) / 2, f"{dR[mid - 1] - dS[mid - 1]:.1f} nats", color=INK2, va="center", fontsize=8)
    a.set_xticks(x)
    a.set_xlabel("volatility decile (1 = calmest)")
    a.set_ylabel("mean surprisal (nats)")
    a.set_title("(a) Surprisal by volatility decile")
    a.legend(loc="lower right")
    names = ["side", "size", "price\nmove", "whole\ntoken", "trans-\nformer"]
    vals = [field_r["side"], field_r["size"], field_r["move"], lad["rarity"]["r"], lad["model"]["r"]]
    b.bar(range(5), vals, color=[GREY, GREY, GREY, "#5a5955", BLUE], width=0.6)
    for i, v in enumerate(vals):
        b.text(i, v + (0.02 if v >= 0 else -0.07), f"{v:.2f}", ha="center", fontsize=8, color=INK)
    b.set_xticks(range(5))
    b.set_xticklabels(names, fontsize=8)
    b.set_ylim(-0.12, 0.95)
    b.set_ylabel("correlation with log volatility")
    b.set_title("(b) Correlation with volatility")
    b.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(color=GREY, label="rarity benchmark"), Patch(color=BLUE, label="transformer")],
             loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "benchmark.pdf")
    plt.close(fig)

    # 4. appendix: training
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.7))
    cols = [BLUE, ORANGE, GREEN]
    for s in seeds:
        a.plot(curves[s][:, 0], curves[s][:, 1], color=cols[s], lw=1.1, label=f"seed {s}")
        lp = np.array(summ[s]["loss_by_position"])
        k = 8
        b.plot(np.arange(len(lp) - k + 1) + k // 2 + 1, np.convolve(lp, np.ones(k) / k, "valid"), color=cols[s], lw=1.1)
    a.set_ylim(3.0, 4.2)
    a.set_xlabel("training step")
    a.set_ylabel("training loss (nats)")
    a.set_title("(a) Training loss")
    a.legend()
    b.set_xscale("log")
    b.set_xlabel("position in the 512-event window")
    b.set_ylabel("mean training loss (nats)")
    b.set_title("(b) Loss by position")
    fig.tight_layout()
    fig.savefig(fig_dir / "training.pdf")
    plt.close(fig)

    # 5. appendix: checks
    bc = R["block_length_check"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.7))
    a.plot([d["block_len"] for d in bc], [d["se"] for d in bc], color=BLUE, marker="o", ms=4)
    a.set_xscale("log")
    a.axvline(R["block_len"], color=GREY, ls="--", lw=0.9)
    a.set_ylim(0, 0.055)
    a.set_xlabel("block length (events)")
    a.set_ylabel("bootstrap standard error")
    a.set_title("(a) Block length")
    b.plot(np.array(lags) / 1000, lagr, color=BLUE, marker="o", ms=4)
    b.set_xlabel("shift (thousand events)")
    b.set_ylabel("correlation")
    b.set_ylim(0.3, 0.6)
    b.set_title("(b) Shifted series")
    fig.tight_layout()
    fig.savefig(fig_dir / "checks.pdf")
    plt.close(fig)

    # 6. appendix: token classes
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.8), gridspec_kw={"width_ratios": [1, 1.3]})
    w = 0.38
    xm = np.arange(7)
    a.bar(xm - w / 2, cls_rar, w, color=GREY, label="rarity benchmark")
    a.bar(xm + w / 2, cls_model, w, color=BLUE, label="transformer")
    a.set_xticks(xm)
    a.set_xticklabels(["down\nlarge", "down\nmid", "down\n1 tick", "flat", "up\n1 tick", "up\nmid", "up\nlarge"], fontsize=7)
    a.set_ylabel("mean surprisal (nats)")
    a.set_title("(a) By price move")
    a.set_ylim(0, 7.6)
    a.legend(loc="upper center", fontsize=7.5, ncol=2)
    a.grid(axis="x", visible=False)
    xs = np.arange(ns)
    b.bar(xs - w / 2, siz_rar, w, color=GREY)
    b.bar(xs + w / 2, siz_model, w, color=BLUE)
    b.set_xticks(xs)
    b.set_xticklabels(xs + 1)
    b.set_xlabel("size bin (1 = smallest)")
    b.set_title("(b) By trade size")
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "classes.pdf")
    plt.close(fig)

    # 7. appendix: knowing only the volatility level
    fig, a = plt.subplots(figsize=(W * 0.62, 2.9))
    a.plot(x, [rar[tt_i][bdec == i].mean() for i in range(10)], color=GREY, marker="o", ms=4, label="rarity benchmark")
    a.plot(x, ent, color=ORANGE, marker="o", ms=4, label="knows the volatility level only")
    a.plot(x, [surp[tt_i][bdec == i].mean() for i in range(10)], color=BLUE, marker="o", ms=4, label="transformer")
    a.set_xticks(x)
    a.set_xlabel("volatility decile before the event (1 = calmest)")
    a.set_ylabel("mean surprisal (nats)")
    a.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "level_only.pdf")
    plt.close(fig)

    # 8. token key: the comet icons
    from matplotlib.patches import FancyBboxPatch, Circle, Polygon
    BUYC, SELLC, TILE = "#2563EB", "#E8590C", "#E9ECF2"
    T_LEN = [14, 8, 4, 0, 4, 8, 14]
    DIRN = [-1, -1, -1, 0, 1, 1, 1]
    rows = sorted(set([0, ns // 4, ns // 2, (3 * ns) // 4, ns - 1]))
    pitch = 38.0
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.95))
    for k, (ax, col, ttl) in enumerate(zip(axs, (BUYC, SELLC), ("(a) Buy orders", "(b) Sell orders"))):
        ax.set_xlim(-6, 7 * pitch - 2)
        ax.set_ylim(len(rows) * pitch - 4, -2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(ttl, loc="left", color=col)
        for ri, sbin in enumerate(rows):
            for m in range(7):
                x0, y0 = m * pitch, ri * pitch
                ax.add_patch(FancyBboxPatch((x0, y0), 32, 32, boxstyle="round,pad=0,rounding_size=5", fc=TILE, ec="none"))
                r_, T, d = 1.6 + 0.42 * sbin, T_LEN[m], DIRN[m]
                cy = 16 - d * T / 2
                tip = cy + d * (r_ + T)
                if d != 0:
                    ax.add_patch(Polygon([(x0 + 16 - 0.8 * r_, y0 + cy), (x0 + 16 + 0.8 * r_, y0 + cy), (x0 + 16, y0 + tip)],
                                         closed=True, fc=col, ec="none", alpha=0.5))
                ax.add_patch(Circle((x0 + 16, y0 + cy), r_, fc=col, ec="none"))
        labels = ["down\nlarge", "down\nmid", "down\n1 tick", "flat", "up\n1 tick", "up\nmid", "up\nlarge"]
        for m in range(7):
            ax.text(m * pitch + 16, len(rows) * pitch - 1, labels[m], ha="center", va="top", fontsize=6.5, color=INK2)
        if k == 0:
            for ri, sbin in enumerate(rows):
                ax.text(-2, ri * pitch + 16, f"bin {sbin + 1}", ha="right", va="center", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(fig_dir / "token_key.pdf")
    plt.close(fig)

    def tex(v):
        v = str(v)
        return "$-$" + v[1:] if v.startswith("-") else v          # a proper minus sign in running text

    # ================================================================ NUMBERS
    def pct(v, d=1):
        return f"{100 * v:.{d}f}\\%"
    fl_mean, nf_mean = float(S[flag].mean()), float(S[~flag].mean())
    lvl_gain = rar_ctx - ent_all            # level-only model, fitted on the test month, vs the benchmark
    shift_gain = rar_ctx - ent_flat         # part of that due to train-to-test change in token frequencies
    lvl_pure = ent_flat - ent_all           # part due to knowing the volatility level
    mod_gain = rar_ctx - float(surp[tt_i].mean())
    n = {
        "NTrain": f"{n_train:,}".replace(",", "\\,"), "NTest": f"{n_test:,}".replace(",", "\\,"),
        "NEval": f"{R['n_events']:,}".replace(",", "\\,"),
        "WinN": N, "WinM": M, "Thr": f"{thr:.0f}", "Vocab": V, "NSize": ns,
        "Corr": f"{R['correlation']:.2f}", "CorrLo": f"{R['ci_low']:.2f}", "CorrHi": f"{R['ci_high']:.2f}",
        "CorrDay": f"{r_day:.2f}", "FlagShare": pct(R["flagged_share_test"]),
        "MeanSurp": f"{R['mean_surprisal_nats']:.2f}", "MeanRar": f"{lad['rarity']['mean']:.2f}",
        "SurpFlag": f"{fl_mean:.2f}", "SurpRest": f"{nf_mean:.2f}", "SurpGap": f"{fl_mean - nf_mean:.2f}",
        "DecNine": f"{dS[8]:.2f}", "DecTen": f"{dS[9]:.2f}", "DecLow": f"{dS[0]:.1f}", "DecHigh": f"{dS[8]:.1f}", "DecTop": f"{dS[9]:.1f}",
        "GapLow": f"{dR[0] - dS[0]:.1f}", "GapHigh": f"{dR[9] - dS[9]:.1f}",
        "GapAvg": f"{lad['rarity']['mean'] - lad['model']['mean']:.1f}",
        "RarCorr": f"{lad['rarity']['r']:.2f}", "PrevCorr": f"{lad['prev']['r']:.2f}",
        "PrevMean": f"{lad['prev']['mean']:.2f}", "ModCorr": f"{lad['model']['r']:.2f}",
        "FieldSide": f"{field_r['side']:.2f}", "FieldSize": f"{field_r['size']:.2f}", "FieldMove": f"{field_r['move']:.2f}",
        "PrevAlpha": f"{a_s:g}",
        "PrevShare": pct((lad['rarity']['mean'] - lad['prev']['mean']) / (lad['rarity']['mean'] - lad['model']['mean']), 0),
        "LevelGain": f"{lvl_gain:.3f}", "ShiftGain": f"{shift_gain:.3f}", "LevelPure": f"{lvl_pure:.3f}",
        "ModelGain": f"{mod_gain:.2f}", "LevelShare": pct(lvl_pure / mod_gain, 0), "EntFlat": f"{ent_flat:.2f}",
        "EntLow": f"{ent[0]:.2f}", "EntHigh": f"{ent[9]:.2f}",
        "BigCalm": pct(np.mean([share_big(dec == 0)]), 0), "BigVol": pct(np.mean([share_big(dec == 9)]), 0),
        "SeedBest": sel["seed"], "SeedSpread": f"{sel['spread']:.4f}",
        "LossBest": f"{min(sel['final_losses'].values()):.3f}",
        "BlockLen": f"{R['block_len']:,}".replace(",", "\\,"), "NBoot": f"{R['n_boot']:,}".replace(",", "\\,"),
        "SEUsed": f"{[d['se'] for d in bc if d['block_len'] == R['block_len']][0]:.3f}",
        "Params": f"{summ[0]['parameters']:,}".replace(",", "\\,"), "Steps": summ[0]["steps"],
        "TrainMinutes": f"{np.mean([summ[s]['train_seconds'] for s in seeds]) / 60:.0f}",
    }
    (gen / "numbers.tex").write_text("% Generated by make_figures.py. Do not edit.\n" + "".join(
        f"\\newcommand{{\\{k}}}{{{tex(v)}}}\n" for k, v in n.items()))

    # ================================================================ TABLES
    def write(name, lines):
        (gen / name).write_text("\n".join(lines) + "\n")

    write("tab_ladder.tex", [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Scoring rule & Mean surprisal & Correlation & Slope & Flagged minus rest \\",
        r" & (nats) & with $\log\mathrm{RV}$ & (nats per unit) & (nats) \\", r"\midrule",
        *[f"{lab} & {lad[k]['mean']:.2f} & {lad[k]['r']:.2f} & {lad[k]['slope']:.2f} & {lad[k]['gap']:.2f} \\\\"
          for lab, k in (("Token rarity (no context)", "rarity"), ("Previous token only", "prev"), ("Transformer (512 events)", "model"))],
        r"\bottomrule", r"\end{tabular}"])
    write("tab_deciles.tex", [
        r"\begin{tabular}{rrrrr}", r"\toprule",
        r"Decile & RV range (ticks) & Transformer & Rarity & Gap \\", r"\midrule",
        *[f"{i + 1} & {dRVlo[i]:.0f}--{dRVhi[i]:.0f} & {dS[i]:.2f} & {dR[i]:.2f} & "
          f"{dR[i] - dS[i]:.2f} \\\\" for i in range(10)],
        r"\bottomrule", r"\end{tabular}"])
    side_tex = tex("%.2f" % field_r["side"])
    write("tab_fields.tex", [
        r"\begin{tabular}{lr}", r"\toprule", r"Rarity of the token's \ldots & Correlation with $\log\mathrm{RV}$ \\", r"\midrule",
        f"side & {side_tex} \\\\", f"price move & {field_r['move']:.2f} \\\\",
        f"trade size & {field_r['size']:.2f} \\\\", f"whole token & {lad['rarity']['r']:.2f} \\\\",
        r"\bottomrule", r"\end{tabular}"])
    write("tab_seeds.tex", [
        r"\begin{tabular}{rrrr}", r"\toprule", r"Seed & Final training loss (nats) & Training time (min) & Selected \\", r"\midrule",
        *[f"{s} & {sel['final_losses'][str(s)]:.4f} & {summ[s]['train_seconds'] / 60:.0f} & {'yes' if s == sel['seed'] else ''} \\\\" for s in seeds],
        r"\bottomrule", r"\end{tabular}"])
    write("tab_blocks.tex", [
        r"\begin{tabular}{rrr}", r"\toprule", r"Block length & Std. error & 95\% interval \\", r"\midrule",
        *[f"{d['block_len']:,} & {d['se']:.3f} & [{d['ci_low']:.2f}, {d['ci_high']:.2f}] \\\\".replace(",", "\\,") for d in bc],
        r"\bottomrule", r"\end{tabular}"])
    lag_keys = sorted(R["acf_log_rv"], key=int)
    write("tab_acf.tex", [
        r"\begin{tabular}{rrr}", r"\toprule", r"Lag & $\log\mathrm{RV}$ & Surprisal \\", r"\midrule",
        *[f"{int(k):,} & {R['acf_log_rv'][k]:.2f} & {R['acf_surprisal_mean'][k]:.2f} \\\\".replace(",", "\\,") for k in lag_keys],
        r"\bottomrule", r"\end{tabular}"])
    # token vocabulary: size bins and move classes with training shares
    edges = [0.0] + list(tok["edges"])
    lot = 1e-5
    size_rows = []
    for i in range(ns):
        lo = edges[tok["lo"] + i] if i > 0 else None
        hi = edges[tok["lo"] + i + 1] if i < ns - 1 else None
        rng_ = ("below " + f"{hi * lot:.2g}" if lo is None else (f"above {lo * lot:.2g}" if hi is None else f"{lo * lot:.2g}--{hi * lot:.2g}"))
        size_rows.append(f"{i + 1} & {rng_} & {pct(sz_tr[i], 1)} \\\\")
    write("tab_sizes.tex", [r"\begin{tabular}{rrr}", r"\toprule", r"Bin & Size range (BTC) & Share of training events \\", r"\midrule",
                            *size_rows, r"\bottomrule", r"\end{tabular}"])
    mnames = ["down, large", "down, mid", "down, 1 tick", "flat", "up, 1 tick", "up, mid", "up, large"]
    write("tab_moves.tex", [r"\begin{tabular}{lr}", r"\toprule", r"Price move & Share of training events \\", r"\midrule",
                            *[f"{mnames[i]} & {pct(mv_tr[i], 1)} \\\\" for i in range(7)], r"\bottomrule", r"\end{tabular}"])
    print("wrote figures and tables to", out)
    print(json.dumps({k: str(v) for k, v in n.items()}, indent=1))


if __name__ == "__main__":
    main()
