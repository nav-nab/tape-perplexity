# Design log

This file mirrors the running design log kept in the Claude project memory (`index.md`). It records each design decision as it was made. The detailed write-ups are in [project-design.md](project-design.md) and [tokeniser-design.md](tokeniser-design.md).

**Last synced with project memory:** 2026-10-09.

## Repo and process

- The git repo is `tape-perplexity` (branch `main`). Commit locally after any substantive change. The owner pushes to the remote.
- When a design decision is made, log it here and in the project memory, and update the two design documents if they are affected.

## Goal

- Laptop-scale ML research project in finance: train a small transformer on tokenised BTC trade sequences and test whether its perplexity ("model confusion") tracks volatility regimes. It is a regime detector, not a price predictor.
- The claim is detection (perplexity tracks regimes). A lead or early-warning test is left for later, only if the first result is strong.

## Data

- Binance spot BTCUSDT aggregated trades (data.binance.vision). Crypto was chosen because the non-crypto options cost money.
- Study period settled: 1 September to 30 November 2025 (UTC), timestamps in microseconds. Chronological split: September and October for training (61 days), November for testing (30 days). Reason: September is very calm, the training months include the 10 October crash, and November has several separate stressed stretches (14 of 30 test days above the training 90th percentile of a daily volatility proxy).
- Records with zero price or zero quantity are dropped.
- Training strictly before testing.

## Tokeniser

- No time fields in the tokens. Records are ordered by aggregate trade ID.
- Merge by default: consecutive records form one event (one taker order) when they have the same side and the same timestamp, and the price moves strictly in the aggressor's direction (higher for buys, lower for sells). The trade-ID contiguity test is dropped.
- Decision 1 settled: the price move of a merged event is its last fill price minus the previous event's last fill price, so it covers the order's whole impact.
- Decision 2 settled: the merge test is side, timestamp, and price strictly continuing in the aggressor's direction.
- Decision 3 settled: the price move is measured in ticks (0.01 USDT, whole numbers), not basis points.
- Decision 4 settled: magnitude bins per direction are: bin 1 is exactly 1 tick; bins 2 and 3 split the moves of 2 or more ticks at their training median. Edges are fit on absolute moves in the training period and shared between up and down moves.
- Decision 5, part 1 settled: size is measured in BTC (the native unit of the data), not USDT notional.
- Decision 5, part 2 settled: size bins are log-spaced, not quantile bins. Resolution is 3 bins per decade, with edges at $5\times10^{-5}\cdot10^{k/3}$ BTC. Tail rule: any top or bottom bin holding under about 1% of training events is merged into its neighbour, so the outermost bins are open-ended. Reason: log scale matches how humans perceive magnitude, fitting the "linguifying the tape" view of tokenisation.
- One composite token per event with three fields: side (2), size (log-spaced bins with tail merge, at most 16), price move (7: flat plus 3 magnitude bins per direction, i.e. $k=3$). Vocabulary at most $2\times16\times7=224$, single softmax.
- Only the tail-merge rule and the price-move magnitude edges are fit to the training period. The log size edges are fixed constants.

## Model, signal, evaluation

- Small causal transformer trained for next-token prediction.
- Model settled: medium, 6 layers, width 192, 6 heads, about 2.8M parameters, context 512 events. Three seeds (0, 1, 2) with one recipe. One model, chosen by the lowest final training loss (not by anything from the test month), is used for all surprisal results. The other seeds check that the losses are similar. Reason: the laptop is CPU-only (Intel i5-1038NG7, 4 cores); measured about 62 minutes per training pass and about 1.5 hours per seed including scoring, so about 4.7 hours for three seeds.
- Test-time scoring settled: sliding 512-event windows over the test stream with a stride of 64, scoring only the last 64 events of each window, so every scored event has 448 to 511 events of history (stride 1 would take about 5.5 days). The first November events take their history from the last 511 October events, which are context only. Estimated cost about 2 hours on the laptop CPU (25 million events, range 2 to 3 hours); only the selected model is scored.
- Training settled: one pass over the training stream, no separate validation set (each batch is new data), non-overlapping shuffled chunks of 512 events, AdamW with peak learning rate 1e-3, warm-up and cosine decay, weight decay 0.1, gradient clipping 1.0, about 16,000 tokens per step, no dropout, weights frozen for the test month.
- Primary signal: rolling mean surprisal over the last $N$ events (perplexity). Settled: $N=M=3000$, so both series summarise the same events with no extra tuned parameter.
- Volatility measure settled: root mean square of the per-event tick moves over the last $M$ events (plain realised volatility in event time). No jump-robust or noise-corrected variants. $M=3000$ events settled, chosen on 23 training days: one large sweep supplies 5 to 8% of a window's variance, a window spans about 12 minutes on a calm day, and adjacent windows agree with correlation about 0.6 (median 0.61) in log RV. Regime threshold settled: the 90th percentile of RV over all training events (about 251 ticks in a check on the 61 training days). The 10 October crash supplies 45% of flagged events (threshold would be about 222 ticks without it; it is kept). The flag is used for shading in plots and does not enter the headline correlation.
- Evaluation: plot perplexity against realised volatility over time, report the correlation on the test period, block-bootstrap confidence interval.
- Left out by default: lead tests, GARCH or trade-intensity baselines, order book data, multiple architectures or token schemes.

## Open decisions

- Project level: bootstrap block length.
- Tokeniser: all decisions settled (1 to 5).
