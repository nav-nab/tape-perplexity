# Tape Perplexity: project design

**Status (2026-10-09):** design complete, no code yet. Every design decision is settled. The tokeniser is specified in [tokeniser-design.md](tokeniser-design.md), and the confidence interval is specified in [block-bootstrap.md](block-bootstrap.md).

Part 1 is a short overview. Part 2 holds the detailed design notes.

---

# Part 1. Overview

## The question

Can a small language-model-style network tell, from the raw sequence of trades alone, that the market has entered a volatile regime?

The test is whether the network's *perplexity* (how surprised it is by the next trade) rises and falls with realised volatility.

## The idea in four steps

1. **Turn each trade event into a token.** A token records who was aggressive (buyer or seller), how big the trade was, and how far the price moved. Time is deliberately left out.
2. **Train a small causal transformer** to predict the next token, as a language model predicts the next word. Training uses the earlier part of the study period.
3. **Measure surprise.** For every event, take the model's surprisal, $-\log p(\text{token}\mid\text{past})$. Average it over a rolling window to get a perplexity series.
4. **Compare with volatility.** On the later test period, check whether the perplexity series moves together with realised volatility.

## Pipeline at a glance

```mermaid
flowchart TD
    A["Binance BTCUSDT aggTrades<br/>raw records, microsecond timestamps"] --> B["Clean and order<br/>drop invalid records, sort by aggregate trade ID"]
    B --> C["Merge records into events<br/>one event = one taker order"]
    C --> D["Tokenise<br/>side, size bin, price-move class<br/>vocabulary up to 224, no time fields"]
    D --> E["Train causal transformer<br/>next-token loss, training period"]
    E --> F["Per-event surprisal<br/>test period"]
    F --> G["Rolling mean over N events<br/>perplexity series"]
    C --> H["Realised volatility over M events<br/>regime flag"]
    G --> I["Compare<br/>plot, correlation, block-bootstrap CI"]
    H --> I
```

## What this project is not

- It is **not a price predictor.** The model is only asked how surprised it is.
- It is **not (yet) an early-warning test.** The claim is *detection*: perplexity tracks the regime. Whether perplexity *leads* volatility is deferred. It will be examined only if the first result is strong.
- It is **not a trading strategy.** It makes no claim about profit.

## Why it might work, and why it might not

- *Might work:* in a calm market the sequence of sizes and price moves is regular, so the model predicts it well. In a stressed market the pattern changes, so the model should be more surprised.
- *Might not:* large price moves are themselves tokens, so some link between perplexity and volatility is expected almost by construction. This is accepted. The model is meant to learn the regime from the data. The question is how strong and stable the relationship is on the held-out test period.

---

# Part 2. Design notes

## 1. Goal and claim

- Use the model's "confusion" as a regime or volatility detector.
- The claim under test: on a held-out test period, rolling perplexity correlates with realised volatility.
- An early-warning test (does perplexity rise *before* volatility?) is out of scope for the first result.

## 2. Scope discipline

The final write-up has to be readable by a human. Every extra test, caveat or variant costs the reader attention. So extras are left out unless they change a conclusion.

Left out by default:

| Item | Why it is out |
|---|---|
| Early-warning (lead) tests | A forecasting claim, deferred until detection is shown |
| GARCH or trade-intensity baselines | A second model to explain |
| Order book data | Extra data and complexity; not needed for the claim |
| Multiple architectures or token schemes | A grid of results to explain |

## 3. Data

- **Source:** Binance spot, symbol BTCUSDT, aggregated trades ("aggTrades"). Daily and monthly files are published at data.binance.vision. The CSV files have no header row. The columns are `a, p, q, f, l, T, m, M` (in that order), with `m` and `M` written as `True` or `False`.
- **Why crypto:** the non-crypto sources with trade-level data cost money. BTC is the single instrument.
- **Period:** 1 September to 30 November 2025 (UTC), 91 days. Timestamps in files from 2025 on are in **microseconds**.
- **Split:** chronological. Training is September and October (61 days). Testing is November (30 days).
- **Why this period:** September 2025 is the calmest month since January 2025. The training months include the 10 October liquidation crash. November has several separate stressed stretches, so the test result does not rest on a single day. In a daily-volatility screen, 14 of the 30 test days were above the training 90th percentile.
- **Size:** the monthly files total about 1.5 GB compressed. The training period has 31 million events (measured). November is estimated at about 25 million, so the whole period holds about 55 million events. Process the files month by month.
- **Cleaning:** drop records with price 0 or quantity 0. Binance uses these to flag duplicates found in a data audit. Order the remaining records by aggregate trade ID.
- **Raw record fields** (names as in the Binance API):

| Field | Meaning |
|---|---|
| `a` | aggregate trade ID (ordering key) |
| `p` | price (USDT) |
| `q` | quantity (BTC) |
| `f`, `l` | first and last individual trade ID in the aggregate |
| `T` | timestamp (microseconds in recent data files) |
| `m` | buyer was the maker. If true, the aggressor was a **seller**. If false, a **buyer**. |
| `M` | best price match flag (not used) |

- **Instrument constants** (from Binance `exchangeInfo`): tick size 0.01 USDT; lot step and minimum quantity 0.00001 BTC; minimum notional 5 USDT.

## 4. Tokeniser

Specified in full in [tokeniser-design.md](tokeniser-design.md). In brief:

- One token per **event**, where an event is one taker order rebuilt from consecutive aggTrade records.
- A token has three fields: **side** (2 values), **size** (log-spaced bins, at most 16), **price move** (7 classes, measured in ticks).
- The three fields form one composite token ID. The vocabulary is at most $2\times16\times7=224$, with a single softmax.
- No time fields. Anything fitted uses the training period only.

## 5. Model

- A small causal (decoder-only) transformer, trained with next-token cross-entropy on the training period.
- **Size:** 6 layers, width 192, 6 attention heads, about 2.8 million parameters. Learned positions, pre-norm layers, GELU, tied input and output embeddings.
- **Context:** 512 events.
- **Compute:** the laptop (Intel i5-1038NG7, 4 cores) has no GPU that PyTorch can use, so training runs on CPU. Measured on random tokens: training runs at about 8,400 tokens/s, so one pass over 31 million events takes about 62 minutes. Forward-only scoring runs at about 27,000 tokens/s. One seed takes about 1.5 hours including scoring the test month, and three seeds about 4.7 hours.
- The vocabulary is small, so the output layer is cheap.

### Training

- **Objective:** next-token cross-entropy over the 224 token types. The loss per event is its surprisal, in nats.
- **One pass** over the training stream (31 million events), with no separate validation set. Each batch is new data, so the running loss on each batch is an honest held-out estimate. If the loss is still falling at the end, a further pass can be added.
- **Batches:** the training stream is cut into non-overlapping chunks of 512 events, and the chunks are shuffled. Each event is a prediction target once. Price moves are computed over the whole stream beforehand, not reset at chunk starts. The daily files join into one continuous stream, because the market is open 24/7.
- **Optimiser (standard defaults):** AdamW, peak learning rate $10^{-3}$ with a short warm-up and cosine decay to 10% of the peak, weight decay 0.1, gradient clipping at 1.0, about 16,000 tokens per step (about 1,900 steps per pass). No dropout.
- **Frozen model:** after training, the weights stay fixed for the whole test month.
- **Seeds:** three (0, 1 and 2). They share one recipe, and differ only in initial weights and chunk order.
- **Which model is used for the results:** one model, the seed with the lowest final training loss, is used for all surprisal results. The final training loss is the mean over the last part of the pass. It is a fair held-out estimate, because every batch in a single pass is new data. Choosing by training loss keeps the test month out of the choice. The other two seeds serve as a check that the loss is similar. If the three losses differ noticeably, that is discussed before going further.

## 6. The perplexity signal

For event $t$ with token $x_t$ and past tokens $x_{\lt t}$, the **surprisal** is

$$
s_t = -\log p_\theta(x_t \mid x_{\lt t}).
$$

The **rolling mean surprisal** over the last $N$ events is

$$
S_t = \frac{1}{N}\sum_{i=0}^{N-1} s_{t-i}.
$$

The **perplexity** is $\exp(S_t)$. The exponential is monotonic, so plots and correlations can use $S_t$ directly.

The window is $N=M=3000$ events. Both series then summarise exactly the same events, both are trailing windows ending at the same event, and there is no separate parameter to tune. If the surprisal series later proves much noisier than $\mathrm{RV}$, $N$ could be revisited using training data only.

### Scoring the test month

Every test event gets a surprisal from the frozen model, given the events just before it.

- **Why not plain chunks.** The model reads at most 512 events. If the test stream were cut into plain 512-event chunks, the first event of each chunk would have no history and the last would have 511. Surprisal falls as history grows, so the series would show a spurious sawtooth with a period of 512 events.
- **Sliding windows.** A 512-event window slides over the test stream with a **stride of 64** events. In each window, only the last 64 events are scored. Every scored event therefore has between 448 and 511 events of history. A stride of 1 would give exactly 511 for every event, but would take about 5.5 days on the laptop.
- **Month boundary.** The stream is continuous across the boundary between October and November. The first November events take their history from the last 511 October events. Those October events are context only and are not scored.
- **Cost.** About 200 million token-forwards for November, estimated at about 2 hours on the laptop CPU (range 2 to 3 hours). Only the selected model is scored.

## 7. The regime (volatility) signal

- Volatility is **realised volatility over the last $M=3000$ events**. Windows are in *event time*, not clock time. This matches the token stream, which also has no clock.
- The estimator is the **root mean square (RMS) of the per-event price moves**, in ticks:

$$
\mathrm{RV}_t=\sqrt{\frac1M\sum_{i=0}^{M-1}\Delta_{t-i}^2},
$$

  where $\Delta_t$ is the price move of event $t$ (see the tokeniser design). The squared moves average to an estimate of the variance of a price move. Large sweeps count as high volatility, which is intended for a crash regime.
- Bid-ask bounce adds a roughly constant offset to $\mathrm{RV}_t$, and heavy-tailed moves make it noisy. Both are accepted as properties of the setup.
- A **high-volatility regime** is a period when $\mathrm{RV}_t$ exceeds a threshold. The threshold is the **90th percentile of $\mathrm{RV}_t$ over all events in the training period**. The test period is judged against this fixed training threshold. The flag is used to shade regimes in the plots. The headline correlation does not depend on it.
- In a check on the 61 training days, the threshold came to about 251 ticks. Four properties are worth knowing:
  - The crash of 10 October supplies 45% of all flagged events. Excluding it would lower the threshold to about 222 ticks. It is kept, because the crash is part of the training data.
  - Flagged events are 10% of events but only about 2% of clock time, because trading is much busier in stressed periods.
  - The volatility level shifts between months. The September threshold (159 ticks) would flag 50% of October events. So the share of flagged events in the test month need not be 10%.
  - Most flagged episodes are short. 82% last under one window (3000 events), but they hold only 16% of flagged events. The 33 episodes of 10,000 events or more hold 56%.
- **Why $M=3000$.** It was chosen on 23 training days (none from the test month).
  - Below about 1000, one large sweep supplies a large share of a window's variance.
  - At 3000, the largest single event contributes about 5 to 8% of a window.
  - A window spans about 12 minutes on a calm day and about 30 seconds at the peak of the 10 October crash.
  - Adjacent non-overlapping windows agree with a correlation of about 0.6 in $\log\mathrm{RV}$ on ordinary days (median 0.61 over the 23 days).
  - Beyond 3000 the gains are small and calm-day windows start to exceed 40 minutes.

## 8. Evaluation

- Plot the perplexity series against realised volatility over the test period.
- Report the correlation between them on the test period.
- Give a **block-bootstrap confidence interval**, because consecutive events are strongly dependent. The method and the evidence for the block length are in [block-bootstrap.md](block-bootstrap.md). The block length is 1,000,000 events (about one trading day), with about 25 blocks in the test month.
- No other statistics are planned (see section 2).

## 9. Inference GUI (planned)

An inference GUI is built alongside the code. It lets a reader see the model at work on real trades. It is a demonstration and exploration tool. It is not part of the claim under test, so it must not change any result.

Intended contents:

- **Icon tape.** A scrollable row of the token icons (see [token-icons.md](token-icons.md)), one per event, with each event's surprisal underneath.
- **Next-token view.** For the selected event, the model's most likely next tokens, shown as icons with their probabilities.
- **Signal panel.** The rolling perplexity and the realised volatility over the same events, with the regime flag shaded.

The details below are not yet decided.

| Question | Options | Suggested default |
|---|---|---|
| Data source | Replay of recorded test-month data, or a live Binance feed | Replay. A live feed needs the same event-merge logic online, and adds nothing to the write-up. |
| Where the model runs | Live in the app, or precomputed surprisals read from a file | Both. Precomputed series for the signal panel, because scoring the month takes hours. A live forward pass for the tape and next-token view, because one 512-event window is small. |
| Technology | A Python web app (Streamlit or Gradio), a local server with a browser front end, or a static page with exported data | A local server with a browser front end. The icons are SVG, so a browser draws them natively. A static export is a good later step for sharing. |
| Interactivity | View only, or the user can also build a sequence by clicking icons | View only at first. Clickable sequences are a natural extension. |
| Code boundary | Inside the main package, or a separate module | A separate module that imports the tokeniser, model and scoring code. The GUI then cannot alter the study pipeline. |

## 10. Open decisions

The study design has no open decisions. The GUI details in section 9 are open. The remaining work is implementation. Two checks are planned for when a model exists:

- Repeat the dependence check for the block length on the surprisal series of the training period (see [block-bootstrap.md](block-bootstrap.md)).
- Read the loss by position within a chunk during training, to confirm that a stride of 64 gives scored events a consistent amount of history.

## 11. Design rationale

- **Why tokens are trade events.** A token describes who was aggressive, how large the trade was and how far the price moved. This is richer than a price or return level.
- **Why no time fields.** The model is not told how long passed between events. Its only evidence about the regime is the content of the trade sequence.
- **Why the model may use volatility information in the tokens.** The test asks whether the model can recognise a volatile regime. The only thing it can use is the data, so using it is the point of the test.
- **Why event time.** The token stream has no clock. Measuring the windows in events keeps the signal and the target on the same footing.
- **Why a chronological split.** A random split would leak future information into training.

## 12. Glossary

| Term | Meaning |
|---|---|
| aggTrade | Binance's aggregated trade record: fills of one taker order at one price and time, summed |
| aggressor / taker | The side that crossed the spread and removed liquidity |
| event | One taker order, rebuilt by merging consecutive aggTrade records |
| tick | Smallest price step, 0.01 USDT |
| surprisal | The negative log of the probability the model gave to the observed token |
| perplexity | The exponential of the mean surprisal over a window |
| regime | A stretch of high or low realised volatility |
| event time | Windows measured in numbers of events, not seconds |

## 13. Working agreements for this repo

- Commit locally after any substantive change. The owner pushes to the remote.
- Design decisions are logged in [design-log.md](design-log.md) as they are made.
- Facts that can be looked up (exchange constants, API semantics) are verified online, not assumed.
