# Tape Perplexity: project design

**Status (snapshot 2026-10-09):** design stage, no code yet. The tokeniser is fully specified (see [tokeniser-design.md](tokeniser-design.md)). Four project-level parameters are still open (see [Open decisions](#10-open-decisions)).

**Who this is for:** a human or an agent picking the project up cold. Part 1 is a short overview. Part 2 holds the detailed design notes, numbered so they can be cited.

---

# Part 1. Overview

## The question

Can a small language-model-style network tell, from the raw sequence of trades alone, when the market has entered a volatile regime?

The test is whether the network's *perplexity* (how surprised it is by the next trade) rises and falls with realised volatility.

## The idea in four steps

1. **Turn each trade event into a token.** A token records who was aggressive (buy or sell), how big the trade was, and how far the price moved. Time is deliberately left out.
2. **Train a small causal transformer** to predict the next token, as a language model would predict the next word. Training uses the earlier part of the study period.
3. **Measure surprise.** For every event, take the model's surprisal, $-\log p(\text{token}\mid\text{past})$. Average it over a rolling window to get a perplexity series.
4. **Compare with volatility.** On a later test period, check whether the perplexity series moves together with realised volatility.

## Pipeline at a glance

```
 Binance BTCUSDT aggTrades (raw records, microsecond timestamps)
              |
              |  clean: drop invalid records, order by aggregate trade ID
              v
 merge records that belong to one taker order  ->  "events"
              |
              |  tokenise: side x size bin x price-move class
              v
 token sequence (vocabulary <= 224, no time fields)
              |
              |  train on the training period (causal transformer, next-token loss)
              v
 per-event surprisal on the test period
              |                                         realised volatility over M events
              |  rolling mean over N events                          |
              v                                                      v
 "perplexity" series  ------------ compare (plot, correlation, block-bootstrap CI) ------------ volatility / regime flag
```

## What this project is not

- It is **not a price predictor.** The model sees the market only to be surprised by it.
- It is **not (yet) an early-warning test.** The claim is *detection*: perplexity tracks the regime. Whether perplexity *leads* volatility is deferred, and is examined only if the first result is strong.
- It is **not a trading strategy** and makes no claim about profit.

## Why this might work, and why it might not

- *Might work:* in a calm market the sequence of sizes and price moves is regular, so the model predicts it well. In a stressed market the pattern changes, and a model trained on mixed history should be more surprised.
- *Might not:* volatility itself is partly visible in the tokens (large price moves are tokens). Perplexity could then track volatility only trivially. This is accepted as part of the claim: the model is supposed to learn regime from the data. The headline result is the strength and stability of the relationship on the held-out test period.

## Current state in one table

| Area | State |
|---|---|
| Data source and fields | Chosen: Binance spot BTCUSDT aggTrades |
| Study period | Open (2 to 3 months, from 2025 onward) |
| Tokeniser | Settled (all five decisions) |
| Model | Small causal transformer; sizes not yet fixed |
| Surprisal window $N$ | Open |
| Volatility window $M$ | Open |
| Regime threshold | Open (candidate: training 90th percentile) |
| Evaluation | Plot, correlation, block-bootstrap CI |

---

# Part 2. Design notes

## 1. Goal and claim

- The goal is to use "model confusion" as a regime or volatility detector.
- The claim under test: on a held-out test period, the rolling perplexity of the model correlates with realised volatility.
- A lead or early-warning test (does perplexity rise *before* volatility?) is out of scope for the first result.

## 2. Scope discipline

The final write-up must be readable by a human. Every extra test, caveat or variant costs the reader attention. So the default is to leave extras out unless they change a conclusion the reader must trust.

Left out by default:

| Item | What it would add | Why it is out |
|---|---|---|
| Lead / early-warning tests | A forecasting claim | Deferred until the detection result is strong |
| GARCH or trade-intensity baselines | A comparison point for the signal | Adds a second model to explain |
| Order book data | Richer state | Cost of data and complexity; not needed for the claim |
| Multiple architectures or token schemes | Robustness | Adds a grid of results to explain |

If a decision is made to add one of these, record it in the decision log (section 11).

## 3. Data

- **Source:** Binance spot, symbol BTCUSDT, aggregated trades ("aggTrades"). Daily and monthly files are published at data.binance.vision. Check the column order of the files when loading them.
- **Why crypto:** the non-crypto options with usable trade-level data cost money. BTC was chosen over other coins to keep the project to one instrument.
- **Period:** one contiguous stretch of 2 to 3 months, from 2025 onward. It should contain both calm and stressed stretches. Timestamps in the files from 2025 on are in **microseconds**. Exact dates are open.
- **Split:** chronological. Training strictly precedes testing. No shuffling across the boundary.
- **Cleaning:** drop records with price 0 or quantity 0. These are duplicates flagged by Binance (they also carry first and last trade ID of -1). Records are ordered by aggregate trade ID.
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

- **Instrument constants (verified from Binance `exchangeInfo`):** tick size 0.01 USDT; lot step and minimum 0.00001 BTC; minimum notional 5 USDT.

## 4. Tokeniser

Summarised here, specified in full in [tokeniser-design.md](tokeniser-design.md).

- One token per **event**. An event is one taker order, built by merging consecutive aggTrade records.
- A token has three fields: **side** (2), **size** (log-spaced bins, at most 16), **price move** (7 classes in ticks). The composite token ID is one number, and the model has a single softmax over it. Vocabulary is at most $2\times16\times7=224$.
- No time fields. Bin edges that are fitted use the training period only.

## 5. Model

- A small causal (decoder-only) transformer, trained with next-token cross-entropy on the training period.
- Model size, context length and optimiser settings are **not yet decided**. They should be small enough to train on a laptop-scale budget.
- The vocabulary is small (at most 224), so the output layer is cheap.

## 6. The perplexity signal

For event $t$ with token $x_t$ and past tokens $x_{<t}$, the **surprisal** is

$$ s_t = -\log p_\theta(x_t \mid x_{<t}). $$

The **rolling mean surprisal** over the last $N$ events is

$$ S_t = \frac{1}{N}\sum_{i=0}^{N-1} s_{t-i}. $$

The **perplexity** is $\exp(S_t)$. Because $\exp$ is monotonic, correlations and plots can use $S_t$ directly.

The window $N$ is open. It trades noise (small $N$) against lag (large $N$).

## 7. The regime (volatility) signal

- Volatility is **realised volatility over the last $M$ events**, so it is measured in *event time*, not clock time. This matches the token sequence, which also has no clock.
- The exact estimator (for example the root mean square of per-event price moves) is fixed together with $M$. It is open.
- A **high-volatility regime** is defined by a threshold on realised volatility. The candidate is the 90th percentile of the training period. This is open.

## 8. Evaluation

- Plot the perplexity series against realised volatility over the test period.
- Report the correlation between them on the test period.
- Give a **block-bootstrap confidence interval**, because consecutive events are strongly dependent. Block length is not yet chosen.
- No other statistics are planned by default (see section 2).

## 9. Design rationale worth remembering

- **Why tokens are trades, not returns or prices.** A trade event carries side, size and impact together. Returns alone throw away who was aggressive and how much traded.
- **Why no time fields.** The model is not given the time between events. Its only evidence about regime is the content of the trade sequence.
- **Why the model may use volatility information in the tokens.** The test asks whether the model can recognise a volatile regime. Information from the data is the only thing it can use, so using it is not cheating.
- **Why event time.** The token stream has no clock. Measuring volatility and windows in events keeps the signal and the target on the same footing.
- **Why chronological split.** It avoids leakage from future to past, which random splits would introduce.

## 10. Open decisions

| Decision | Notes |
|---|---|
| Study period | 2 to 3 months from 2025 onward; must include calm and stressed stretches. Choosing it fixes how much data $N$ and $M$ are chosen from. |
| $N$, surprisal window | Noise versus lag |
| $M$ and the volatility estimator | Event-time window |
| Regime threshold | Candidate: training 90th percentile |
| Model size and training settings | Not yet discussed |
| Bootstrap block length | Not yet discussed |

## 11. Decision log (chronological)

1. Focus on detection of volatility regimes, not price prediction.
2. Tokens encode a trade, not a price or return.
3. Strip time fields from tokens.
4. Data: crypto only (non-crypto options cost money), BTC, Binance spot BTCUSDT aggTrades.
5. Three token fields: side, size, price move. (Size binning was first quantile-based and was later replaced by log-spaced bins.)
6. Price move: $k=3$ magnitude bins per direction (7 classes in all).
7. Merge Binance aggregate records into events by default (verified: 58.4% of records sit in multi-record events in a live sample).
8. Tokeniser decisions 1 to 5 settled. See the decision log in [tokeniser-design.md](tokeniser-design.md).

The authoritative running log lives in the project memory (`index.md`). This section is a snapshot.

## 12. Glossary

| Term | Meaning |
|---|---|
| aggTrade | Binance's aggregated trade record: fills of one taker order at one price and time, summed |
| aggressor / taker | The side that crossed the spread and removed liquidity |
| event | One taker order, rebuilt by merging consecutive aggTrade records |
| tick | Smallest price step, 0.01 USDT |
| surprisal | $-\log$ probability the model gave the observed token |
| perplexity | $\exp$ of the mean surprisal over a window |
| regime | A stretch of high or low realised volatility |
| event time | Measuring windows in numbers of events, not seconds |

## 13. Working agreements for this repo

- Commit locally after any substantive change. The owner pushes to the remote.
- Design decisions are logged in the project memory when they are made.
- Facts that can be looked up (exchange constants, API semantics) are verified online, not assumed.
