# Tape Perplexity: project design

**Status (2026-10-09):** design stage, no code yet. The tokeniser is fully specified in [tokeniser-design.md](tokeniser-design.md). The study dates, the two window lengths, the regime threshold and the model settings are still open (see [section 9](#9-open-decisions)).

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

- **Source:** Binance spot, symbol BTCUSDT, aggregated trades ("aggTrades"). Daily and monthly files are published at data.binance.vision. Check the column order of the files when loading them.
- **Why crypto:** the non-crypto sources with trade-level data cost money. BTC is the single instrument.
- **Period:** one contiguous stretch of 2 to 3 months, from 2025 onward. It should contain both calm and stressed stretches. Timestamps in files from 2025 on are in **microseconds**. The exact dates are open.
- **Split:** chronological. Training strictly precedes testing.
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
- Model size, context length and optimiser settings are not yet decided. They should be small enough for a laptop-scale budget.
- The vocabulary is small, so the output layer is cheap.

## 6. The perplexity signal

For event $t$ with token $x_t$ and past tokens $x_{<t}$, the **surprisal** is

$$ s_t = -\log p_\theta(x_t \mid x_{<t}). $$

The **rolling mean surprisal** over the last $N$ events is

$$ S_t = \frac{1}{N}\sum_{i=0}^{N-1} s_{t-i}. $$

The **perplexity** is $\exp(S_t)$. The exponential is monotonic, so plots and correlations can use $S_t$ directly.

The window $N$ trades noise (small $N$) against lag (large $N$). It is open.

## 7. The regime (volatility) signal

- Volatility is **realised volatility over the last $M$ events**. Windows are in *event time*, not clock time. This matches the token stream, which also has no clock.
- The exact estimator (for example the root mean square of per-event price moves) is fixed together with $M$.
- A **high-volatility regime** is a threshold on realised volatility. The candidate threshold is the 90th percentile of the training period.

## 8. Evaluation

- Plot the perplexity series against realised volatility over the test period.
- Report the correlation between them on the test period.
- Give a **block-bootstrap confidence interval**, because consecutive events are strongly dependent. The block length is open.
- No other statistics are planned (see section 2).

## 9. Open decisions

| Decision | Notes |
|---|---|
| Study dates | 2 to 3 months from 2025 onward, with calm and stressed stretches. Fixes how much data $N$ and $M$ can be chosen from. |
| Surprisal window $N$ | Noise versus lag |
| Volatility window $M$ and estimator | Event-time window |
| Regime threshold | Candidate: training 90th percentile |
| Model size and training settings | Not yet discussed |
| Bootstrap block length | Not yet discussed |

## 10. Design rationale

- **Why tokens are trade events.** A token describes who was aggressive, how large the trade was and how far the price moved. This is richer than a price or return level.
- **Why no time fields.** The model is not told how long passed between events. Its only evidence about the regime is the content of the trade sequence.
- **Why the model may use volatility information in the tokens.** The test asks whether the model can recognise a volatile regime. The only thing it can use is the data, so using it is the point of the test.
- **Why event time.** The token stream has no clock. Measuring the windows in events keeps the signal and the target on the same footing.
- **Why a chronological split.** A random split would leak future information into training.

## 11. Glossary

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

## 12. Working agreements for this repo

- Commit locally after any substantive change. The owner pushes to the remote.
- Design decisions are logged in [design-log.md](design-log.md) as they are made.
- Facts that can be looked up (exchange constants, API semantics) are verified online, not assumed.
