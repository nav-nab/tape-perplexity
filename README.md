<p align="center"><img src="docs/assets/tape.svg" alt="A paper ticker tape of trade tokens: calm, then a volatile stretch, then calm" width="800"></p>

# tape-perplexity

Does a small transformer trained on BTCUSDT trades get more "confused" when the market becomes volatile?

Each trade event becomes a token. A causal transformer learns to predict the next token. *Perplexity* measures how surprised the model is by what actually happens next. We test whether the model's perplexity, averaged over a rolling window, rises and falls with realised volatility. This is a regime detector, not a price predictor.

**Status:** design complete. Implementation has not started, so there is no code to run yet.

## Where to start

Read Part 1 of the [project design](docs/project-design.md) for the question, the method and a pipeline diagram. The other documents hold the details.

## Docs

- [Project design](docs/project-design.md): the question, the pipeline, the signal, the evaluation, the planned generation GUI and the structure of the code.
- [Tokeniser design](docs/tokeniser-design.md): how raw Binance trades become tokens.
- [Token icons](docs/token-icons.md): the picture for each token, used in figures and the banner above.
- [Block bootstrap](docs/block-bootstrap.md): how the confidence interval is computed and why.
- [Design log](docs/design-log.md): every design decision as it was made.

## Progress

Design is complete: all decisions are settled and recorded in the docs above.

Implementation, in the order the single script runs it:

- [ ] Script skeleton: stage runner, 10-minute checkpoints, one-minute heartbeat
- [ ] Download, clean and merge into events
- [ ] Tokeniser (fitting and encoding)
- [ ] Training (three seeds, seed selection)
- [ ] Scoring the test month
- [ ] Evaluation (rolling series, correlation, block-bootstrap interval, plots)
- [ ] Export of the final weights, tokeniser and prompt excerpts
- [ ] Generation GUI

## License

See [LICENSE](LICENSE).
