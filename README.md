# tape-perplexity

Does a small transformer trained on BTCUSDT trades get more "confused" when the market becomes volatile?

Each trade event becomes a token. A causal transformer learns to predict the next token. We then test whether its perplexity tracks realised volatility. This is a regime detector, not a price predictor.

**Status:** design stage. No code yet.

## Docs

- [Project design](docs/project-design.md): the question, the pipeline, the signal and the evaluation.
- [Tokeniser design](docs/tokeniser-design.md): how raw Binance trades become tokens.
- [Design log](docs/design-log.md): decisions as they were made, and what is still open.

## Progress

- [x] Project design
- [x] Tokeniser design
- [ ] Study dates, window lengths and regime threshold
- [ ] Data download and tokeniser implementation
- [ ] Model training
- [ ] Evaluation

## License

See [LICENSE](LICENSE).
