<p align="center"><img src="docs/assets/tape.svg" alt="A paper ticker tape of trade tokens: calm, then a volatile stretch, then calm" width="800"></p>

# tape-perplexity

Does a small transformer trained on BTCUSDT trades get more "confused" when the market becomes volatile?

Each trade event becomes a token. A causal transformer learns to predict the next token. We then test whether its perplexity tracks realised volatility. This is a regime detector, not a price predictor.

**Status:** design complete. No code yet.

## Docs

- [Project design](docs/project-design.md): the question, the pipeline, the signal and the evaluation.
- [Tokeniser design](docs/tokeniser-design.md): how raw Binance trades become tokens.
- [Token icons](docs/token-icons.md): the visual icon for each token, used in figures and the logo.
- [Block bootstrap](docs/block-bootstrap.md): how the confidence interval is computed and why.
- [Design log](docs/design-log.md): decisions as they were made, and what is still open.

## Progress

- [x] Project design
- [x] Tokeniser design
- [x] Study period (1 Sep to 30 Nov 2025; train Sep–Oct, test Nov)
- [x] Volatility measure and window ($M=3000$ events)
- [x] Surprisal window ($N=M=3000$ events)
- [x] Regime threshold (90th percentile of training RV)
- [x] Model and training recipe (medium model, 3 seeds)
- [x] Test-time scoring (sliding windows, stride 64)
- [x] Bootstrap block length (1,000,000 events)
- [x] Token icon style (comet)
- [ ] Implementation: data pipeline, tokeniser, model, scoring, evaluation
- [ ] Data download and tokeniser implementation
- [ ] Model training
- [ ] Evaluation

## License

See [LICENSE](LICENSE).
