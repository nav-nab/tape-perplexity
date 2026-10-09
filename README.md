<p align="center"><img src="docs/assets/tape.svg" alt="A paper ticker tape of trade tokens: calm, then a volatile stretch, then calm" width="800"></p>

# tape-perplexity

Does a small transformer trained on BTCUSDT trades get more "confused" when the market becomes volatile?

Each trade event becomes a token. A causal transformer learns to predict the next token. *Perplexity* measures how surprised the model is by what actually happens next. We test whether the model's perplexity, averaged over a rolling window, rises and falls with realised volatility. This is a regime detector, not a price predictor.

**Status:** design complete and code written. The code is tested on synthetic data and on a five-day real-data smoke run. The full study has not been run yet.

## Where to start

Read Part 1 of the [project design](docs/project-design.md) for the question, the method and a pipeline diagram. The other documents hold the details.

## Running it

```bash
pip install -r requirements.txt        # numpy, pandas, torch (CPU is enough), matplotlib, pytest
python run.py --smoke                  # tiny version of every stage; a few minutes; checks your setup
python run.py                          # the full study: about 7.5 hours on a laptop CPU
python gui.py --open                   # generation GUI; needs the files in models/
pytest                                 # unit and end-to-end tests
```

The full run downloads 91 daily Binance files, trains three seeds, scores the test month and writes the results to `results/`. The last two stages score the training period for a dependence check on the confidence interval. It can be interrupted at any time and resumed by running the same command again. It saves a checkpoint every 10 minutes and prints a status line with an ETA every minute. See section 10 of the project design.

## Docs

- [Project design](docs/project-design.md): the question, the pipeline, the signal, the evaluation, the generation GUI and the structure of the code.
- [Tokeniser design](docs/tokeniser-design.md): how raw Binance trades become tokens.
- [Token icons](docs/token-icons.md): the picture for each token, used in figures, the GUI and the banner above.
- [Block bootstrap](docs/block-bootstrap.md): how the confidence interval is computed and why.
- [Design log](docs/design-log.md): every design decision as it was made.

## Progress

Design is complete: all decisions are settled and recorded in the docs above.

- [x] Script skeleton: stage runner, 10-minute checkpoints, one-minute heartbeat
- [x] Download, clean and merge into events
- [x] Tokeniser (fitting and encoding)
- [x] Training (three seeds, seed selection)
- [x] Scoring the test month
- [x] Evaluation (rolling series, correlation, block-bootstrap interval, plots)
- [x] Export of the final weights, tokeniser and prompt excerpts
- [x] Generation GUI
- [ ] Full run on the 91 days of data
- [ ] Commit the final weights, tokeniser, excerpts and results

## License

See [LICENSE](LICENSE).
