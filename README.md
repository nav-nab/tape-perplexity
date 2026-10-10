<p align="center"><img src="docs/assets/tape.svg" alt="A paper ticker tape of trade tokens: calm, then a volatile stretch, then calm" width="800"></p>

# tape-perplexity

Does a small transformer trained on BTCUSDT trades get more "confused" when the market becomes volatile?

Each trade event becomes a token. A causal transformer learns to predict the next token. *Perplexity* measures how surprised the model is by what actually happens next. We test whether the model's perplexity, averaged over a rolling window, rises and falls with realised volatility. This is a regime detector, not a price predictor.

**Status:** the study has been run on the 91 days of data, and the results are written up in the [paper](paper/main.pdf). The model's surprise rises with volatility, with a test-month correlation of 0.56. A benchmark that scores events only by token rarity shows that part of this link is built in.

## Where to start

Read the [paper](paper/main.pdf) for the question, the method and the results. Part 1 of the [project design](docs/project-design.md) gives the pipeline and the reasoning behind each design choice. The other documents hold the details.

## Running it

```bash
pip install -r requirements.txt        # numpy, pandas, torch (CPU is enough), matplotlib, pytest
python run.py --smoke                  # tiny version of every stage; a few minutes; checks your setup
python run.py                          # the full study: about 6 hours on a laptop CPU
python gui.py --open                   # generation GUI; needs the files in models/
pytest                                 # unit and end-to-end tests
python paper/make_figures.py           # rebuild the paper's figures, numbers and tables from the run outputs
make -C paper                          # compile paper/main.pdf (needs LaTeX with latexmk)
```

The full run downloads 91 daily Binance files, trains three seeds, scores the test month and writes the results to `results/`. It can be interrupted at any time and resumed by running the same command again. It saves a checkpoint every 10 minutes and prints a status line with an ETA every minute. See section 10 of the project design.

## Docs

- [Paper](paper/main.pdf): the write-up. The TeX source, bibliography and figure script are in `paper/`.
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
- [x] Full run on the 91 days of data
- [x] Commit the final weights, tokeniser, excerpts and results
- [x] Write-up (draft)

## License

See [LICENSE](LICENSE).
