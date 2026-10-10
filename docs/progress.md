# Progress

Design is complete: all decisions are settled and recorded in the [design log](design-log.md) and the design documents.

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
- [x] Write-up

## Not done, by design

- A lead test (does surprisal rise before volatility?). This is left for later.
- Scoring the two seeds that were not selected.
- Other assets, other months and other tokenisations.
