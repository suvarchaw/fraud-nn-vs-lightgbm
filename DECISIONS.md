# Decisions log

For each phase, write:
- **Prediction (before running):** what I expect to see
- **Explain-back (my own words, 3-5 sentences):** what was built and why
- **Break-it exercise:** the flaw introduced and how I found it
- **Decisions:** non-obvious choices and the reason

## Phase 0: setup / EDA

**Decisions (Phase 0):**
- Python 3.12 via Homebrew: system 3.14 may lack LightGBM/torch wheels.
- float64->float32, int64->int32, object->category: memory only (2081 MB -> 977 MB). TransactionID/TransactionDT stay int64, never floats.
- Left join, not inner: not every transaction has identity data, and having identity is itself a signal (fraud 7.8% with vs 2.1% without).
- Parquet cache skipped; add only if CSV load time becomes a nuisance.
- Notebook shows aggregates only: outputs are saved in the file and the data is not redistributable.
- Phase 1 cut points are chosen by fixed fractions of the time range, never by looking at where fraud is higher or lower in the held-out period.
