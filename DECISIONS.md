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

## Phase 1: time-based split

**Prediction (before running):**

**Explain-back:**

**Break-it exercise:**

**Decisions (Phase 1):**
- Cut points at 60% and 80% of the time range (fixed fractions, chosen without looking at fraud rates). With two 7-day gaps the true time shares are train 60.0%, validation 16.2%, test 16.2%, gaps 7.7%. Row shares: 64.5% / 14.2% / 14.3%, gaps 7.0% (41,399 rows).
- Validation ends at 80%, not 85%: with gaps, 85% leaves only ~20 days (~57k rows) each for validation and test, too few for stable fraud numbers at a ~3.5% fraud rate.
- 7-day gap, cut from the start of the later block, so train keeps its full 60%. Main reason: stop near-duplicate payments and fraud sprees at the border from leaking into evaluation. Trade-off: 3 days loses 3% of rows, 14 days loses 14%.
- The gap does NOT model label delay. Real chargeback delays are often weeks to months, so the newest training labels would be incomplete in practice, while ours are final. The gap is a simplification; real-world performance would likely be worse than measured.
- Split by fractions of time, not rows: each block is a clean stretch of calendar ("how good is the next month?"), which also makes drift (Phase 6) easier to read. Row counts are reported so the uneven traffic stays visible.
- DT_START / DT_END are hard-coded and a test checks the data still matches them, so a changed data file is caught instead of silently moving the boundaries.
- Boundaries live only in `src/split.py`; other code (notebooks, later phases) imports from there and never repeats the numbers. The `BOUNDS` test guards `split_by_time()` only, not copies elsewhere.
- Test set is never used for tuning; every choice in Phases 2-5 uses validation.
- Shared test fixture loads data once per run (22 s for 8 tests) and is read-only; it fingerprints key columns and errors at teardown if a test changed the data.

## Phase 2: LightGBM baseline

**Prediction (before running):** validation ROC-AUC about 0.90-0.95; validation PR-AUC about 0.5-0.7 (no-skill PR-AUC is about 0.035).

**Explain-back:**

**Break-it exercise:** in a scratch copy I left isFraud in the features. You predicted ~0.95 and concluded "suspect leakage". Result: validation ROC-AUC 1.0000, PR-AUC 1.0000, and early stopping ended after 1 tree: the model read the answer directly. Your conclusion was right; the score was even higher than predicted. A near-perfect score on rare fraud means check for leakage first.

**Result (seed 42, validation):** ROC-AUC 0.9189, PR-AUC 0.5691 (no-skill 0.5000 / 0.0366); best_iteration 557, 431 features. Your prediction (0.90-0.95, 0.5-0.7) was right on both. Top feature V258 holds 11.5% of gain, so none dominates. Training took ~55 s with 4 threads.

**Decisions (Phase 2):**
- Excluded isFraud (the answer), TransactionID (a row counter that proxies for time and memorizes rows) and TransactionDT (validation is later than all training times, so trees cannot extrapolate calendar position; time effects are Phase 6's job).
- Category columns go straight into LightGBM; no encoding, scaling or resampling. The category label lists come from the whole file (names only, no fitted statistics); LightGBM records the train categories itself.
- Early stopping on validation ROC-AUC, patience 100, best round kept, max 2000 trees, learning rate 0.05. The validation score is slightly optimistic because validation chose the tree count; the test set stays honest.
- Stopping metric and rule used here (validation ROC-AUC, patience 100, best round kept) must be reused for the neural net in Phase 3 so the comparison is fair. They are named constants in `src/lgbm.py`.
- Baseline is untuned (defaults + learning rate and tree cap). Phase 4 gives both models equal tuning effort on validation, or states plainly that the baseline is untuned.
- Determinism: seed 42 plus deterministic=True, force_row_wise=True and a fixed 4 threads (the thread count must stay fixed for repeatable results). Checked by a test.
- "Trained only on train" test reads the row count from the model's own first tree (`trees_to_dataframe`), because a check against rows our own function reports can be fooled by a bug in that function.
- The test block is discarded inside `train()` and never scored, counted or looked at in this phase.
