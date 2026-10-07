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

**Explain-back:** "Phase 2 included training a LightGBM model to be used as a baseline. We validated on ROC-AUC (one fraud and one legit payment) as 0.919 and PR-AUC (precision/recall) as 0.569. We also did a break-it exercise where we left the answer in a copy, which proves leakage." (Missing, to add next time: trained on train only, scored on validation, and validation is slightly optimistic because it picked the tree count.)

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

## Phase 3: neural network

**Prediction (before running):** "im not sure just guessing that it loses on roc-auc and on pr-auc" (no reason given).

**Explain-back:** "we trained a neural model on the same data and found out that lightgbm and the neural model cant train in the same program run. we also observed that peeking at future data doesnt change the score. in the one run we did the neural model lost to the lightgbm (neural model score for next month-0.879 light gbm- 0.919)" (To fix next time: peeking changed the score only a little (0.8788 -> 0.8791) but is still cheating, which is why tests, not scores, must catch it. Missing: averages come only from older months because the future is not available in real life, so using it flatters the score. The thread freeze is a side issue, not a main finding.)

**Break-it exercise:** in a scratch copy, the medians, means and standard deviations were computed from train + validation together. Your prediction: "hardly". Result: validation ROC-AUC 0.8788 -> 0.8791, PR-AUC 0.5116 -> 0.5164, so your prediction was right. Tests 2 (statistics equal a train-only recount; first mismatch: TransactionAmt median 4.2478 vs 4.2525) and 3 (changing validation must not change any statistic) both failed and caught it. Lesson: this leak is invisible in the score, so only a test catches it.

**Result (seed 42, validation):** ROC-AUC 0.8788, PR-AUC 0.5116 (LightGBM 0.9189 / 0.5691; no-skill 0.5000 / 0.0366). Train-sample (50,000 rows) ROC-AUC 0.9093, a small gap of 0.03. Best epoch 4 of 9 run, 28 s training, 274,187 weights, 895 inputs (400 numbers + 341 missing flags + 154 embedding numbers). Your prediction (loses on both) was right for this seed. One seed cannot say which model is better; Phase 4 measures the spread.

**Sanity checks:** overfit 1,000 train rows (dropout off, 300 epochs): train loss 0.0000, PASS. Shuffled labels: validation ROC-AUC 0.6321, which FAILED the pre-set rule 0.48-0.52. Investigation (scratch only): 20 untrained networks scored 0.29-0.68; 5 shuffled-label runs scored 0.26-0.70 on both sides of 0.5 (mean 0.45); shuffled labels overlapped true fraud 3.50% vs 3.42% by chance. Not a leak: the rule was wrong. A network that learned nothing still ranks payments by some random mix of the columns, and the columns relate to fraud, so its score lands far from 0.5 by accident. The +-0.005 "luck" estimate wrongly assumed scores unrelated to the columns. New rule (your choice): the shuffled-label score must not beat the best of 20 untrained networks by more than 0.05. 0.6321 <= 0.7254, PASS.

**Decisions (Phase 3):**
- Same rows (`split_by_time`), same columns (`feature_names` imported from `src/lgbm.py`, not copied), same metrics, no resampling, no class weights. Test set not touched.
- Every preprocessing statistic comes from train rows only: median, mean, standard deviation, which columns get a missing flag, and each category's label list. Validation only has them applied.
- Numbers: signed log on every number column (one rule, nothing fitted), blanks filled with the train median plus a 0/1 missing flag, then scaled with the train mean and standard deviation, then clipped to [-5, 5] so wild future values cannot swamp the sum.
- Category label lists are counted in train rows. The pandas category list from `load_raw()` covers the whole file (including validation-only labels), so it is not used.
- Labels seen fewer than 10 times in train share the "unknown" slot, so that slot is trained and new labels in the future get a learned meaning, not random numbers.
- Embedding size = min(16, (slots + 1) // 2), counting slots (labels + unknown + missing), not labels as first planned, so a column with only rare labels still gets size 1, not 0.
- ID-like columns stored as numbers (card1, addr1, ...) stay numbers, exactly as LightGBM saw them. This may disadvantage the network; changing it would be tuning (Phase 4).
- One configuration, fixed before any validation score: 256 -> 128, ReLU, dropout 0.3, Adam, learning rate 0.001, batch 1,024. Not tuned.
- Stopping rule reused from Phase 2: validation ROC-AUC, best kept, patience 5 epochs, cap 50. 100 trees and 5 epochs are not the same unit; what is shared is the rule.
- CPU, not the Mac GPU (MPS), with deterministic algorithms and 4 fixed threads, because GPU results are not guaranteed to repeat exactly. Seed 42.
- LightGBM and PyTorch each bring their own OpenMP thread library; after LightGBM has trained in a process, PyTorch on 4 threads freezes (deadlock). The network tests therefore run on 1 thread (still fixed, so same-seed still repeats); real runs are separate processes with 4 threads. Phase 4 must train the two models in separate processes.
- Train score is measured on a fixed 50,000-row train sample to keep it quick; it is only used to read the train-validation gap.
- No model file saved: Phase 4 retrains 5 seeds; Phase 7 saves the final model.

## Phase 4: equal tuning + multi-seed comparison

**Prediction (before running):** gap stays (no reason given).

**Verdict rule (fixed before any run, on val ROC-AUC over the 5 final seeds), checked in this order:**
1. "Flipped": the neural net's worst seed beats LightGBM's best seed.
2. "Gap closed": the mean difference is under 0.005, or the two models' min-max ranges overlap.
3. Otherwise "gap stayed".
`python -m src.compare report` prints which one applies.
