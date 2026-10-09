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

**Explain-back:** "we retrained on fresh seeds to give both neural network and lightgbm an equal chance and phase 3 was just first guesses. gap stayed means that earlier the gab was about 0.04 bw lgbm and nn so after phase 4 also they improved the same little amount which means that gap of 0.04 still stayed. the scores are slightly optimistic so 5 seed doesnt tell us the actual number bcs we havent looked at the test set" (To fix next time: equal tuning gave the equal chance; fresh seeds remove the winner's curse, since the best trial won partly because seed 42 suited it, and both winners did drop on new seeds. "Gap stayed" was right; under the rule, the network's best seed is still below LightGBM's worst. The main thing the spread misses: all seeds share one time split and one validation month, so it cannot show whether another month would reorder the models. Optimism from tuning on validation is true, but it is a separate caveat.)

**Result (validation, 5 fresh seeds each):**
| | LightGBM | neural net |
|---|---|---|
| baseline (trial 1, seed 42) ROC-AUC / PR-AUC | 0.9189 / 0.5691 | 0.8788 / 0.5116 |
| best trial (seed 42) ROC-AUC / PR-AUC | 0.9273 / 0.5891 (trial 13) | 0.8906 / 0.5237 (trial 8) |
| 5 seeds ROC-AUC mean +- std (min-max) | 0.9260 +- 0.0011 (0.9245-0.9270) | 0.8874 +- 0.0034 (0.8821-0.8899) |
| 5 seeds PR-AUC mean +- std (min-max) | 0.5827 +- 0.0033 (0.5801-0.5881) | 0.5217 +- 0.0026 (0.5177-0.5243) |
| tuning compute (20 trials) | 872 s | 887 s |

Verdict (rule above): **gap stayed**. Mean gap LightGBM - network: ROC-AUC 0.0387 (untuned: 0.0401), PR-AUC 0.0609 (untuned: 0.0575). The ROC-AUC gap is about 11 times the larger seed spread, and the network's best seed is below LightGBM's worst. Tuning helped both by a similar amount (ROC-AUC +0.0071 LightGBM, +0.0086 network). Your prediction (gap stays) was right. Total run time 41 min.

**Decisions (Phase 4):**
- Validation is used for BOTH tuning (choosing the settings) and early stopping (choosing the tree/epoch count). Both validation scores are therefore slightly optimistic, for both models. The untouched test set gives the honest number in the final report.
- Random search for both: the same simple method, no new library, and no trial depends on earlier scores, so neither model benefits from a smarter search adapting to it.
- 20 trials per model: trial 1 is the exact Phase 2/3 config, 19 are random draws. 19 random trials give about a 62% chance (1 - 0.95^19) of landing in the best 5% of the space. One fixed search seed (0).
- 6 knobs each, learning rate tuned for both. Ranges centred on library defaults / Phase 2-3 values (reasons in the Phase 4 plan and `src/compare.py`).
- Winner chosen on val ROC-AUC (the stopping metric). PR-AUC is recorded, never used to choose.
- Tuning uses seed 42; the winner is retrained on fresh seeds 1-5. Reason: the winner may have won partly because seed 42 was lucky for it (winner's curse). Seen here: both winners scored lower on fresh seeds (LightGBM 0.9273 -> mean 0.9260, network 0.8906 -> mean 0.8874).
- What the seed spread does NOT cover: all seeds share one time split and one validation month (a different period could reorder the models; Phase 6 looks at drift); a different search seed could pick a different winner; preprocessing and features are fixed.
- No p-value: 5 seeds on one split are not independent samples of the world, so a significance test would overstate certainty.
- Winners at a range edge (outer 10%; for 3-choice lists, the first or last choice): LightGBM num_leaves 218, reg_lambda 5.1; network lr 0.00287, width 512, batch 2048. Ranges were NOT widened afterwards (that would be extra tuning for one model), so both scores are lower bounds. The network has more edge knobs, all pointing to "bigger/faster", so it may have more headroom; a 3-choice list flags 2 of 3 values, so that signal is weak.
- No trial or seed hit the 2,000-tree / 50-epoch cap.
- Compute came out nearly equal (872 s vs 887 s). The plan guessed LightGBM trials would take twice as long; column sampling made them faster than the 55 s baseline.
- `subsample_freq=1` is always passed: LightGBM silently ignores `subsample` without it, and with subsample 1.0 it changes nothing (trial 1 reproduced 0.9189 exactly; the network's trial 1 reproduced 0.8788).
- Each model trains in its own process; files pass results along, and `report` imports neither library and refuses to compare unless trial counts, train/validation row fingerprints and seeds match.
- New finding: the clash also runs the other way. If PyTorch's thread pool starts first in a process, LightGBM training then crashes (segmentation fault). So each model's training tests live in that model's own test file, which pytest runs in the safe order (test_lgbm before test_nn); test_compare.py trains nothing. The suite now takes ~2 min.

## Phase 5: cost, threshold, calibration, and the one test run

**Predictions (before running):**
- (a) Better calibrated raw: LightGBM. Reason given: "in the past phases lightgbm worst score was better than nn's best".
- (b) Test ROC-AUC falls 0.01-0.03 below validation.
- (c) The headline catches 40-60% of test fraud dollars.

**Explain-back:** "the models percentages were already fairly honest so calibration didnt change the score much. the chance x amount caught about 81% of fraud dollars" (To fix next time: calibration never changes the ranking score (ROC-AUC); it changes the percentages, and because they were already close, the savings barely moved ($283,448 vs $283,508). Missing: why chance x amount beat the plain score cut-off. The cut-off ignores the amount, so a $5 and a $2,000 payment look the same; the amount rule spends reviews where the money is ($283k vs $239k at C = $10).)

**Break-it exercise:** in a scratch run (not committed), `freeze()` was made to treat test rows as validation rows. The test "gap and test rows cannot move the choices" failed, so it would catch that leak. No prediction was asked this time.

**Verdict rules (fixed before any run):**
- (a) Raw (uncalibrated) test ECE, 5 seeds each. A model wins only if its worst seed is better than the other model's best seed; otherwise "about the same".
- (b) Drop = Phase 4 mean val ROC-AUC - mean test ROC-AUC (5 seeds). Right only if BOTH models' drops fall in the predicted band.
- (c) Headline dollar recall on test, against the predicted band.

**Headline (fixed before any run):** LightGBM + calibrated rule at C = $10, mean over 5 seeds. Prediction (c) refers to this only. Every other model / rule / C is reported alongside, never quoted as the result.

**Test-set rules (fixed before any run):**
- The test set is evaluated exactly once, in one run (`python -m src.business test`), with the frozen Phase 4 configs (LightGBM trial 13, network trial 8), retrained on seeds 1-5 on train only, early stopping on validation as before.
- Every calibrator, threshold and cut-off is chosen from validation only, saved, and fingerprinted (sha256) before any test score exists. The test run refuses to start, and refuses to apply them, if the fingerprint changed.
- A second test run is refused unless `--override "reason"` is passed; the code then appends a dated line to this file.
- **Crash policy:** an override is allowed only if the run crashed before producing any result. The override line must say so in its reason.
- If something looks wrong after seeing test results, we report it; we do not re-tune.
- A rehearsal (`test --rehearsal`) must pass first: the same pipeline on validation rows, writing no lock and no test metrics.

**Decisions (Phase 5):**
- Cost model: a missed fraud costs its TransactionAmt; every flag costs C (a caught fraud costs C, not its amount); a legit payment left alone costs 0. A caught fraud is assumed fully saved.
- C is an assumption, swept over $1, $2, $5, $10, $20, $50 (a quick check up to a long investigation).
- Three rules, all frozen on validation: calibrated (flag if calibrated p x amount > C; nothing tuned), raw (the same formula on the uncalibrated score, as a control), threshold (one score cut-off per C that minimised validation cost; ignores amount). Baselines: flag nothing, flag everything, top 1% (cut-off = 99th percentile of validation scores, applied unchanged to test).
- Platt vs isotonic, one rule for both models: fit both on the first half of validation days, Brier on the second half; isotonic only if lower by more than 0.0001 (Platt is simpler and cannot overfit). The winner is refitted on all of validation. One calibrator and one set of thresholds per seed, so every money number has a seed spread.
- Validation now did four jobs (settings, stopping point, calibrators, thresholds), so validation money numbers are flattering. Only the test numbers are quoted as results.
- ECE uses 10 equal-count groups. Equal-width groups would put almost every payment in the lowest group because fraud is rare.
- ROC-AUC and PR-AUC are computed on raw scores; calibration does not change the order (Platt) or barely does (isotonic).
- Paired bootstrap: 1,000 draws of whole test days (seed 0); both models scored on the same draws; AUC averaged over the 5 seeds inside each draw. The same draws give a range for savings at C = $10. Not covered: only ~30 days; a card can span days; a different month; label delay; a different search seed or feature set.
- Test amount statistics are computed only inside the test run, after freezing; `choose` reports train and validation amounts only.
- Fit check: every retrained seed must reproduce its Phase 4 validation ROC-AUC exactly, or `fit` stops. The rehearsal also checks that the saved model files reproduce the validation scores made right after training.
- Per-row scores, model files and calibrators (isotonic steps are raw score values) stay in git-ignored `models/phase5/`; `metrics/` holds aggregates only.
- pytest-timeout: every test fails after 10 minutes instead of hanging (e.g. the LightGBM / PyTorch thread clash).
- Rehearsal finding: the network's scoring step crashed (segmentation fault) because it loaded LightGBM's library (via `src.lgbm`) before PyTorch. Reproduced in isolation: `import lightgbm; import torch` then 4-thread maths exits 139; the reverse order runs. Third form of the same OpenMP clash. Fix: that step imports PyTorch first. The rehearsal caught it before the one test run was spent.
- Calibrator choice (validation, held-out second half, Brier): LightGBM isotonic 0.02367 vs Platt 0.02379, a margin of 0.00012, just past the 0.0001 tie rule, so isotonic. Network: Platt 0.02570 vs isotonic 0.02570, tie, so Platt.
- Fit check passed: all 10 retrained models reproduced their Phase 4 validation ROC-AUC exactly. Rehearsal: reloaded model files reproduced their validation scores exactly (max difference 0).

**Result (test set, one run 2026-10-09, run 47c03d89; 5 seeds each; C is an assumption):**

Headline (fixed in advance): LightGBM + calibrated rule at C = $10 saves **$283,448** of $451,813 test fraud dollars (62.7%, net of review cost), seed spread +- $2,750, day-bootstrap 95% range $241k-$329k. It catches **81.2% of fraud dollars** and 58.7% of fraud payments, with 278 flags per day at precision 0.208 (about 1 in 5 flags is fraud).

| test, 5 seeds | LightGBM | neural net |
|---|---|---|
| ROC-AUC mean +- std | 0.9053 +- 0.0023 | 0.8754 +- 0.0034 |
| drop from Phase 4 validation | 0.0207 | 0.0120 |
| PR-AUC mean +- std | 0.5362 +- 0.0041 | 0.4377 +- 0.0145 |
| raw ECE per seed (min-max) | 0.0067-0.0080 | 0.0037-0.0078 |
| ECE raw -> calibrated (mean) | 0.0073 -> 0.0047 | 0.0057 -> 0.0046 |
| Brier raw -> calibrated (mean) | 0.02258 -> 0.02260 | 0.02556 -> 0.02547 |
| savings at C = $10: calibrated / raw / threshold / top 1% | $283k / $284k / $239k / $52k | $269k / $268k / $211k / $53k |
| dollar recall at C = $10, calibrated rule | 0.812 | 0.788 |

Paired day bootstrap (1,000 draws of 30 days): ROC-AUC gap LightGBM - network 0.0245 to 0.0350; PR-AUC gap 0.077 to 0.118; no draw had the network ahead. Test fraud rate 3.51% (validation 3.66%, train 3.42%); test median amount $68.50, mean $137.29.

**Verdicts (rules fixed above):**
- (a) Predicted LightGBM. Result: **about the same** (seed ranges overlap; the network's mean raw ECE is in fact lower). Prediction wrong. Note: the reason given was about ranking (Phase 4 ROC-AUC), and ranking well does not imply honest percentages; that is this phase's main lesson.
- (b) Predicted a drop of 0.01-0.03. Result: LightGBM 0.0207, network 0.0120; both in the band. **Prediction right.**
- (c) Predicted 40-60% of fraud dollars. Result: **81.2%**. Prediction wrong (one band above "60-80%").

**What the results say (reported, not re-tuned):**
- Using the amount is what pays: both amount-aware rules beat the amount-blind threshold at every C (at $10, $283k vs $239k for LightGBM).
- Calibration barely changed the money. Raw scores were already close to honest (ECE under 0.01), so the calibrated and raw rules saved almost the same ($283,448 vs $283,508 at $10). Calibration only helped clearly at C = $50 (LightGBM $155k vs $142k).
- The calibrators, fitted on validation, transferred only partly. On test LightGBM's isotonic calibrator cut ECE (0.0073 -> 0.0047) but left Brier no better (0.02258 -> 0.02260). It now slightly over-predicts in the riskiest tenth (says 28%, real 25%). Validation had a higher fraud rate (3.66% vs 3.51%), which fits this.
- LightGBM stays ahead on ranking (bootstrap gap clearly above 0). In dollars at C = $10 it saves about $15k more than the network, but the two models' savings ranges overlap heavily. A paired savings difference was not computed, so the dollar gap is not established.
- Top 1% (25-30 flags per day) is very precise (0.74-0.90) but catches only about 13% of fraud dollars. At C <= $2, flagging everything saves more than top 1%. The money-optimal rules want about 280 reviews per day; whether a team can do that is outside the data.
- The PR-AUC gap widened on test (validation 0.061, test 0.098); the network's PR-AUC also varies more across seeds (std 0.0145).
