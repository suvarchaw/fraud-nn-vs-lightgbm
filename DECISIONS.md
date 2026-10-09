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
- Phase 1 cut points are chosen by fixed fractions of the time range, never by looking at where fraud is higher or lower in the held-out period. (Disclosure, audit A6: the EDA notebook printed the weekly fraud rate for every week, test weeks included, before the cut points were set.)

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

**Break-it exercise:** in a scratch copy I left isFraud in the features. The prediction was ~0.95 with the conclusion "suspect leakage". Result: validation ROC-AUC 1.0000, PR-AUC 1.0000, and early stopping ended after 1 tree: the model read the answer directly. The conclusion was right; the score was even higher than predicted. A near-perfect score on rare fraud means check for leakage first.

**Result (seed 42, validation):** ROC-AUC 0.9189, PR-AUC 0.5691 (no-skill 0.5000 / 0.0366); best_iteration 557, 431 features. The prediction (0.90-0.95, 0.5-0.7) was right on both. Top feature V258 holds 11.5% of gain, so none dominates. Training took ~55 s with 4 threads.

**Decisions (Phase 2):**
- Excluded isFraud (the answer), TransactionID (a row counter that proxies for time and memorizes rows) and TransactionDT (validation is later than all training times, so trees cannot extrapolate calendar position; time effects are Phase 6's job).
- Category columns go straight into LightGBM; no encoding, scaling or resampling. The category label lists come from the whole file (names only, no fitted statistics); LightGBM records the train categories itself.
- Early stopping on validation ROC-AUC (in fact ROC-AUC or log-loss, whichever stalled first; audit A2), patience 100, best round kept, max 2000 trees, learning rate 0.05. The validation score is optimistic, by an amount not measured here (audit A4), because validation chose the tree count; the test set stays honest.
- Stopping metric and rule used here (validation ROC-AUC, patience 100, best round kept) must be reused for the neural net in Phase 3 so the comparison is fair. They are named constants in `src/lgbm.py`.
- Baseline is untuned (defaults + learning rate and tree cap). Phase 4 gives both models equal tuning effort on validation, or states plainly that the baseline is untuned.
- Determinism: seed 42 plus deterministic=True, force_row_wise=True and a fixed 4 threads (the thread count must stay fixed for repeatable results). Checked by a test.
- "Trained only on train" test reads the row count from the model's own first tree (`trees_to_dataframe`), because a check against rows our own function reports can be fooled by a bug in that function.
- The test block is discarded inside `train()` and never scored, counted or looked at in this phase.

## Phase 3: neural network

**Prediction (before running):** "im not sure just guessing that it loses on roc-auc and on pr-auc" (no reason given).

**Explain-back:** "we trained a neural model on the same data and found out that lightgbm and the neural model cant train in the same program run. we also observed that peeking at future data doesnt change the score. in the one run we did the neural model lost to the lightgbm (neural model score for next month-0.879 light gbm- 0.919)" (To fix next time: peeking changed the score only a little (0.8788 -> 0.8791) but is still cheating, which is why tests, not scores, must catch it. Missing: averages come only from older months because the future is not available in real life, so using it flatters the score. The thread freeze is a side issue, not a main finding.)

**Break-it exercise:** in a scratch copy, the medians, means and standard deviations were computed from train + validation together. Prediction: "hardly". Result: validation ROC-AUC 0.8788 -> 0.8791, PR-AUC 0.5116 -> 0.5164, so the prediction was right. Tests 2 (statistics equal a train-only recount; first mismatch: TransactionAmt median 4.2478 vs 4.2525) and 3 (changing validation must not change any statistic) both failed and caught it. Lesson: this leak is invisible in the score, so only a test catches it.

**Result (seed 42, validation):** ROC-AUC 0.8788, PR-AUC 0.5116 (LightGBM 0.9189 / 0.5691; no-skill 0.5000 / 0.0366). Train-sample (50,000 rows) ROC-AUC 0.9093, a small gap of 0.03. Best epoch 4 of 9 run, 28 s training, 274,187 weights, 895 inputs (400 numbers + 341 missing flags + 154 embedding numbers). The prediction (loses on both) was right for this seed. One seed cannot say which model is better; Phase 4 measures the spread.

**Sanity checks:** overfit 1,000 train rows (dropout off, 300 epochs): train loss 0.0000, PASS. Shuffled labels: validation ROC-AUC 0.6321, which FAILED the pre-set rule 0.48-0.52. Investigation (scratch only): 20 untrained networks scored 0.29-0.68; 5 shuffled-label runs scored 0.26-0.70 on both sides of 0.5 (mean 0.45); shuffled labels overlapped true fraud 3.50% vs 3.42% by chance. Not a leak: the rule was wrong. A network that learned nothing still ranks payments by some random mix of the columns, and the columns relate to fraud, so its score lands far from 0.5 by accident. The +-0.005 "luck" estimate wrongly assumed scores unrelated to the columns. New rule (my choice): the shuffled-label score must not beat the best of 20 untrained networks by more than 0.05. 0.6321 <= 0.7254, PASS.

**Decisions (Phase 3):**
- Same rows (`split_by_time`), same columns (`feature_names` imported from `src/lgbm.py`, not copied), same metrics, no resampling, no class weights. Test set not touched.
- Every preprocessing statistic comes from train rows only: median, mean, standard deviation, which columns get a missing flag, and each category's label list. Validation only has them applied.
- Numbers: signed log on every number column (one rule, nothing fitted), blanks filled with the train median plus a 0/1 missing flag, then scaled with the train mean and standard deviation, then clipped to [-5, 5] so wild future values cannot swamp the sum.
- Category label lists are counted in train rows. The pandas category list from `load_raw()` covers the whole file (including validation-only labels), so it is not used.
- Labels seen fewer than 10 times in train share the "unknown" slot, so that slot is trained and new labels in the future get a learned meaning, not random numbers.
- Embedding size = min(16, (slots + 1) // 2), counting slots (labels + unknown + missing), not labels as first planned, so a column with only rare labels still gets size 1, not 0.
- ID-like columns stored as numbers (card1, addr1, ...) stay numbers, exactly as LightGBM saw them. This may disadvantage the network; changing it would be tuning (Phase 4).
- One configuration, fixed before any validation score: 256 -> 128, ReLU, dropout 0.3, Adam, learning rate 0.001, batch 1,024. Not tuned.
- Stopping rule reused from Phase 2: validation ROC-AUC, best kept, patience 5 epochs, cap 50. (LightGBM's version also watched log-loss, audit A2, so the rules match only approximately.) 100 trees and 5 epochs are not the same unit; what is shared is the rule.
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

Verdict (rule above): **gap stayed**. Mean gap LightGBM - network: ROC-AUC 0.0387 (untuned: 0.0401), PR-AUC 0.0609 (untuned: 0.0575). The ROC-AUC gap is about 11 times the larger seed spread, and the network's best seed is below LightGBM's worst. Tuning helped both by a similar amount (ROC-AUC +0.0071 LightGBM, +0.0086 network). The prediction (gap stays) was right. Total run time 41 min.

**Decisions (Phase 4):**
- Validation is used for BOTH tuning (choosing the settings) and early stopping (choosing the tree/epoch count). Both validation scores are therefore optimistic, for both models, by an amount not measured here (audit A4). The untouched test set gives the honest number in the final report.
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

Headline (fixed in advance): LightGBM + calibrated rule at C = $10 saves **$283,448** of $451,813 test fraud dollars (62.7%, net of review cost), seed spread +- $2,750, day-bootstrap 95% range $241k-$329k. It catches **81.2% of fraud dollars** and 58.7% of fraud payments, with 278 flags per day at precision 0.208 (about 1 in 5 flags is fraud). The 81.2% belongs to this operating point, not to the model (audit A7).

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
- Calibration barely changed the money. Raw scores were already fairly close (ECE under 0.01; against a 3.5% fraud rate that is still about a 20% relative error, audit A7), so the calibrated and raw rules saved almost the same ($283,448 vs $283,508 at $10). Calibration only helped clearly at C = $50 (LightGBM $155k vs $142k).
- The calibrators, fitted on validation, transferred only partly. On test LightGBM's isotonic calibrator cut ECE (0.0073 -> 0.0047) but left Brier no better (0.02258 -> 0.02260). It now slightly over-predicts in the riskiest tenth (says 28%, real 25%). Validation had a higher fraud rate (3.66% vs 3.51%), which fits this.
- LightGBM stays ahead on ranking (bootstrap gap clearly above 0). In dollars at C = $10 it saves about $15k more than the network, but the two models' savings ranges overlap heavily. A paired savings difference was not computed, so the dollar gap is not established.
- Top 1% (25-30 flags per day) is very precise (0.74-0.90) but catches only about 13% of fraud dollars. At C <= $2, flagging everything saves more than top 1%. The money-optimal rules want about 280 reviews per day; whether a team can do that is outside the data.
- The PR-AUC gap widened on test (validation 0.061, test 0.098); the network's PR-AUC also varies more across seeds (std 0.0145).

## Phase 6: drift (descriptive only)

**Status of this phase:** the models are frozen (Phase 4 configs, seeds 1-5, trained on train only, files in `models/phase5/`), and the test set was spent in Phase 5. This phase re-reads test rows and labels descriptively; the Phase 5 lock guards only Phase 5 scoring (audit A10). Nothing in this phase changes a model, a threshold or a config, and nothing here is used to choose between models. Every finding is descriptive, or a hypothesis to check on a future month.

**Predictions (before running, 2026-10-09, no reasons given):**
- (a) Weekly ROC-AUC over weeks 17-25: **stays flat**.
- (b) Faster degrader: **neural net**.
- (c) Adversarial AUC, train vs test: **0.6-0.8**.

**Scoring rules for the predictions (fixed before any run):**
- (a) Checked in this order: "falls steadily" = H1 holds; "bounces around" = the slope range includes 0 and max - min of the weekly AUC > 2 x the median bootstrap range width; "stays flat" = the slope range includes 0 and |slope| x 9 weeks < 0.01. Otherwise "unclear". A slope range above 0 is reported as "rises".
- (b) Scored by the H2 rule. (c) Scored by the mean fold AUC.

**Hypotheses (fixed before any run):**
- Weeks are counted from the first payment (week 0-25). "Post-gap weeks" = weeks 17-25, the 9 weeks with no train or first-gap rows. The first gap is excluded because cards active at the end of train carry on into it, which flatters the model (memory, not lack of drift) and would fake a decline.
- **H1:** weekly ROC-AUC (mean of 5 seeds) falls with distance from the end of training. Holds if the straight-line (OLS) slope over weeks 17-25 is < 0 and its day-bootstrap 95% range excludes 0.
- **H2:** LightGBM degrades faster than the net (Phase 5 drop: 0.021 vs 0.012). Uses the paired bootstrap of (LightGBM slope - net slope) on the same draws: entirely < 0 = "LightGBM faster", entirely > 0 = "net faster", otherwise "cannot tell".
- **"H1 holds" does not establish drift.** A falling slope fits both drift and validation optimism, because the validation weeks sit early in weeks 17-25. Only the block table (gap1 / val / gap2 / test) can separate the two.
- **Alternative explanation for the Phase 5 drop difference:** validation optimism. Validation chose the settings and the stopping point, so validation scores sit a little high. That is not drift.
- **What separates them:** gap2 (the 7 days between validation and test) was never used by anything and sits right after validation. Each block is scored as a whole, with day-bootstrap ranges.
  - Optimism signature: the drop happens at the val -> gap2 edge (val above gap2, gap2 about equal to test), and there is no downward trend inside the validation weeks (17-19).
  - Drift signature: a steady decline. The within-validation (weeks 17-19) and within-test (weeks 22-25) slopes are negative, and gap2 lies between val and test.
  - If LightGBM's (val - gap2) step is bigger than the net's while their within-block slopes are similar, optimism explains the H2 difference.
  - Expected: gap2 has ~20k rows over 7 days and a within-block slope has 3-4 points, so the likely answer is "inconclusive". It will be reported as such.
- **Label-delay rule:** a fall confined to the final one or two weeks is flagged as "possible label delay, not drift". It is flagged if (i) H1 holds on weeks 17-25 but the slope over weeks 17-23 has a range including 0, or (ii) weeks 24 and/or 25 are the only weeks whose bootstrap range lies wholly below the mean of weeks 17-23.

**Decisions (Phase 6):**
- Re-scoring: only train and gap rows are scored new. Validation and test scores are reused from Phase 5's saved files, so the test rows are not re-scored. A sha256 of every file in `models/phase5/` is recorded and checked, to prove no model or frozen choice changed.
- Exact reproducibility check: for each block, pool its rows, compute the AUC per seed, then average over the 5 seeds. It must match Phase 5 test (LightGBM 0.9053, net 0.8754) and Phase 4 validation (0.9260, 0.8874) to 4 decimals, or the run stops.
- Weekly ranges: 1,000 draws of the week's 7 days with replacement (seed = week number), AUC averaged over the 5 seeds inside each draw (as in Phase 5). With only 7 days the range is slightly too narrow, and neighbouring weeks share cards and fraud rings, so the weeks are not independent.
- PR-AUC moves with the weekly fraud rate (2.1%-5.1%) even for an unchanged model, so H1 and H2 use ROC-AUC only. PR-AUC is shown next to the fraud rate.
- PSI bins:
  - Numeric columns: 10 bins from train deciles (repeated edges merged), plus missing as its own bin.
  - Category columns: one bin per label over every label seen in either block (labels unseen in train get train share 0), plus missing.
  - Shares are floored at 0.0001 before the log, so a zero share does not give infinity.
- PSI cut-offs 0.1 and 0.25 are industry conventions, not laws.
- PSI noise floor: early-train vs late-train (train days split at the middle day) is reported next to val-vs-train and test-vs-train.
- PSI "explained by cardinality alone": with no real change, PSI x n1n2/(n1+n2) is roughly chi-square with (bins - 1) degrees of freedom. A column whose PSI is below the 99th percentile of that is listed as explained by its number of labels alone. The approximation is rough when many labels have fewer than 5 rows.
- Adversarial validation:
  - It is a separate classifier: target = "which period is this row from", with the fraud model's features (no isFraud, no TransactionID, no TransactionDT). It never reads `models/phase5/` or the fraud label and saves no model.
  - LightGBM library defaults (100 trees), not tuned. Same pairs as PSI: early vs late train (noise floor), train vs val, train vs test.
  - Held-out AUC from 5 folds. Each fold = one contiguous fifth of the days of each period, so no random row split and no day in two folds.
- Retraining exercise (exploratory, LightGBM only):
  - Frozen Phase 4 config. Each seed uses its own Phase 5 tree count (339-431), with no early stopping, so nothing is tuned and no block is used to stop.
  - Origins: day 90, 120, 150, each with a 7-day gap before it. Windows: expanding (all earlier days) vs trailing (last 45 days, fixed now). Each following 30-day block is scored.
  - It reuses months already seen in earlier phases. The config and tree counts were tuned for a 109-day window, so the 45-day window is handicapped. Nothing is re-tuned to fix that.
- Link between drift and the fraud model: LightGBM gain importance (mean of the 5 frozen models) vs PSI and adversarial importance. Rank correlation and top-20 overlap only. Any link is a hypothesis, not proof. The net has no gain importance.
- Retrain rebuild check: seed 1 retrained with the retrain recipe on Phase 5's train rows reproduced the frozen model's validation scores exactly (max difference 0). So "frozen config, own tree count, no early stopping" is the same model recipe.
- Early vs late train was planned as a PSI / adversarial noise floor. It turned out to contain real change: the first ~8 weeks have a different product mix (products H and R about 11% each early, 2-4% later) and different missing-data rates (e.g. V1 missing 66% early, 45% late). It is reported as "within-train drift", not as noise. The chance level (chi-square) is the pure-luck reference instead.

**Result (2026-10-09; descriptive only; models unchanged, sha256 checked by `report`):**

Reproducibility: pooled block ROC-AUC (per seed, then mean) = val 0.9260 / 0.8874, test 0.9053 / 0.8754 (LightGBM / net), equal to Phases 4-5 to 4 decimals.

Block table (ROC-AUC, mean of 5 seeds, day-bootstrap 95% range):
| block | LightGBM | neural net |
|---|---|---|
| train (in-sample) | 0.9877 | 0.9149 |
| gap1 (7 days right after train) | 0.9392 (0.931-0.949) | 0.9056 (0.896-0.917) |
| val (weeks 16-20, used for tuning) | 0.9260 (0.917-0.934) | 0.8874 (0.880-0.895) |
| gap2 (7 days, never used) | 0.8947 (0.871-0.912) | 0.8431 (0.826-0.857) |
| test (spent in Phase 5) | 0.9053 (0.899-0.913) | 0.8754 (0.869-0.883) |

Weekly (weeks 17-25, ROC-AUC): LightGBM 0.944, 0.926, 0.924, 0.893, 0.910, 0.893, 0.910, 0.898, 0.911; net 0.902, 0.884, 0.884, 0.858, 0.863, 0.859, 0.880, 0.876, 0.880. Each week has 565-1,069 frauds; a weekly range is about +-0.01 to 0.02. Chart: `reports/weekly_auc.png`.

| pre-registered test | LightGBM | neural net |
|---|---|---|
| H1 slope per week, weeks 17-25 (95% range) | -0.0041 (-0.0058 to -0.0024) | -0.0020 (-0.0038 to -0.0002) |
| H1 | holds | holds |
| slope without weeks 24-25 | -0.0065 (-0.0087 to -0.0043) | -0.0049 (-0.0070 to -0.0028) |
| label-delay flag | no | no |
| within-val slope (weeks 17-19) | -0.020 to -0.002 | -0.017 to -0.001 |
| within-test slope (weeks 22-25) | -0.002 to +0.010 | +0.000 to +0.011 |
| step val - gap2 | +0.031 (0.010 to 0.056) | +0.044 (0.028 to 0.063) |
| step gap2 - test | -0.011 (-0.035 to 0.009) | -0.032 (-0.051 to -0.017) |
| optimism vs drift signature | inconclusive | inconclusive |

H2: slope difference LightGBM - net = -0.0021 (95% -0.0033 to -0.0009): **LightGBM faster**. The (val - gap2) step difference LightGBM - net is -0.033 to +0.007, so there is no sign that a bigger validation-optimism step explains LightGBM's faster fall.

PSI (`reports/psi_top20.png`; 0.1 and 0.25 are conventions, not laws):
| pair | rows | PSI > 0.1 | PSI > 0.25 | above 0.1 but within chance level |
|---|---|---|---|---|
| early vs late train (within-train drift) | 209,660 vs 171,155 | 298 | 18 | 0 |
| train vs val | 380,815 vs 84,093 | 35 | 9 | 0 |
| train vs test | 380,815 vs 84,233 | 25 | 20 | 0 |
Top train vs test: id_31 (browser version) 1.18, id_13 0.52, M7-M9 0.38, D11 0.36, V1-V11 0.33 (one shared missing-data pattern), M1-M3 0.31. Most of these shifts are changes in how often the column is missing.

Adversarial AUC (5 day-block folds; LightGBM defaults; no fraud label): early vs late train 0.833 (folds 0.66-0.91), train vs val 0.877 (0.66-0.96), **train vs test 0.893 (0.71-0.95)**. Top train-vs-test columns: id_31, id_13, D15, dist1, D11.

Link to the fraud model (LightGBM gain; hypothesis, not proof):
- Spearman rank correlation of gain vs train-vs-test PSI: +0.10. The top-20 gain and top-20 PSI lists share **no** column.
- Gain vs adversarial gain: +0.57, but both are close to 0 for most columns, which inflates a rank correlation. The top-20 lists share 2 columns (C13, D15).
- The columns the model leans on most (V258, C14, card2, C13, C1, card1) all have train-vs-test PSI under 0.05.

Retraining exercise (exploratory, LightGBM, ROC-AUC mean +- std over 5 seeds):
| window | origin day | next 30 days | +30 | +60 |
|---|---|---|---|---|
| expanding | 90 | 0.9074 +- 0.0005 | 0.8964 +- 0.0015 | 0.8911 +- 0.0010 |
| expanding | 120 | 0.9241 +- 0.0007 | 0.9091 +- 0.0012 | |
| expanding | 150 | 0.9248 +- 0.0008 | | |
| trailing 45 days | 90 | 0.8968 +- 0.0015 | 0.8805 +- 0.0028 | 0.8782 +- 0.0022 |
| trailing 45 days | 120 | 0.9164 +- 0.0006 | 0.8997 +- 0.0024 | |
| trailing 45 days | 150 | 0.9173 +- 0.0011 | | |

*Caption: Exploratory. Tree counts and config were tuned for a 109-day window, so the 45-day trailing window is handicapped (a bias against recency); nothing was re-tuned to fix it. Every month here was seen in earlier phases.*

How to read the retraining table:
- **Staleness.** Read the same target month (days 150-180) down the diagonal. Models whose data ended before day 143, 113 and 83 (last day 142, 112, 82) score 0.925 / 0.909 / 0.891 (expanding) and 0.917 / 0.900 / 0.878 (trailing, same 45-day size each time). Fresher models scored higher on the same month, but age is not isolated: model age, amount of data and which days are in the window change together (audit A5).
- **Recency vs amount of data.** Expanding beat trailing in every cell (by 0.007-0.016). Keeping older data helped, under the handicap stated in the caption.

**Verdicts on the predictions (rules fixed above):**
- (a) Predicted "stays flat". Result: **falls steadily** for both models (H1 slope range below 0). **Wrong.** The fall is front-loaded: steepest in weeks 16-20, roughly flat across the test weeks (22-25).
- (b) Predicted "net degrades faster". Result: **LightGBM faster** (slope difference range below 0). **Wrong.**
- (c) Predicted adversarial AUC 0.6-0.8. Result: **0.893** (over 0.8). **Wrong.**

**What the results say (descriptive; hypotheses for a future month, not conclusions):**
- Both frozen models lose ranking quality after training ends (for the net, the post-gap fall rests on week 17, audit A3). LightGBM falls about twice as fast per week, but stays ahead of the net in every out-of-sample week.
- **H1 holding does not establish drift.** The separator is inconclusive by the pre-set rule, for two reasons:
  - Against pure optimism: there is a downward trend inside the validation weeks.
  - Against a smooth drift line: gap2 (never used) sits below both validation and test, a dip rather than a point between them. Weeks 20-21 were hard for both models.
- Not label delay: the last two weeks are not where the fall is. Without them the slope is steeper.
- Hypothesis: "memory fade". LightGBM scores 0.988 in-sample and 0.939 in the week right after training, then settles near 0.90. Its top columns include card and address identifiers (card1, card2, addr1). A model that partly remembers recently active cards would lose that edge as cards turn over, faster for the model that memorises more. This fits the front-loaded fall, the faster LightGBM slope, and fresher retrains doing better. It is not tested here. (A post-hoc check on validation, audit A1, fits it.)
- Column drift is large and easy to detect, but it is mostly in columns the fraud model barely uses (browser version, M flags, missing-data patterns). The months look different (adversarial AUC 0.89), yet the train period also looks different from itself (0.83). "Looks different" is this data's normal state and does not by itself predict the score loss. PSI and the adversarial check only see changes in the columns themselves, not in how the columns relate to fraud, and the latter is what lowers ROC-AUC (audit A13).
- The net also dips inside train (weeks 12-13, about 0.88 in-sample) while LightGBM does not. Some fraud patterns there are hard for the net even on rows it trained on.
- PR-AUC fell more than ROC-AUC (LightGBM about 0.9 in-sample, about 0.5 on test). It also moves with the weekly fraud rate (2.1%-5.1%), so it is not used for the trend tests.

## Independent audit (2026-10-09, after Phase 6)

A read-only review that tried to break the claims above. No model, threshold, test or test-run file was changed, and nothing was re-scored on the test block. Labels A1-A13 follow the audit's numbering; the inline notes above point here.

**A1. Card memory explains most of LightGBM's lead (post-hoc, descriptive, validation only).**
- Client key: card1 + addr1 + (day - D1), the usual IEEE-CIS client id. It is approximate (different clients can share a key).
- 53.7% of validation rows (45,130; 1,769 frauds) come from clients already present in train ("returning"); 46.3% (38,963; 1,306 frauds) are new. 57% of validation fraud rows belong to clients that already had a fraud label in train: their fraud rate is 15.7%, against 0.03% for returning clients with a clean train history and 3.35% for new clients.
- Validation ROC-AUC (frozen Phase 5 models, mean of 5 seeds), with a paired day-block bootstrap (1,000 draws of whole validation days, seed 0, AUC averaged over the 5 seeds inside each draw, as in Phase 5):

| clients | LightGBM | neural net | gap (95% range) |
|---|---|---|---|
| all | 0.9260 | 0.8874 | 0.039 |
| returning | 0.9575 | 0.9007 | 0.057 (0.048 to 0.066) |
| new | 0.8751 | 0.8695 | 0.006 (-0.001 to 0.012; 4.9% of draws at or below 0) |

- The returning gap minus the new gap is 0.040 to 0.063, so the difference is not luck of the days drawn.
- Why: the dataset's hosts described labels as spreading from a chargeback to later linked transactions, and card / address columns let a model recognise those clients again. LightGBM uses them more (card1, card2, addr1 are top-10 gain).
- What it changes: "LightGBM beats the network by about 0.04 ROC-AUC" holds overall, but on new clients the two are within about 0.006 on validation. Test ROC-AUC overstates what a new client would see.
- This edge assumes the train labels are known by the time the validation payments arrive, i.e. within the 7-day gap. Real chargebacks take weeks to months, so in practice the card-memory edge (and LightGBM's lead) would be smaller.
- Not checked on the test block (spent). Validation chose both models' settings and stopping points, which affects both equally but means these numbers are not held-out.
- Side check: a plain blocklist (flag every validation payment of a client with a train fraud) catches 34.7% of validation fraud dollars at precision 0.157, saving $65k at C = $10. So the money result is not just a blocklist.

**A2. LightGBM's early stopping also watched log-loss.** `eval_metric="auc"` adds ROC-AUC to LightGBM's default log-loss, and `lgb.early_stopping` without `first_metric_only=True` stops when either metric has not improved for 100 rounds, keeping that metric's best round. Checked by re-running the Phase 2 fit (train and validation only): both metrics were watched, `best_iteration_` 557 is the log-loss minimum, and the ROC-AUC peak was round 569 (0.91895 vs 0.91885 kept). This applies to every LightGBM fit in Phases 2, 4 and 5 and to the tree counts reused in Phase 6. The effect on scores is tiny, but "the same stopping rule as the network" is only approximately true. Not fixed: refitting would change frozen models after the test set was spent. The planned Phase 7 refit uses a fixed tree count with no early stopping, so the stopping metric does not apply there.

**A3. For the net, "falls after training" rests on week 17 (post-hoc point estimates).** Straight-line slope of weekly ROC-AUC (`metrics/phase6_weekly.json`, `np.polyfit`): weeks 17-25 LightGBM -0.0041, net -0.0020 (as pre-registered); weeks 18-25 LightGBM -0.0024, net -0.0001. Week 17 is the first validation week: closest to train (card memory, A1) and used for tuning and stopping. Within the test weeks both slopes are flat or rising. Safer wording: both models are about 0.02-0.04 ROC-AUC lower than in the weeks right after training, mostly early, then flat across the test month; not "a steady decay".

**A4. "Slightly optimistic" validation was not measured.** The only optimism actually measured is the seed winner's curse (0.0013 LightGBM, 0.0032 net). gap2, never used and right after validation, scored 0.031 (LightGBM) and 0.044 (net) below validation; test scored 0.021 and 0.012 below. gap2 has only 602 frauds, so its range is wide (step val - gap2: 0.010 to 0.056 and 0.028 to 0.063), and drift plays a part. The size of the optimism is unknown, not "slight".

**A5. Retraining table: causes are mixed, and the target month is the test block.** On the expanding diagonal the amount of training data also changes (143 / 113 / 83 days); on the trailing diagonal the window content changes (days 38-82 overlap the early-train period with a different product mix). So the table shows "fresher scored higher", not the cost of age alone. The target month, days 150-180, is the Phase 5 test block, re-scored here by 30 new models (`drift retrain` does not check the lock). Choosing a retraining policy for Phase 7 from this table would be a test-informed decision.

**A6. Aggregate test labels were visible before the split was fixed.** The Phase 0 notebook (2026-10-02) printed the weekly fraud rate for every week, test weeks included, three days before the cut points were set, and `python -m src.split` prints the test row count and fraud rate. Nothing suggests a choice depended on it (60% / 80% are conventional fractions), but the accurate claim is "the test block was never used for any fit or choice", not "never looked at".

**A7. Framing of the money numbers.** At C = $10 the raw rule saves the same as the calibrated one ($283,508 vs $283,448) with 186 flags per day and 75.1% of fraud dollars, against 278 flags and 81.2%. On test the calibrator over-predicts the riskiest tenth (28% vs 24.8%), which adds flags. So 81.2% describes one operating point. About 4 of 5 flags are legitimate payments, and a false flag is costed at C only (no customer-friction cost). ECE under 0.01 is about a 20% relative error against a 3.5% fraud rate. Safe summary: about $283k net savings on one 30-day held-out window, under an assumed $10 review cost, catching 81% of fraud dollars at about 280 reviews per day, 1 in 5 flags being fraud.

**A9. Phase 6 cannot be re-run from a fresh clone.** `drift score` needs the per-row test scores in `models/phase5/` (git-ignored), which only the one locked test run writes; re-creating them needs `--override`, which the crash policy forbids. The sha256 of every file in `models/phase5/` is recorded in the Phase 6 metrics files, so the files used are pinned, but a fresh clone can reproduce Phases 1-5 only.

**A10. The lock covers Phase 5 only.** `business.score` refuses test rows outside the locked run. Phase 6 (`weekly`, `psi`, `adversarial`, `retrain`) reads test rows and labels without that check. That is by design once the test set is spent, but the accurate claim is "test labels were used once to judge the frozen models; Phase 6 re-reads them descriptively".

**A13. "Drift mostly in unused columns" means low-gain columns.** The drifting columns are rarely used by LightGBM, not unused. PSI and adversarial validation only see changes in the columns themselves, not in how the columns relate to fraud, and only the latter lowers ROC-AUC directly.

## Phase 7: serving the frozen model (FastAPI + Docker)

**Predictions (before running, 2026-10-09):** (a) p95 single-request latency 10-50 ms; (b) image size 500 MB-1 GB.

**Explain-back:**

**Break-it exercise:** not asked for this phase.

**Decisions (Phase 7):**
- Serve exactly one model: Phase 5 LightGBM seed 1 plus its Phase 5 isotonic calibrator, byte-identical to the evaluated files (sha256 of the model and of the calibrator are in `spec.json` and checked at startup). Seed 1 is fixed by the rule "the first Phase 5 seed", not by scores.
- No refit for v1. The frozen model is the only one with test numbers; any refit is a new, unevaluated model. A refit on train+validation would have no honest score left, and a retraining policy chosen from the Phase 6 table would be test-informed (audit A5). No ensemble either: an ensemble was never evaluated. A2 does not apply (no refit, no stopping rule).
- Skew control: `src/export.py` writes the feature list and column order (`feature_names`), and the category levels (from the model file's own `pandas_categorical`, asserted equal to `load_raw()` levels). Nothing is retyped. One function builds the feature table for single and batch requests: float32 numbers, categories with the training levels.
- Unseen label -> missing, because that is what LightGBM did in training and evaluation (tested against offline with the label set missing and with an added unseen category).
- Request fields are created from `spec.json`, all optional except `TransactionAmt`. The amount in p x amount is that same field, so a request cannot carry two amounts that disagree.
- Strict types: numbers must be JSON numbers (no "12" strings, no booleans, no NaN/inf); labels are strings up to 100 characters.
- Batch limit 500 rows, body limit 8 MB, both 413; empty batch 422. A POST without Content-Length is refused (411) so the size limit cannot be bypassed by chunked bodies.
- C: environment variable `REVIEW_COST`, default $10, overridable per request. Stated as assumed in the API docs and `/model-info`. Equal to C is not flagged (strict `>`, as in Phase 5).
- Scoring uses `num_threads=1`: predictions are per row, so the thread count cannot change a score, and a single small request does not pay for thread start-up.
- The calibrator maths (`np.interp`) is copied into the service instead of importing `src.business`, which pulls in the training code. A test checks the copy equals `business.calibrate` on a grid.
- The service imports nothing from `src/` and no PyTorch (a test imports it in a fresh process and checks). Neither `service/` nor `src/export.py` names the test block (a test checks the text).
- Validation errors return field path and error type only; the log has request number, route, status, latency and score only (tested with distinctive made-up values).
- Image: `python:3.12-slim` + `libgomp1`, `requirements-service.txt` (exact pins; lightgbm, numpy, pandas, scipy as in `requirements.txt`), non-root user, model mounted at `/models` read-only, `.dockerignore` excludes everything except `service/` and the requirements file. Not pushed anywhere. `requirements.txt` gained fastapi, uvicorn, pydantic, httpx (tests).
- The test file is named `tests/test_lgbm_service.py` so pytest runs it after `test_lgbm` and before `test_nn` (the LightGBM / PyTorch order rule).
- Docker's command was not on PATH (Docker Desktop is installed but its `docker` lives in `/Applications/Docker.app/Contents/Resources/bin`); the smoke script adds it for its own run.

**Result (2026-10-09, this laptop):**
- Parity: 2,000 validation rows through `/score_batch` and 50 through `/score` equal the offline calibrated scores to under 1e-9, and equal the scores Phase 5 saved after training.
- Image size: **649 MB** (Docker's figure for the disk size).
- Smoke run (real): build, run with mounted model, `/health` ok, `/score` returned a calibrated probability, logs held no payload values, stopped.
- Latency, loopback on this Mac, one worker, 1,000 requests after a 50-request warm-up, two runs (ms, p50 / p95 / max):

| case | run 1 | run 2 |
|---|---|---|
| single, 6 fields | 19.4 / 62.2 / 590 | 15.3 / 21.5 / 81 |
| single, all 431 fields | 18.8 / 25.1 / 97 | 16.0 / 25.5 / 158 |
| batch of 10, 431 fields each | 22.2 / 52.5 / 293 | 19.9 / 35.3 / 158 |

  These are local-machine numbers, not production numbers. They moved a lot between runs (the 6-field p95 was 62 ms, then 22 ms), so only the order of magnitude is meaningful.

- Where the time goes (`scripts/latency.py`, same three steps run in-process, 1,000 repeats after warm-up, ms p50 / p95; a third run of the script, so single-request totals differ slightly from the table above):

| case | request validation | frame build | model call + calibration |
|---|---|---|---|
| single, 6 fields | 0.03 / 0.07 | 6.2 / 7.0 | 7.8 / 8.8 |
| single, 431 fields | 0.10 / 0.14 | 7.3 / 8.4 | 7.2 / 8.3 |
| batch of 10, 431 fields each | 0.63 / 0.75 | 7.7 / 8.4 | 7.2 / 8.2 |

  Validation is negligible. Frame build (making a 431-column table) and the model call (which includes LightGBM converting that table) cost about 7 ms each, almost regardless of rows, so latency is dominated by per-request fixed cost, not by trees or rows. The rest of the single-request time (about 1-2 ms) is HTTP and thread hand-off. A faster path would build a plain array instead of a DataFrame; not done, because the DataFrame path is the one the parity test proves and category handling goes through it.
- Float32 guard: `src/export.py` asserts every integer feature column has max absolute value below 2**24 (float32 holds such integers exactly). Only `card1` is an integer column (max 18,396). The check reads the whole file's column bound, not test rows' labels or scores.
- Unknown vs known-but-never-trained label: a label outside the model's category list is turned into missing; a label inside the list (built from the whole file) that has no training rows goes to LightGBM unchanged and each split's own category rule decides. The parity test covers validation rows plus one constructed unseen-label row, so the second case is not tested for every label.

**Verdicts on the predictions:** (a) 10-50 ms: p95 was 21-26 ms in five of six single-request cells and 62 ms in one, so **right on the whole, not on every run**. (b) 500 MB-1 GB: 649 MB, **right**.

**What could make the deployment story misleading:**
- The test numbers come from one 30-day window with an assumed C = $10.
- Card memory (audit A1): the edge needs fraud labels of earlier payments from the same card; the service has no label feed or card history, and real chargebacks arrive weeks late.
- The model is frozen (Phase 6): scores fell 0.02-0.04 ROC-AUC after training, and nothing here retrains or monitors.
- Parity shows the service equals the offline code on validation rows, not that production inputs look like these anonymised columns.
- Latency and image size are for this laptop and this base image.
- "Flag" is the cost rule only: no customer-friction cost, no review capacity (the measured operating point was about 280 flags per day).

## Phase 8: scheduled retraining with delayed labels (exploratory, descriptive)

**Status of this phase:** exploratory and descriptive. The test set was spent in Phase 5. This phase reuses months already seen: the evaluation window (days 98-181) contains late train, validation (which chose both configs and their stopping points) and the test block. Nothing here changes a model, threshold or config already reported, and nothing here chooses a model for the service. Every number is a description of these 12 weeks, not a held-out estimate.

**Question:** if a fraud team retrained on a schedule using only the labels it would really have, how many dollars is that worth, and which model copes better? This removes the "labels known within 7 days" assumption behind audit A1.

**Predictions (before running, 2026-10-09, no reasons given):**
- (a) Retraining every 4 weeks beats never retraining in dollars at D = 30: **neither** model.
- (b) A longer label delay (60 vs 7 days) makes retraining matter: **more**.
- (c) Gains more from retraining: **LightGBM**.

**Design (fixed before any run):**
- Deployment clock: first deployment day r0 = 98 (days counted from the first payment). Decision points r = 98, 105, ..., 175; each is followed by a 7-day evaluation block [r, r+7); the 12 blocks cover days 98-181 exactly. Options weighed: r0 = 70 gives 16 weeks but a D = 60 first model from days 0-9 only (~40k rows, all from the early product mix); r0 = 112 (just after the frozen models' train end, day 109) gives only 10 weeks; r0 = 98 gives 12 weeks (whole 4-, 2- and 1-week cycles) and a D = 60 first model from days 0-37 (~160k rows, ~4.3k frauds).
- Label delay D: a row at time s is trainable at decision day r only if s + D days <= R and s < R (R = start of day r). D in {7, 30, 60}; D = 0 is shown only as an unrealistic upper bound and never used in a verdict. Rows whose label has not arrived are dropped from training, features included, so a training row always has its label.
- Policies, expanding window: never = {98}; every 4 weeks = {98, 126, 154}; every 2 weeks = {98, 112, ..., 168}; weekly = {98, 105, ..., 175}. At each block the active model is the latest one in the schedule retrained on or before the block's start. Models are shared across policies and delays when their training rows are identical (detected by hashing the row mask).
- Both models, identical rules, fixed recipes (nothing tuned on evaluation data): LightGBM = Phase 4 trial 13 with each seed's Phase 4 tree count (350/339/368/431/352), no early stopping; net = Phase 4 trial 8 with each seed's Phase 4 best epoch (6/5/4/10/14), no early stopping; the net's preprocessing is refitted on each model's own training rows. Seeds 1-5.
- Recipe check before any grid fit: on Phase 5's train rows, seed 1 of each recipe must reproduce the frozen Phase 5 validation scores, or the run stops.
- Decision rule: raw probabilities with the Phase 5 cost rule (flag if p x amount > C), C = $10. A calibrator per retrain would need held-out labelled data that a delayed-label team does not have.

**Metrics and rules (fixed before any run):**
- Primary: total net savings at C = $10 (fraud dollars caught minus C x flags) summed over the 12 blocks, the Phase 5 cost model with the same caveats. Per seed, a policy uses that seed's models throughout; mean +- std over 5 seeds. The realistic headline cell is D = 30.
- Secondary: ROC-AUC and PR-AUC over the 84 days pooled (per seed, then mean); flags per day; savings split by returning vs new clients. Client key = card1 + addr1 + (day - D1), as in audit A1. A row is "returning" for delay D if its client has an earlier row whose label is available at the block's decision day; the rule is fixed per D, so policies and models are compared on the same rows. A missing key counts as new.
- Extra dollars per retrain = (savings(policy) - savings(never)) / number of retrains after r0. The cost of a retrain (compute, engineering, checking) is not priced.
- Ranges: paired day-block bootstrap, 1,000 draws of the 84 evaluation days, seed 0; every model, delay, policy and segment on the same draws; per-row contributions averaged over seeds before drawing (as in Phase 5).
- **"Retraining pays"** for (model, D, policy): mean savings(policy) - savings(never) > 0 and its 95% range excludes 0. Range wholly below 0 = "retraining costs money". Otherwise "cannot tell".
- Prediction (a) is scored per model at D = 30, every 4 weeks vs never, by the rule above.
- Prediction (b), per model: gain(D = 60) - gain(D = 7), gain = (every 4 weeks) - never. Range > 0 = "more", < 0 = "less", else "cannot tell".
- **"Network copes better"** = it gains more dollars from retraining: [net(4-weekly) - net(never)] - [LightGBM(4-weekly) - LightGBM(never)] at D = 30. Range > 0 = net, < 0 = LightGBM, else "cannot tell". This scores prediction (c). Coping better is not saving more: absolute dollars are reported separately.
- Sanity ordering (a check, not a verdict): for each policy and model, savings should not rise as D grows (0 >= 7 >= 30 >= 60). A pair where the longer-delay savings minus the shorter-delay savings has a range wholly above 0 stops the report for investigation before any result is written.

**Compute budget:** the full grid (4 policies x 4 delays x 2 models x 5 seeds, ~37 distinct training sets per model) was estimated at ~4.5 h. Reduced, keeping 5 seeds: the weekly policy is run only for D = 0 and 7 (where it costs one extra training set because those delays share a chain); D = 30 and 60 get never / every 4 weeks / every 2 weeks. 25 training sets per model, estimated ~3.2 h. No pre-registered verdict uses the weekly policy.

**What could make the conclusion misleading (stated before running):**
- Reused months: the evaluation window contains validation and the spent test block.
- Fresher and more data are mixed (audit A5): with an expanding window, a retrain adds newer rows and more rows. A longer delay also leaves the never-retrain model with less data (at D = 60, days 0-37, the period with the different product mix), so "retraining matters more at long delay" may partly mean "the never-model was starved".
- Recipes tuned for one window: tree and epoch counts were chosen for a 109-day window; on smaller sets the net gets fewer gradient steps per epoch. Nothing is re-tuned.
- Those counts came from early stopping on validation days (116-145), which lie inside the evaluation window.
- Savings use raw probabilities, so they mix ranking quality with calibration; ROC-AUC is the ranking-only check.
- Labels trickle in and some never arrive, and this dataset's labels spread from a chargeback to the client's later payments; one fixed D is a simplification.
- C = $10 is assumed; no review-capacity limit; day-bootstrap ranges ignore clients spanning days; 84 days is one quarter.
- Seed k is reused at every retrain, so the seed spread is not independent across models. The client key is approximate. Pooled AUC mixes scores from different models across blocks.
