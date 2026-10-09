"""Phase 6: drift. DESCRIPTIVE ONLY: the models are frozen (models/phase5/) and the test set is spent.
Nothing here changes a model, threshold or config, and nothing here chooses between models.

LightGBM and PyTorch clash in one process, so every step is its own process:

    python -m src.drift score lgbm|nn   score train + gap rows with the frozen models (val/test reuse Phase 5 scores)
    python -m src.drift weekly          weekly AUC + day-bootstrap ranges, H1/H2, block table   (no ML library)
    python -m src.drift psi             PSI for every feature: early/late train, train/val, train/test
    python -m src.drift adversarial     a SEPARATE "which period is this row from" classifier (never the fraud label)
    python -m src.drift retrain         exploratory: expanding vs trailing window, LightGBM only, nothing tuned
    python -m src.drift report          link drift to the fraud model's gain importance; checks no model changed
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.business import BOOT_SEED, N_BOOT, day_draws, day_of, rows_in
from src.compare import SEEDS, _read, _stats, _write, nn_knobs
from src.split import BOUNDS, DAY, DT_START, GAP_DAYS, split_by_time

ROOT = Path(__file__).resolve().parent.parent
P5 = ROOT / "models" / "phase5"    # frozen models: read, never written
OUT = ROOT / "models" / "phase6"   # git-ignored: per-row scores
REPORTS = ROOT / "reports"
MODELS = ("lgbm", "nn")
NAMES = {"lgbm": "LightGBM", "nn": "neural net"}
BLOCKS = ("train", "gap1", "val", "gap2", "test")
FLOOR = 1e-4          # PSI: shares floored before the log, so a zero share is not infinite
N_BINS = 10           # PSI: train deciles
CHANCE_Q = 0.99       # PSI below this chi-square percentile = explained by number of bins/labels alone
N_FOLDS = 5           # adversarial: contiguous fifths of each period's days
FLAT = 0.01           # prediction (a): "flat" if |slope| x weeks < this
ORIGINS = (90, 120, 150)  # retrain: day the scored block starts; training ends GAP_DAYS earlier
TRAIL_DAYS = 45
HORIZON_DAYS = 30
RETRAIN_CAPTION = ("Exploratory. Tree counts and config were tuned for a 109-day window, so the 45-day trailing window "
                   "is handicapped (a bias against recency); nothing was re-tuned to fix it. Every month here was seen "
                   "in earlier phases.")


# ---------- time: weeks and blocks ----------

def week_of(dt):
    return ((np.asarray(dt) - DT_START) // (7 * DAY)).astype(int)


def block_of(dt):
    """train / gap1 / val / gap2 / test for each row, from split.py's BOUNDS (no numbers copied)."""
    t, b = np.asarray(dt), BOUNDS
    return np.select([t < b["train_end"], t < b["val_start"], t < b["val_end"], t < b["test_start"]],
                     list(BLOCKS[:4]), BLOCKS[4])


def weeks_made_of(wk, blk, allowed):
    """Weeks whose rows all come from the `allowed` blocks."""
    bad = np.unique(wk[~np.isin(blk, allowed)])
    return [int(w) for w in np.unique(wk) if w not in bad]


def model_hashes():
    """sha256 of every frozen file: recorded by each step, checked by `report`."""
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(P5.iterdir()) if f.is_file()}


# ---------- small maths: no LightGBM, no PyTorch ----------

def seed_mean(f, y, P):
    return float(np.mean([f(y, p) for p in P]))


def slope(x, Y):
    """OLS slope of Y against x. Y may be (draws, len(x)): one slope per draw."""
    return np.polyfit(np.asarray(x, float), np.asarray(Y, float).T, 1)[0]


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def trend_label(s, rng, weekly, widths, n_weeks):
    """Prediction (a), rule fixed in DECISIONS.md: falls, rises, bounces, flat, else unclear."""
    if rng[1] < 0:
        return "falls steadily"
    if rng[0] > 0:
        return "rises"
    if np.ptp(weekly) > 2 * np.median(widths):
        return "bounces around"
    return "stays flat" if abs(s) * n_weeks < FLAT else "unclear"


def label_delay_flag(h1_holds, early_rng, week_hi, early_mean, last_weeks):
    """Fall confined to the final weeks: (i) H1 holds but not without them, or (ii) only they sit wholly below
    the earlier weeks' mean. week_hi: {week: upper end of its range}."""
    below = {w for w, hi in week_hi.items() if hi < early_mean}
    return bool((h1_holds and early_rng[0] <= 0 <= early_rng[1]) or (below and below <= set(last_weeks)))


def separator(pt, step_vg, step_gt, val_slope, test_slope):
    """Optimism vs drift signatures fixed in DECISIONS.md. Ranges are (lo, hi); pt = block point ROC-AUCs."""
    optimism = step_vg[0] > 0 and step_gt[0] <= 0 <= step_gt[1] and val_slope[1] >= 0
    drift = val_slope[1] < 0 and test_slope[1] < 0 and pt["val"] >= pt["gap2"] >= pt["test"]
    return "optimism" if optimism and not drift else "drift" if drift and not optimism else "inconclusive"


def shares(ref, cmp):
    """Bin shares for one column in two row sets. Numbers: train-decile bins + missing. Categories: one bin per
    label of the (whole-file) category list, so unseen labels count, + missing. Returns (ref, cmp, bins used)."""
    if isinstance(ref.dtype, pd.CategoricalDtype):
        n = len(ref.cat.categories) + 1
        cr = np.bincount(ref.cat.codes.to_numpy() + 1, minlength=n)  # code -1 (missing) -> bin 0
        cc = np.bincount(cmp.cat.codes.to_numpy() + 1, minlength=n)
    else:
        r, c = ref.to_numpy("float64"), cmp.to_numpy("float64")
        ok = ~np.isnan(r)
        edges = np.unique(np.quantile(r[ok], np.linspace(0, 1, N_BINS + 1)[1:-1])) if ok.any() else np.array([])
        n = len(edges) + 2
        bins = lambda v: np.where(np.isnan(v), n - 1, np.searchsorted(edges, v, side="right"))
        cr, cc = np.bincount(bins(r), minlength=n), np.bincount(bins(c), minlength=n)
    used = int(((cr > 0) | (cc > 0)).sum())
    return cr / cr.sum(), cc / cc.sum(), used


def psi(e, a):
    e, a = np.maximum(e, FLOOR), np.maximum(a, FLOOR)
    return float(np.sum((a - e) * np.log(a / e)))


def chance_psi(k, n1, n2):
    """99th percentile of PSI when nothing changed (chi-square approximation, k bins in use)."""
    from scipy.stats import chi2
    return float(chi2.ppf(CHANCE_Q, max(k - 1, 1)) * (1 / n1 + 1 / n2))


def pairs(df):
    """(reference rows, compared rows) for each comparison. Early/late: train days split at the middle day."""
    tr, val, test = split_by_time(df)
    d = day_of(tr["TransactionDT"])
    mid = (d.min() + d.max() + 1) // 2
    return {"early_vs_late_train": (tr[d < mid], tr[d >= mid]), "train_vs_val": (tr, val), "train_vs_test": (tr, test)}


def psi_table(df):
    """{pair: {feature: {psi, bins, chance}}}. Reads features only, never the label."""
    from src.lgbm import feature_names
    out = {}
    for name, (a, b) in pairs(df).items():
        out[name] = {}
        for c in feature_names(df):
            e, s, k = shares(a[c], b[c])
            out[name][c] = {"psi": psi(e, s), "bins": k, "chance": chance_psi(k, len(a), len(b))}
    return out


def adversarial_xy(df, a, b):
    """Features and target for 'is this row from period b'. Same columns as the fraud model; no label, no time."""
    from src.lgbm import feature_names
    rows = pd.concat([a, b])
    return rows[feature_names(df)], np.r_[np.zeros(len(a), int), np.ones(len(b), int)], day_of(rows["TransactionDT"])


def folds(days, k=N_FOLDS):
    """Fold per row: each period's distinct days cut into k contiguous runs. `days` of one period only."""
    u = np.unique(days)
    return (np.arange(len(u)) * k // len(u))[np.searchsorted(u, days)]


# ---------- steps ----------

def score(model):
    """Score train + gap rows with the 5 frozen models; validation and test scores are Phase 5's saved ones."""
    if model == "nn":
        import torch  # must load before LightGBM's library (src.lgbm below): the other order crashes PyTorch
    from src.data import load_raw
    from src.lgbm import feature_names
    before = model_hashes()
    df = load_raw(verbose=False)
    tr, val, test = split_by_time(df)
    rows = df.drop(val.index.union(test.index))  # train + both gaps
    feats = feature_names(df)
    gain = None
    if model == "lgbm":
        import lightgbm as lgb
        boosters = [lgb.Booster(model_file=str(P5 / f"lgbm_seed{s}.txt")) for s in SEEDS]
        assert all(b.feature_name() == feats for b in boosters)
        p = [b.predict(rows[feats]) for b in boosters]
        g = np.array([b.feature_importance("gain") for b in boosters])
        gain = (g / g.sum(axis=1, keepdims=True)).mean(axis=0)  # each model's gain shares, averaged over seeds
    else:
        from src import nn
        nn.setup(0)
        cfg = nn_knobs(_read("phase4_tune_nn.json")["best_config"])
        prep = nn.fit_prep(tr, feats)  # train-only statistics, rebuilt exactly as in training
        xn, xc = nn.transform(rows, prep)
        p = []
        for s in SEEDS:
            net = nn.Net(xn.shape[1], nn.slots(prep), cfg["dropout"], cfg["hidden"])
            net.load_state_dict(torch.load(P5 / f"nn_seed{s}.pt"))
            p.append(nn.predict(net, xn, xc))
    ids, P = [rows["TransactionID"].to_numpy()], [np.array(p)]
    for b in ("val", "test"):
        f = np.load(P5 / f"{model}_{b}_scored.npz")
        ids.append(f["ids"])
        P.append(f["p"])
    pos = pd.Index(df["TransactionID"]).get_indexer(np.concatenate(ids))
    assert (pos >= 0).all() and len(np.unique(pos)) == len(df), "every row must be scored exactly once"
    full = np.empty((len(SEEDS), len(df)))
    full[:, pos] = np.concatenate(P, axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    extra = {"gain": gain, "feats": np.array(feats)} if gain is not None else {}
    np.savez(OUT / f"{model}_all.npz", ids=df["TransactionID"].to_numpy(), p=full, **extra)
    assert model_hashes() == before, "a frozen file changed during scoring"
    print(f"{model}: scored {len(rows):,} train + gap rows new; {len(df) - len(rows):,} val/test rows from Phase 5")


def load_scores(df):
    S = {}
    for m in MODELS:
        f = np.load(OUT / f"{m}_all.npz")
        assert (f["ids"] == df["TransactionID"].to_numpy()).all()
        S[m] = f["p"]
    return S


def check_reproduces(y, blk, S):
    """Pooled block AUC (per seed, then averaged) must equal Phase 4 val / Phase 5 test to 4 decimals."""
    ref = {"val": {m: _read(f"phase4_seeds_{m}.json")["val"]["roc_auc"]["mean"] for m in MODELS},
           "test": {m: _read("phase5_test.json")["auc"][m]["roc_auc"]["mean"] for m in MODELS}}
    out = {}
    for b, r in ref.items():
        i = blk == b
        for m in MODELS:
            got = seed_mean(roc_auc_score, y[i], S[m][:, i])
            out[f"{b}/{m}"] = {"got": got, "phase4_5": r[m]}
            if round(got, 4) != round(r[m], 4):
                sys.exit(f"{b} {m}: pooled ROC-AUC {got:.6f} does not reproduce {r[m]:.6f}; refusing")
    return out


def weekly():
    from src.data import load_raw
    df = load_raw(verbose=False)
    S = load_scores(df)
    y, dt = df["isFraud"].to_numpy(), df["TransactionDT"].to_numpy()
    wk, blk, day = week_of(dt), block_of(dt), day_of(dt)
    assert np.bincount(wk).sum() == len(df)
    repro = check_reproduces(y, blk, S)
    weeks = np.unique(wk)
    post = weeks_made_of(wk, blk, ["val", "gap2", "test"])
    val_w, test_w = weeks_made_of(wk, blk, ["val"]), weeks_made_of(wk, blk, ["test"])
    roc_draws = {m: np.empty((N_BOOT, len(weeks))) for m in MODELS}
    rows = []
    for w in weeks:
        r = np.flatnonzero(wk == w)
        rows_of, draws = day_draws(day[r], seed=BOOT_SEED + int(w))
        pr_d = {m: [] for m in MODELS}
        for i, dr in enumerate(draws):
            idx = r[rows_in(rows_of, dr)]
            for m in MODELS:
                roc_draws[m][i, w] = seed_mean(roc_auc_score, y[idx], S[m][:, idx])
                pr_d[m].append(seed_mean(average_precision_score, y[idx], S[m][:, idx]))
        mix = pd.Series(blk[r]).value_counts()
        rows.append({"week": int(w), "rows": len(r), "frauds": int(y[r].sum()), "fraud_rate": float(y[r].mean()),
                     "blocks": {k: int(v) for k, v in mix.items()}, "in_sample": bool("train" in mix),
                     **{m: {"roc_auc": seed_mean(roc_auc_score, y[r], S[m][:, r]), "roc_auc_95": ci(roc_draws[m][:, w]),
                            "pr_auc": seed_mean(average_precision_score, y[r], S[m][:, r]), "pr_auc_95": ci(pr_d[m])}
                        for m in MODELS}})
        print(f"week {w:2d}  rows {len(r):6,}  frauds {int(y[r].sum()):5}  " + "  ".join(
            f"{m} {rows[-1][m]['roc_auc']:.4f}" for m in MODELS), flush=True)

    # Block table: each block scored as a whole (ROC-AUC only), draws of the block's own days.
    blocks, blk_draws = {}, {m: {} for m in MODELS}
    for j, b in enumerate(BLOCKS[1:]):  # train is in-sample: point value only below
        r = np.flatnonzero(blk == b)
        rows_of, draws = day_draws(day[r], seed=BOOT_SEED + 1000 + j)
        for m in MODELS:
            blk_draws[m][b] = np.array([seed_mean(roc_auc_score, y[r[rows_in(rows_of, dr)]],
                                                  S[m][:, r[rows_in(rows_of, dr)]]) for dr in draws])
        blocks[b] = {"rows": len(r), "days": len(rows_of), "frauds": int(y[r].sum()),
                     **{m: {"roc_auc": seed_mean(roc_auc_score, y[r], S[m][:, r]), "roc_auc_95": ci(blk_draws[m][b]),
                            "pr_auc": seed_mean(average_precision_score, y[r], S[m][:, r])} for m in MODELS}}
        print(f"block {b:5} done", flush=True)
    r = blk == "train"
    blocks["train"] = {"rows": int(r.sum()), "in_sample": True,
                       **{m: {"roc_auc": seed_mean(roc_auc_score, y[r], S[m][:, r])} for m in MODELS}}

    res = {"post_weeks": post, "val_weeks": val_w, "test_weeks": test_w, "reproduce": repro, "weeks": rows,
           "blocks": blocks, "boot": {"draws": N_BOOT, "seed": f"{BOOT_SEED} + week number", "unit": "whole days"},
           "model_hashes": model_hashes(), "models": {}}
    early = [w for w in post if w < post[-2]]
    slopes = {}
    for m in MODELS:
        pt = np.array([rows[w][m]["roc_auc"] for w in post])
        s, s_d = float(slope(post, pt)), slope(post, roc_draws[m][:, post])
        slopes[m] = s_d
        rng = ci(s_d)
        h1 = rng[1] < 0
        e_rng = ci(slope(early, roc_draws[m][:, early]))
        widths = [rows[w][m]["roc_auc_95"][1] - rows[w][m]["roc_auc_95"][0] for w in post]
        bp = {b: blocks[b][m]["roc_auc"] for b in BLOCKS[1:]}
        step_vg, step_gt = blk_draws[m]["val"] - blk_draws[m]["gap2"], blk_draws[m]["gap2"] - blk_draws[m]["test"]
        v_rng, t_rng = ci(slope(val_w, roc_draws[m][:, val_w])), ci(slope(test_w, roc_draws[m][:, test_w]))
        res["models"][m] = {
            "h1_slope_per_week": s, "h1_slope_95": rng, "h1_holds": h1,
            "slope_weeks_without_last_two": float(slope(early, [rows[w][m]["roc_auc"] for w in early])),
            "slope_weeks_without_last_two_95": e_rng,
            "label_delay_flag": label_delay_flag(h1, e_rng, {w: rows[w][m]["roc_auc_95"][1] for w in post},
                                                 float(np.mean([rows[w][m]["roc_auc"] for w in early])), post[-2:]),
            "prediction_a_class": trend_label(s, rng, pt, widths, len(post)),
            "weekly_spread": float(np.ptp(pt)), "median_range_width": float(np.median(widths)),
            "step_val_minus_gap2": bp["val"] - bp["gap2"], "step_val_minus_gap2_95": ci(step_vg),
            "step_gap2_minus_test": bp["gap2"] - bp["test"], "step_gap2_minus_test_95": ci(step_gt),
            "within_val_slope_95": v_rng, "within_test_slope_95": t_rng,
            "signature": separator(bp, ci(step_vg), ci(step_gt), v_rng, t_rng),
        }
    d = slopes["lgbm"] - slopes["nn"]
    h2 = ci(d)
    res["h2"] = {"slope_diff_lgbm_minus_nn": res["models"]["lgbm"]["h1_slope_per_week"] -
                 res["models"]["nn"]["h1_slope_per_week"], "slope_diff_95": h2,
                 "verdict": "LightGBM faster" if h2[1] < 0 else "net faster" if h2[0] > 0 else "cannot tell",
                 "step_diff_lgbm_minus_nn_95": ci((blk_draws["lgbm"]["val"] - blk_draws["lgbm"]["gap2"]) -
                                                  (blk_draws["nn"]["val"] - blk_draws["nn"]["gap2"]))}
    _write("phase6_weekly.json", res)
    weekly_chart(res)
    print_weekly(res)


def print_weekly(res):
    print("\nreproduce (pooled block, per seed then mean):",
          ", ".join(f"{k} {v['got']:.4f}" for k, v in res["reproduce"].items()))
    print(f"post-gap weeks {res['post_weeks']}; pure val {res['val_weeks']}; pure test {res['test_weeks']}")
    for m, r in res["models"].items():
        lo, hi = r["h1_slope_95"]
        print(f"{NAMES[m]:10} H1 slope {r['h1_slope_per_week']:+.5f}/week (95% {lo:+.5f} to {hi:+.5f})  "
              f"H1 {'holds' if r['h1_holds'] else 'does not hold'}  label-delay flag {r['label_delay_flag']}  "
              f"(a) class: {r['prediction_a_class']}  signature: {r['signature']}")
    print(f"H2 slope diff LightGBM - net {res['h2']['slope_diff_lgbm_minus_nn']:+.5f} "
          f"(95% {res['h2']['slope_diff_95'][0]:+.5f} to {res['h2']['slope_diff_95'][1]:+.5f}): {res['h2']['verdict']}")
    print("block ROC-AUC (whole block):", "  ".join(
        f"{b} " + "/".join(f"{res['blocks'][b][m]['roc_auc']:.4f}" for m in MODELS) for b in BLOCKS))


def psi_step():
    from src.data import load_raw
    df = load_raw(verbose=False)
    t = psi_table(df)
    n = {k: [len(a), len(b)] for k, (a, b) in pairs(df).items()}
    summary = {}
    for name, cols in t.items():
        v = pd.Series({c: x["psi"] for c, x in cols.items()})
        summary[name] = {"rows": n[name], "over_0.1": int((v > 0.1).sum()), "over_0.25": int((v > 0.25).sum()),
                         "top20": {c: round(float(x), 4) for c, x in v.nlargest(20).items()},
                         "explained_by_cardinality_alone": sorted(c for c, x in cols.items() if x["psi"] < x["chance"]
                                                                  and x["psi"] > 0.1)}
    res = {"cutoffs_note": "0.1 and 0.25 are industry conventions, not laws.",
           "cardinality_note": "explained_by_cardinality_alone lists columns with PSI > 0.1 that are still below the "
                               "99th-percentile chance level for their number of bins/labels.", "floor": FLOOR, "bins": N_BINS,
           "summary": summary, "per_feature": t, "model_hashes": model_hashes()}
    _write("phase6_psi.json", res)
    psi_chart(t)
    for name, s in summary.items():
        print(f"{name:20} rows {s['rows'][0]:,} vs {s['rows'][1]:,}  PSI > 0.1: {s['over_0.1']:3}  > 0.25: "
              f"{s['over_0.25']:3}  over 0.1 but below chance level: {len(s['explained_by_cardinality_alone'])}")
    print("top 20 train vs test:", ", ".join(f"{c} {v}" for c, v in summary["train_vs_test"]["top20"].items()))


def adversarial():
    """Separate from the fraud models: never reads models/phase5/ or isFraud, saves no model."""
    import lightgbm as lgb
    from src.data import load_raw
    df = load_raw(verbose=False)
    res = {"model": "LightGBM library defaults (100 trees), seed 0, not tuned", "folds": N_FOLDS, "pairs": {}}
    for name, (a, b) in pairs(df).items():
        X, t, d = adversarial_xy(df, a, b)
        assert "isFraud" not in X and "TransactionDT" not in X and "TransactionID" not in X
        fold = np.r_[folds(d[t == 0]), folds(d[t == 1])]
        aucs, gains = [], []
        for k in range(N_FOLDS):
            m = lgb.LGBMClassifier(random_state=0, deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)
            m.fit(X[fold != k], t[fold != k])
            aucs.append(float(roc_auc_score(t[fold == k], m.predict_proba(X[fold == k])[:, 1])))
            g = m.booster_.feature_importance("gain")
            gains.append(g / g.sum())
        share = pd.Series(np.mean(gains, axis=0), index=X.columns)
        res["pairs"][name] = {"rows": [len(a), len(b)], "auc": _stats(aucs), "auc_per_fold": aucs,
                              "top20_gain_share": {c: round(float(v), 4) for c, v in share.nlargest(20).items()},
                              "gain_share": {c: float(v) for c, v in share.items()}}
        print(f"{name:20} adversarial AUC {np.mean(aucs):.4f} (folds {min(aucs):.4f}-{max(aucs):.4f})  top: "
              + ", ".join(f"{c} {v:.3f}" for c, v in share.nlargest(5).items()), flush=True)
    res["model_hashes"] = model_hashes()
    _write("phase6_adversarial.json", res)


def retrain():
    """Exploratory: frozen Phase 4 config, each seed's own Phase 5 tree count, no early stopping, nothing tuned."""
    import lightgbm as lgb
    from src.data import load_raw
    from src.lgbm import LEARNING_RATE, feature_names
    cfg = _read("phase4_tune_lgbm.json")["best_config"]
    trees = {s: lgb.Booster(model_file=str(P5 / f"lgbm_seed{s}.txt")).num_trees() for s in SEEDS}
    df = load_raw(verbose=False)
    feats, y, d = feature_names(df), df["isFraud"].to_numpy(), day_of(df["TransactionDT"])

    def fit(rows, s):
        m = lgb.LGBMClassifier(n_estimators=trees[s], learning_rate=LEARNING_RATE, random_state=s, deterministic=True,
                               force_row_wise=True, n_jobs=4, verbose=-1)
        m.set_params(**cfg, subsample_freq=1)
        return m.fit(df.loc[rows, feats], y[rows])

    # Check: on Phase 5's own train rows, seed 1 must rebuild the frozen model's validation scores exactly.
    tr, val, _ = split_by_time(df)
    s1 = fit(df.index.isin(tr.index), SEEDS[0]).predict_proba(val[feats])[:, 1]
    diff = float(np.abs(s1 - np.load(P5 / "lgbm_val.npz")["p"][0]).max())
    print(f"rebuild check: seed {SEEDS[0]} on Phase 5 train rows vs frozen val scores, max difference {diff:.1e}")
    assert diff < 1e-9, "the retrain recipe does not rebuild the frozen model"

    cells = []
    for o in ORIGINS:
        end = o - GAP_DAYS
        targets = [(h, o + HORIZON_DAYS * (h - 1)) for h in (1, 2, 3) if o + HORIZON_DAYS * h <= d.max() + 1]
        for window, start in (("expanding", 0), ("trailing", end - TRAIL_DAYS)):
            rows = (d >= start) & (d < end)
            auc = {h: [] for h, _ in targets}
            for s in SEEDS:
                m = fit(rows, s)
                for h, a in targets:
                    t = (d >= a) & (d < a + HORIZON_DAYS)
                    auc[h].append(float(roc_auc_score(y[t], m.predict_proba(df.loc[t, feats])[:, 1])))
            for h, a in targets:
                cells.append({"origin": o, "window": window, "train_days": [int(start), int(end)],
                              "train_rows": int(rows.sum()), "horizon": h, "target_days": [a, a + HORIZON_DAYS],
                              "model_age_days": a - end, "roc_auc": _stats(auc[h]), "per_seed": auc[h]})
                print(f"origin {o} {window:9} days {start}-{end} -> {a}-{a + HORIZON_DAYS}  ROC-AUC "
                      f"{np.mean(auc[h]):.4f} +- {np.std(auc[h], ddof=1):.4f}", flush=True)
    _write("phase6_retrain.json", {"caption": RETRAIN_CAPTION, "config": cfg, "trees": trees, "gap_days": GAP_DAYS,
                                   "trail_days": TRAIL_DAYS, "rebuild_check_max_diff": diff, "cells": cells,
                                   "model_hashes": model_hashes()})
    print_retrain(cells)


def print_retrain(cells):
    print(f"\n{RETRAIN_CAPTION}\nROC-AUC mean +- std over 5 seeds")
    print(f"{'window':10}{'origin':>7}" + "".join(f"{'horizon ' + str(h):>18}" for h in (1, 2, 3)))
    for w in ("expanding", "trailing"):
        for o in ORIGINS:
            c = {x["horizon"]: x["roc_auc"] for x in cells if x["window"] == w and x["origin"] == o}
            print(f"{w:10}{o:>7}" + "".join(f"{c[h]['mean']:>11.4f} +-{c[h]['std']:.3f}" if h in c else f"{'':>18}"
                                            for h in (1, 2, 3)))


def report():
    now = model_hashes()
    for f in ("weekly", "psi", "adversarial", "retrain"):
        if _read(f"phase6_{f}.json")["model_hashes"] != now:
            sys.exit(f"a frozen model file changed since `{f}`; refusing")
    w, p, a = _read("phase6_weekly.json"), _read("phase6_psi.json"), _read("phase6_adversarial.json")
    g = np.load(OUT / "lgbm_all.npz")
    gain = pd.Series(g["gain"], index=g["feats"])
    drift = pd.Series({c: x["psi"] for c, x in p["per_feature"]["train_vs_test"].items()})[gain.index]
    adv = pd.Series(a["pairs"]["train_vs_test"]["gain_share"])[gain.index]
    top = lambda s: set(s.nlargest(20).index)
    card = set(p["summary"]["train_vs_test"]["explained_by_cardinality_alone"])
    res = {
        "note": "LightGBM only (the net has no gain importance). Any link is a hypothesis, not proof.",
        "spearman_gain_vs_psi_test": float(gain.corr(drift, method="spearman")),
        "spearman_gain_vs_adversarial_gain": float(gain.corr(adv, method="spearman")),
        "top20_overlap": {"gain_and_psi": sorted(top(gain) & top(drift)), "gain_and_adversarial": sorted(top(gain) & top(adv)),
                          "psi_and_adversarial": sorted(top(drift) & top(adv))},
        "top20_gain_table": [{"feature": c, "gain_share": round(float(gain[c]), 4),
                              "psi_val": round(p["per_feature"]["train_vs_val"][c]["psi"], 4),
                              "psi_test": round(float(drift[c]), 4),
                              "psi_early_late_train": round(p["per_feature"]["early_vs_late_train"][c]["psi"], 4),
                              "adversarial_gain_share": round(float(adv[c]), 4), "cardinality_alone": c in card}
                             for c in gain.nlargest(20).index],
        "predictions": {"a": {m: w["models"][m]["prediction_a_class"] for m in MODELS}, "a_predicted": "stays flat",
                        "b": w["h2"]["verdict"], "b_predicted": "net faster",
                        "c": a["pairs"]["train_vs_test"]["auc"]["mean"], "c_predicted": "0.6-0.8"},
        "model_hashes_unchanged": True,
    }
    _write("phase6_report.json", res)
    print_weekly(w)
    print(f"\nadversarial AUC: " + "  ".join(f"{k} {v['auc']['mean']:.4f}" for k, v in a["pairs"].items()))
    for k, s in p["summary"].items():
        print(f"PSI {k:20} > 0.1: {s['over_0.1']:3}  > 0.25: {s['over_0.25']:3}")
    print(f"\nlink (hypothesis, not proof): Spearman gain vs PSI(test) {res['spearman_gain_vs_psi_test']:+.3f}, "
          f"gain vs adversarial gain {res['spearman_gain_vs_adversarial_gain']:+.3f}")
    for k, v in res["top20_overlap"].items():
        print(f"  top-20 overlap {k}: {len(v)} {v}")
    print(f"\n{'feature':12}{'gain':>8}{'PSI e/l':>9}{'PSI val':>9}{'PSI test':>9}{'adv gain':>10}  card.")
    for r in res["top20_gain_table"]:
        print(f"{r['feature']:12}{r['gain_share']:8.4f}{r['psi_early_late_train']:9.4f}{r['psi_val']:9.4f}"
              f"{r['psi_test']:9.4f}{r['adversarial_gain_share']:10.4f}  {'yes' if r['cardinality_alone'] else ''}")
    pr = res["predictions"]
    print(f"\npredictions: (a) predicted 'stays flat', got {pr['a']}; (b) predicted 'net faster', got '{pr['b']}'; "
          f"(c) predicted 0.6-0.8, got {pr['c']:.4f}")
    print_retrain(_read("phase6_retrain.json")["cells"])


# ---------- charts (aggregates only) ----------

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOR = {"lgbm": "#2a78d6", "nn": "#eb6834"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def weekly_chart(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W = res["weeks"]
    x = [r["week"] for r in W]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 2, 1.3]})
    in_sample = [r["week"] for r in W if r["in_sample"]]
    for ax in axes:
        _style(ax)
        ax.axvspan(-0.5, max(in_sample) + 0.5, color="#f0efec", zorder=0)
    for ax, key, label in ((axes[0], "roc_auc", "ROC-AUC"), (axes[1], "pr_auc", "PR-AUC")):
        for m in MODELS:
            ax.fill_between(x, [r[m][key + "_95"][0] for r in W], [r[m][key + "_95"][1] for r in W],
                            color=COLOR[m], alpha=0.18, lw=0)
            ax.plot(x, [r[m][key] for r in W], color=COLOR[m], lw=2, marker="o", ms=4, label=NAMES[m])
        ax.set_ylabel(f"{label}\n(mean of 5 seeds)", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    axes[2].bar(x, [100 * r["fraud_rate"] for r in W], color="#86b6ef", width=0.7)
    axes[2].set_ylabel("fraud rate %", color=INK2, fontsize=9)
    axes[2].set_xlabel("week (from first payment); shaded = 95% day-bootstrap range", color=INK2, fontsize=9)
    blocks = {"train (in-sample)": [], "val": res["val_weeks"], "test": res["test_weeks"]}
    blocks["train (in-sample)"] = in_sample
    for name, ws in blocks.items():
        if ws:
            axes[0].text((min(ws) + max(ws)) / 2, 1.02, name, transform=axes[0].get_xaxis_transform(), ha="center",
                         color=INK2, fontsize=9)
    fig.suptitle("Frozen models, scored week by week", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    REPORTS.mkdir(exist_ok=True)
    fig.savefig(REPORTS / "weekly_auc.png", dpi=130, facecolor=SURFACE)


def psi_chart(t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = pd.Series({c: x["psi"] for c, x in t["train_vs_test"].items()}).nlargest(20).index[::-1]
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    yy = np.arange(len(top))
    for name, color, label, mk in (("early_vs_late_train", "#8a8984", "early vs late train (within-train drift)", "s"),
                                   ("train_vs_val", COLOR["nn"], "train vs validation", "o"),
                                   ("train_vs_test", COLOR["lgbm"], "train vs test", "o")):
        ax.scatter([max(t[name][c]["psi"], 1e-3) for c in top], yy, color=color, s=40, marker=mk, label=label,
                   zorder=3, edgecolor=SURFACE, lw=1)
    for v in (0.1, 0.25):
        ax.axvline(v, color=INK2, lw=1, ls="--", zorder=1)
        ax.text(v, len(top) - 0.3, f" {v}", color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks(yy, top)
    ax.set_xlabel("PSI (log scale; dashed lines are conventions, not laws)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    ax.set_title("Top 20 drifting features (by train vs test PSI)", color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    REPORTS.mkdir(exist_ok=True)
    fig.savefig(REPORTS / "psi_top20.png", dpi=130, facecolor=SURFACE)


if __name__ == "__main__":
    args = sys.argv[1:]
    steps = {"weekly": weekly, "psi": psi_step, "adversarial": adversarial, "retrain": retrain, "report": report}
    if args[:1] == ["score"] and args[1:2] and args[1] in MODELS:
        score(args[1])
    elif len(args) == 1 and args[0] in steps:
        steps[args[0]]()
    else:
        sys.exit(__doc__)
