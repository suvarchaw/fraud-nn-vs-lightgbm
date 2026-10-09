"""Phase 5: turn scores into money decisions, check calibration, and open the test set ONCE.

Cost model (C is an ASSUMPTION, swept over COSTS): a missed fraud costs its TransactionAmt, every flag costs C
(a caught fraud costs C, not its amount), a legit payment left alone costs 0.
LightGBM and PyTorch clash in one process, so every step that trains or scores is its own process.

    python -m src.business fit lgbm|nn       retrain the frozen Phase 4 winner on seeds 1-5; save models + val scores
    python -m src.business choose            calibrators, thresholds, top-1% cut-off, from VALIDATION only
    python -m src.business test --rehearsal  the whole test pipeline on VALIDATION rows; writes nothing
    python -m src.business test              THE one test run; a second needs --override "reason" (logged)
"""
import datetime
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.compare import SEEDS, _read, _stats, _write, nn_knobs
from src.split import DAY, DT_START, split_by_time

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "phase5"            # git-ignored: models, per-row scores, calibrators
LOCK = ROOT / "metrics" / "phase5_test_lock.json"
DECISIONS = ROOT / "DECISIONS.md"
MODELS = ("lgbm", "nn")
COSTS = (1, 2, 5, 10, 20, 50)  # review cost C in dollars: an assumption, so we sweep it
REF_C = 10                     # headline C (fixed before any run)
HEADLINE = ("lgbm", "calibrated")
TOP_SHARE = 0.01               # review-capacity baseline: flag the top 1% (cut-off from validation)
N_BINS = 10                    # equal-count groups for ECE and the reliability table
TIE = 1e-4                     # Platt wins unless isotonic's held-out Brier is lower by more than this
N_BOOT, BOOT_SEED = 1000, 0
EPS = 1e-7
DROP_BANDS = [(-1, 0.01, "< 0.01"), (0.01, 0.03, "0.01-0.03"), (0.03, 0.06, "0.03-0.06"), (0.06, 1, "> 0.06")]
RULES = ("calibrated", "raw", "threshold", "top1%")


# ---------- small maths: no LightGBM, no PyTorch ----------

def day_of(dt):
    return ((np.asarray(dt) - DT_START) // DAY).astype(int)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_calibrator(p, y, kind):
    """Returns a plain dict, so it can be saved as JSON and compared exactly."""
    if kind == "platt":
        m = LogisticRegression(C=np.inf).fit(logit(p)[:, None], y)  # C=inf: no shrinkage, a plain 2-number fit
        return {"kind": "platt", "a": float(m.coef_[0, 0]), "b": float(m.intercept_[0])}
    m = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
    return {"kind": "isotonic", "x": m.X_thresholds_.tolist(), "y": m.y_thresholds_.tolist()}


def calibrate(p, cal):
    if cal["kind"] == "platt":
        return 1 / (1 + np.exp(-(cal["a"] * logit(p) + cal["b"])))
    return np.interp(p, cal["x"], cal["y"])  # same as IsotonicRegression.predict with clipping


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def reliability(y, p, bins=N_BINS):
    """Equal-count groups by predicted %: (rows, mean predicted, real fraud rate) per group."""
    order = np.argsort(p, kind="stable")
    return [(len(g), float(p[g].mean()), float(y[g].mean())) for g in np.array_split(order, bins)]


def ece(y, p, bins=N_BINS):
    return float(sum(n * abs(mp - fr) for n, mp, fr in reliability(y, p, bins)) / len(y))


def best_threshold(p, y, amt, c):
    """Score cut-off that minimises cost on these rows (flag p >= t). inf means flag nothing."""
    order = np.argsort(-p, kind="stable")
    ps, caught = p[order], np.cumsum(np.where(y[order] == 1, amt[order], 0.0))
    k = np.arange(1, len(p) + 1)
    cost = c * k - caught                     # cost of flagging the top k, minus the constant "all fraud dollars"
    cost[:-1][ps[:-1] == ps[1:]] = np.inf     # cannot cut between equal scores
    best = int(np.argmin(cost))
    return float(ps[best]) if cost[best] < 0 else float("inf")


def outcome(y, amt, flag, c, n_days):
    """Money and hit rates for one set of flags at review cost c."""
    fraud = y == 1
    missed = float(amt[fraud & ~flag].sum())
    cost = c * int(flag.sum()) + missed
    nothing = float(amt[fraud].sum())
    tp = int((flag & fraud).sum())
    return {"cost": cost, "savings": nothing - cost, "savings_pct": 100 * (nothing - cost) / nothing,
            "flags_per_day": int(flag.sum()) / n_days, "precision": tp / flag.sum() if flag.any() else float("nan"),
            "recall": tp / fraud.sum(), "dollar_recall": float(amt[fraud & flag].sum()) / nothing}


def flags(rule, p, amt, seed_frozen, c):
    if rule == "calibrated":
        return calibrate(p, seed_frozen["calibrator"]) * amt > c  # expected loss > review cost
    if rule == "raw":
        return p * amt > c                                        # same formula, uncalibrated %
    if rule == "threshold":
        return p >= seed_frozen["thresholds"][str(c)]
    return p >= seed_frozen["top_cut"]


def amount_context(rows):
    a, f = rows["TransactionAmt"].astype("float64"), rows["isFraud"] == 1
    return {"rows": len(rows), "days": int(np.ptp(day_of(rows["TransactionDT"]))) + 1,
            "fraud_rate": float(f.mean()), "median_amt": float(a.median()), "mean_amt": float(a.mean()),
            "fraud_dollars": float(a[f].sum()), "median_fraud_amt": float(a[f].median())}


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ---------- choose: validation only ----------

def freeze(df, scores):
    """Every Phase 5 choice, from VALIDATION rows only.

    scores: {model: array (n_seeds, len(df))} aligned with df rows; only validation positions are read."""
    _, val, _ = split_by_time(df)
    pos = df.index.get_indexer(val.index)
    y, amt = val["isFraud"].to_numpy(), val["TransactionAmt"].to_numpy("float64")
    d = day_of(val["TransactionDT"])
    early = d < np.unique(d)[len(np.unique(d)) // 2]  # first half of validation days
    out = {}
    for m, s in scores.items():
        p = s[:, pos]
        held = {k: [brier(y[~early], calibrate(ps[~early], fit_calibrator(ps[early], y[early], k))) for ps in p]
                for k in ("platt", "isotonic")}
        kind = "isotonic" if np.mean(held["isotonic"]) < np.mean(held["platt"]) - TIE else "platt"
        out[m] = {"method": kind, "held_out_brier": {k: float(np.mean(v)) for k, v in held.items()},
                  "seeds": [{"calibrator": fit_calibrator(ps, y, kind),
                             "thresholds": {str(c): best_threshold(ps, y, amt, c) for c in COSTS},
                             "top_cut": float(np.quantile(ps, 1 - TOP_SHARE))} for ps in p]}
    return out


def choose():
    from src.data import load_raw
    df = load_raw(verbose=False)
    tr, val, _ = split_by_time(df)
    ids = pd.Index(df["TransactionID"])
    scores = {}
    for m in MODELS:
        f = np.load(OUT / f"{m}_val.npz")
        full = np.full((len(SEEDS), len(df)), np.nan)  # only validation positions are filled
        full[:, ids.get_indexer(f["ids"])] = f["p"]
        scores[m] = full
    frozen = freeze(df, scores)
    (OUT / "frozen.json").write_text(json.dumps(frozen))
    summary = {
        "frozen_sha256": sha(frozen), "costs": COSTS, "ref_c": REF_C, "headline": HEADLINE,
        "amounts": {"train": amount_context(tr), "val": amount_context(val)},  # test amounts only after freezing
        "models": {m: {"method": f["method"], "held_out_brier_val_late_half": f["held_out_brier"],
                       "platt_a_b": [[s["calibrator"]["a"], s["calibrator"]["b"]] for s in f["seeds"]]
                       if f["method"] == "platt" else None,
                       "isotonic_steps": [len(s["calibrator"]["x"]) for s in f["seeds"]]
                       if f["method"] == "isotonic" else None,
                       "threshold_mean": {c: float(np.mean([s["thresholds"][str(c)] for s in f["seeds"]])) for c in COSTS},
                       "top_cut": _stats([s["top_cut"] for s in f["seeds"]])} for m, f in frozen.items()},
        "caveat": "Validation already chose the settings and stopping point (Phase 4) and now also fits calibrators "
                  "and thresholds, so validation numbers are flattering. Test numbers are the honest ones.",
    }
    _write("phase5_val.json", summary)
    for k in ("train", "val"):
        a = summary["amounts"][k]
        print(f"{k:6} rows {a['rows']:,}  median amt ${a['median_amt']:.2f}  mean ${a['mean_amt']:.2f}  "
              f"fraud dollars ${a['fraud_dollars']:,.0f}")
    for m, f in frozen.items():
        print(f"{m}: calibrator {f['method']} (held-out Brier platt {f['held_out_brier']['platt']:.5f}, "
              f"isotonic {f['held_out_brier']['isotonic']:.5f})")
    print(f"frozen choices sha256 {summary['frozen_sha256']}")


# ---------- fit: train on train, score validation ----------

def fit(model):
    from src.data import load_raw
    cfg = _read(f"phase4_tune_{model}.json")["best_config"]
    phase4 = {r["seed"]: r["val"]["roc_auc"] for r in _read(f"phase4_seeds_{model}.json")["runs"]}
    df = load_raw(verbose=False)
    _, val, _ = split_by_time(df)
    OUT.mkdir(parents=True, exist_ok=True)
    ps = []
    if model == "nn":
        import torch
        from src import nn
        data = nn.prepare(df)
    for s in SEEDS:
        if model == "lgbm":
            from src import lgbm
            m, _, _, vx, _ = lgbm.train(df, s, params={**cfg, "subsample_freq": 1})
            m.booster_.save_model(str(OUT / f"lgbm_seed{s}.txt"), num_iteration=m.best_iteration_)
            p = m.predict_proba(vx, num_iteration=m.best_iteration_)[:, 1]
        else:
            net, _, _, _, (vn, vc, _) = nn.train(df, s, cfg=nn_knobs(cfg), data=data, verbose=False)
            torch.save(net.state_dict(), OUT / f"nn_seed{s}.pt")
            p = nn.predict(net, vn, vc)
        auc = roc_auc_score(val["isFraud"], p)
        print(f"{model} seed {s}  val ROC-AUC {auc:.4f}  (Phase 4: {phase4[s]:.4f})", flush=True)
        if abs(auc - phase4[s]) > 1e-9:
            sys.exit(f"seed {s} did not reproduce Phase 4; the frozen config was not rebuilt exactly")
        ps.append(p)
    np.savez(OUT / f"{model}_val.npz", ids=val["TransactionID"].to_numpy(), p=np.array(ps))


# ---------- test run ----------

def start_test_run(override=None):
    """Refuse a second test run unless overridden; an override is appended to DECISIONS.md. Returns the run id."""
    if LOCK.exists():
        prior = json.loads(LOCK.read_text())
        if not override:
            sys.exit(f"refusing: the test set was already evaluated (run {prior['run_id']}, status {prior['status']}). "
                     "Pass --override \"reason\" only if that run crashed before producing any result.")
        with DECISIONS.open("a") as f:
            f.write(f"- Test-run OVERRIDE {datetime.date.today()}: previous run {prior['run_id']} had status "
                    f"'{prior['status']}'. Reason: {override}\n")
    run_id = uuid.uuid4().hex
    frozen = json.loads((OUT / "frozen.json").read_text())
    LOCK.write_text(json.dumps({"run_id": run_id, "started": datetime.datetime.now().isoformat(timespec="seconds"),
                                "status": "running", "frozen_sha256": sha(frozen), "override": override}, indent=2))
    return run_id


def score(model, block, run_id):
    """Child process: score `block` rows with the 5 saved models. Test rows only inside the locked run."""
    if block == "test":
        lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
        if lock.get("run_id") != run_id or lock.get("status") != "running":
            sys.exit("refusing to score test rows outside the one locked test run")
    if model == "nn":
        import torch  # must load before LightGBM's library (src.lgbm below): the other order crashes PyTorch
    from src.data import load_raw
    from src.lgbm import feature_names
    df = load_raw(verbose=False)
    tr, val, test = split_by_time(df)
    rows = test if block == "test" else val
    if model == "lgbm":
        import lightgbm as lgb
        feats = feature_names(df)
        p = [lgb.Booster(model_file=str(OUT / f"lgbm_seed{s}.txt")).predict(rows[feats]) for s in SEEDS]
    else:
        from src import nn
        nn.setup(0)
        cfg = nn_knobs(_read("phase4_tune_nn.json")["best_config"])
        prep = nn.fit_prep(tr, feature_names(df))  # train-only statistics, rebuilt exactly as in training
        xn, xc = nn.transform(rows, prep)
        p = []
        for s in SEEDS:
            net = nn.Net(xn.shape[1], nn.slots(prep), cfg["dropout"], cfg["hidden"])
            net.load_state_dict(torch.load(OUT / f"nn_seed{s}.pt"))
            p.append(nn.predict(net, xn, xc))
    np.savez(OUT / f"{model}_{block}_scored.npz", ids=rows["TransactionID"].to_numpy(), p=np.array(p))


def evaluate(rows, scores, frozen, val_auc=None):
    """Apply the frozen choices to `rows` (the evaluation block). scores: {model: (n_seeds, len(rows))}."""
    y, amt = rows["isFraud"].to_numpy(), rows["TransactionAmt"].to_numpy("float64")
    days = day_of(rows["TransactionDT"])
    n_days = len(np.unique(days))
    res = {"amounts": amount_context(rows), "costs": {}, "calibration": {}, "auc": {}}
    nothing = float(amt[y == 1].sum())
    res["baselines"] = {str(c): {"flag_nothing": outcome(y, amt, np.zeros(len(y), bool), c, n_days),
                                 "flag_everything": outcome(y, amt, np.ones(len(y), bool), c, n_days)}
                        for c in COSTS}
    contrib = {}  # per-row savings vs flag-nothing at REF_C, for the bootstrap
    for m, P in scores.items():
        fz = frozen[m]["seeds"]
        res["costs"][m] = {r: {} for r in RULES}
        for r in RULES:
            for c in COSTS:
                runs = [outcome(y, amt, flags(r, p, amt, f, c), c, n_days) for p, f in zip(P, fz)]
                res["costs"][m][r][str(c)] = {k: _stats([o[k] for o in runs]) for k in runs[0]}
            contrib[(m, r)] = np.mean([np.where(flags(r, p, amt, f, REF_C), (y == 1) * amt - REF_C, 0.0)
                                       for p, f in zip(P, fz)], axis=0)
        cal = {"raw": list(P), "calibrated": [calibrate(p, f["calibrator"]) for p, f in zip(P, fz)]}
        res["calibration"][m] = {k: {"brier": _stats([brier(y, p) for p in v]), "ece": _stats([ece(y, p) for p in v]),
                                     "ece_per_seed": [ece(y, p) for p in v],
                                     "reliability": [[float(x) for x in t] for t in
                                                     np.mean([reliability(y, p) for p in v], axis=0)]}
                                 for k, v in cal.items()}
        res["auc"][m] = {"roc_auc": _stats([roc_auc_score(y, p) for p in P]),
                         "pr_auc": _stats([average_precision_score(y, p) for p in P])}
        if val_auc:
            drop = val_auc[m] - res["auc"][m]["roc_auc"]["mean"]
            res["auc"][m]["drop_from_val"] = drop
            res["auc"][m]["drop_band"] = next(b for lo, hi, b in DROP_BANDS if lo <= drop < hi)
    e = {m: res["calibration"][m]["raw"]["ece_per_seed"] for m in scores}
    res["verdict_a_raw_calibration"] = ("lgbm" if max(e["lgbm"]) < min(e["nn"]) else
                                        "nn" if max(e["nn"]) < min(e["lgbm"]) else "about the same")
    res["headline"] = {"model": HEADLINE[0], "rule": HEADLINE[1], "c": REF_C,
                       **{k: res["costs"][HEADLINE[0]][HEADLINE[1]][str(REF_C)][k]["mean"]
                          for k in ("savings", "savings_pct", "dollar_recall", "recall", "precision", "flags_per_day")},
                       "fraud_dollars": nothing}
    res["bootstrap"] = bootstrap(y, days, scores, contrib)
    return res


def day_draws(days, n=N_BOOT, seed=BOOT_SEED):
    """Each draw: as many whole days as the block has, picked with replacement.
    Returns (row positions of each day, draws as day numbers 0..days-1)."""
    uniq = np.unique(days)
    return [np.flatnonzero(days == d) for d in uniq], np.random.default_rng(seed).choice(len(uniq), size=(n, len(uniq)))


def rows_in(rows_of, draw):
    return np.concatenate([rows_of[i] for i in draw])


def bootstrap(y, days, scores, contrib):
    """Paired day-block bootstrap: both models (5 seeds each) are scored on the same drawn days."""
    rows_of, draws = day_draws(days)
    day_sum = {k: np.array([v[r].sum() for r in rows_of]) for k, v in contrib.items()}
    auc = {k: [] for k in ("roc_auc", "pr_auc")}
    sav = {k: [] for k in contrib}
    for dr in draws:
        idx = rows_in(rows_of, dr)
        for k, f in (("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)):
            auc[k].append([np.mean([f(y[idx], p[idx]) for p in scores[m]]) for m in MODELS])
        for k in contrib:
            sav[k].append(day_sum[k][dr].sum())
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = {"draws": N_BOOT, "days": len(rows_of), "seed": BOOT_SEED}
    for k, v in auc.items():
        v = np.array(v)
        out[k] = {"lgbm_95": ci(v[:, 0]), "nn_95": ci(v[:, 1]), "gap_lgbm_minus_nn_95": ci(v[:, 0] - v[:, 1]),
                  "gap_draws_below_0": float(np.mean(v[:, 0] <= v[:, 1]))}
    out[f"savings_at_c{REF_C}_95"] = {f"{m}/{r}": ci(v) for (m, r), v in sav.items()}
    return out


def run_block(rehearsal=False, override=None):
    """Children score the block (each model its own process); then the frozen choices are applied."""
    frozen = json.loads((OUT / "frozen.json").read_text())
    if sha(frozen) != _read("phase5_val.json")["frozen_sha256"]:
        sys.exit("frozen choices changed since `choose`; refusing")
    block = "val" if rehearsal else "test"
    run_id = "rehearsal" if rehearsal else start_test_run(override)
    for m in MODELS:
        subprocess.run([sys.executable, "-m", "src.business", "_score", m, block, run_id], cwd=ROOT, check=True)
    if not rehearsal and json.loads(LOCK.read_text())["frozen_sha256"] != sha(frozen):
        sys.exit("frozen choices changed during the run; refusing")
    from src.data import load_raw
    df = load_raw(verbose=False)
    _, val, test = split_by_time(df)
    rows = val if rehearsal else test
    scores = {}
    for m in MODELS:
        f = np.load(OUT / f"{m}_{block}_scored.npz")
        assert (f["ids"] == rows["TransactionID"].to_numpy()).all()
        scores[m] = f["p"]
        if rehearsal:  # saved models must reproduce the scores made right after training
            diff = np.abs(f["p"] - np.load(OUT / f"{m}_val.npz")["p"]).max()
            print(f"rehearsal: {m} reloaded models vs training-time val scores, max difference {diff:.2e}")
            assert diff < 1e-5, "saved models do not reproduce their validation scores"
    val_auc = {m: _read(f"phase4_seeds_{m}.json")["val"]["roc_auc"]["mean"] for m in MODELS}
    res = evaluate(rows, scores, frozen, val_auc)
    res.update({"block": block, "frozen_sha256": sha(frozen), "cost_model_assumption":
                "C (review cost per flag) is assumed, not known; a caught fraud is assumed fully saved."})
    report(res)
    if not rehearsal:
        _write("phase5_test.json", res)
        lock = json.loads(LOCK.read_text())
        LOCK.write_text(json.dumps({**lock, "status": "done"}, indent=2))


def report(res):
    a, h = res["amounts"], res["headline"]
    print(f"\n[{res['block']}] rows {a['rows']:,}  days {a['days']}  median amt ${a['median_amt']:.2f}  "
          f"mean ${a['mean_amt']:.2f}  fraud dollars ${a['fraud_dollars']:,.0f}")
    print(f"HEADLINE (fixed in advance): LightGBM + calibrated rule, C=${REF_C} (C is an assumption): "
          f"saves ${h['savings']:,.0f} ({h['savings_pct']:.1f}% of fraud dollars net of review cost), "
          f"catches {100 * h['dollar_recall']:.1f}% of fraud dollars, {100 * h['recall']:.1f}% of frauds, "
          f"{h['flags_per_day']:.0f} flags/day, precision {h['precision']:.3f}")
    print(f"\nsavings $ by C (mean over 5 seeds){'':4}" + "".join(f"{'C=$' + str(c):>11}" for c in COSTS))
    for c_row in ("flag_everything",):
        print(f"{'  ' + c_row:38}" + "".join(f"{res['baselines'][str(c)][c_row]['savings']:>11,.0f}" for c in COSTS))
    for m in MODELS:
        for r in RULES:
            print(f"  {m:5} {r:30}" + "".join(f"{res['costs'][m][r][str(c)]['savings']['mean']:>11,.0f}" for c in COSTS))
    print(f"\nat C=${REF_C}: {'':24}dollar-recall  recall  precision  flags/day")
    for m in MODELS:
        for r in RULES:
            o = res["costs"][m][r][str(REF_C)]
            print(f"  {m:5} {r:30}{o['dollar_recall']['mean']:>10.3f}{o['recall']['mean']:>9.3f}"
                  f"{o['precision']['mean']:>10.3f}{o['flags_per_day']['mean']:>10.0f}")
    print("\ncalibration (mean over seeds)      Brier raw -> cal      ECE raw -> cal")
    for m in MODELS:
        c = res["calibration"][m]
        print(f"  {m:5}{'':28}{c['raw']['brier']['mean']:.5f} -> {c['calibrated']['brier']['mean']:.5f}"
              f"    {c['raw']['ece']['mean']:.5f} -> {c['calibrated']['ece']['mean']:.5f}")
    print(f"verdict (a) better calibrated raw: {res['verdict_a_raw_calibration']}")
    for m in MODELS:
        u = res["auc"][m]
        print(f"  {m:5} ROC-AUC {u['roc_auc']['mean']:.4f} +- {u['roc_auc']['std']:.4f}  PR-AUC {u['pr_auc']['mean']:.4f}"
              f"  drop from val {u['drop_from_val']:+.4f} (band {u['drop_band']})")
    b = res["bootstrap"]
    for k in ("roc_auc", "pr_auc"):
        lo, hi = b[k]["gap_lgbm_minus_nn_95"]
        print(f"bootstrap ({b['draws']} draws of {b['days']} whole days) {k} gap LightGBM - net: 95% range "
              f"{lo:+.4f} to {hi:+.4f}")
    lo, hi = b[f"savings_at_c{REF_C}_95"]["/".join(HEADLINE)]
    print(f"bootstrap headline savings 95% range: ${lo:,.0f} to ${hi:,.0f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["fit"] and args[1:2] and args[1] in MODELS:
        fit(args[1])
    elif args == ["choose"]:
        choose()
    elif args[:1] == ["_score"] and len(args) == 4:
        score(*args[1:])
    elif args[:1] == ["test"]:
        ov = args[args.index("--override") + 1] if "--override" in args else None
        run_block(rehearsal="--rehearsal" in args, override=ov)
    else:
        sys.exit(__doc__)
