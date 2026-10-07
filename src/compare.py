"""Phase 4: equal tuning effort for both models, then each winner retrained on fresh seeds.

Validation picks the settings AND the stopping point, so validation scores are slightly optimistic for both.
LightGBM and PyTorch deadlock in one process (two OpenMP libraries), so `tune` and `seeds` train ONE model
and import only that model's library. Results go to metrics/ as aggregates; `report` imports neither.
Test rows are never used: every model's train function calls split_by_time and discards the test block.

    python -m src.compare tune lgbm|nn
    python -m src.compare seeds lgbm|nn
    python -m src.compare report
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.split import split_by_time

METRICS = Path(__file__).resolve().parent.parent / "metrics"
N_TRIALS = 20            # per model; trial 1 is the Phase 2/3 config, the other 19 are random draws
SEARCH_SEED = 0          # draws the random configs; same for both models
TUNE_SEED = 42           # training seed for every tuning trial, as in Phases 2-3
SEEDS = (1, 2, 3, 4, 5)  # fresh seeds for each winner; none is TUNE_SEED
CLOSE = 0.005            # verdict rule (DECISIONS.md): mean val ROC-AUC gap under this = "gap closed"
EDGE = 0.10              # a winning value in the outer 10% of its range is flagged

# ("log", lo, hi) log-uniform, ("intlog", lo, hi) the same rounded to a whole number, ("lin", lo, hi) uniform,
# a list: pick one. Both models get 6 knobs, including the learning rate.
SPACE = {
    "lgbm": {
        "learning_rate": ("log", 0.02, 0.1),
        "num_leaves": ("intlog", 16, 256),
        "min_child_samples": ("intlog", 20, 500),
        "colsample_bytree": ("lin", 0.3, 1.0),
        "subsample": ("lin", 0.5, 1.0),
        "reg_lambda": ("log", 1e-3, 10.0),
    },
    "nn": {
        "lr": ("log", 3e-4, 3e-3),
        "layers": [1, 2, 3],
        "width": [128, 256, 512],
        "dropout": ("lin", 0.0, 0.5),
        "batch": [512, 1024, 2048],
        "weight_decay": ("log", 1e-6, 1e-3),
    },
}
# Trial 1: exactly the Phase 2 / Phase 3 settings (some lie outside the ranges, e.g. reg_lambda 0).
BASELINE = {
    "lgbm": {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
             "colsample_bytree": 1.0, "subsample": 1.0, "reg_lambda": 0.0},
    "nn": {"lr": 1e-3, "layers": 2, "width": 256, "dropout": 0.3, "batch": 1024, "weight_decay": 0.0},
}
NAMES = {"lgbm": "LightGBM", "nn": "neural net"}


def _draw(rng, spec):
    if isinstance(spec, list):
        return spec[rng.integers(len(spec))]
    kind, lo, hi = spec
    if kind == "lin":
        return float(rng.uniform(lo, hi))
    v = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    return int(round(v)) if kind == "intlog" else v


def sample_configs(model, n=N_TRIALS):
    """Random search: the baseline, then n-1 draws. Same method and count for both models."""
    rng = np.random.default_rng(SEARCH_SEED)
    return [dict(BASELINE[model])] + [{k: _draw(rng, s) for k, s in SPACE[model].items()} for _ in range(n - 1)]


def run(model, df, cfg, seed, data=None, **caps):
    """Train one model on train rows, early-stop and score on validation.

    hit_cap: the best point was within `patience` of the tree/epoch cap, so the stopping rule never fired."""
    if model == "lgbm":
        from src import lgbm
        t = time.time()
        m, _, _, vx, vy = lgbm.train(df, seed, params={**cfg, "subsample_freq": 1}, **caps)
        cap = caps.get("max_trees", lgbm.MAX_TREES)
        return {"val": {k: float(v) for k, v in lgbm.scores(m, vx, vy).items()},
                "stop_at": int(m.best_iteration_), "hit_cap": bool(m.best_iteration_ > cap - lgbm.PATIENCE),
                "seconds": round(time.time() - t, 1)}
    from src import nn
    knobs = {"hidden": tuple(cfg["width"] // 2 ** i for i in range(cfg["layers"])),  # each layer half the last
             **{k: cfg[k] for k in ("dropout", "lr", "batch", "weight_decay")}}
    info = nn.train(df, seed, cfg=knobs, data=data, verbose=False, **caps)[2]
    cap = caps.get("max_epochs", nn.MAX_EPOCHS)
    return {"val": info["val"], "stop_at": info["best_epoch"],
            "hit_cap": info["best_epoch"] > cap - nn.PATIENCE, "seconds": info["train_seconds"]}


def fingerprint(df):
    """Row counts plus a hash of the TransactionIDs in train and validation: proves both processes used the same rows."""
    tr, val, _ = split_by_time(df)  # test block is discarded, never named
    h = lambda s: str(int(pd.util.hash_pandas_object(s["TransactionID"], index=False).sum()))
    return {"n_train": len(tr), "n_val": len(val), "train_hash": h(tr), "val_hash": h(val)}


def _stats(v):
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "min": float(min(v)), "max": float(max(v))}


def verdict(lgbm_roc, nn_roc):
    """Fixed before any run (DECISIONS.md). Inputs: val ROC-AUC per final seed."""
    if min(nn_roc) > max(lgbm_roc):
        return "flipped"
    overlap = min(nn_roc) <= max(lgbm_roc) and min(lgbm_roc) <= max(nn_roc)
    if abs(np.mean(lgbm_roc) - np.mean(nn_roc)) < CLOSE or overlap:
        return "gap closed"
    return "gap stayed"


def edges(model, cfg):
    """Winning knobs in the outer EDGE share of their range (log knobs on the log scale; lists: first/last)."""
    out = []
    for k, spec in SPACE[model].items():
        v = cfg[k]
        if isinstance(spec, list):
            hit = v in (spec[0], spec[-1])
        else:
            kind, lo, hi = spec
            f = (lambda x: x) if kind == "lin" else math.log
            hit = v <= lo or v >= hi or not EDGE <= (f(v) - f(lo)) / (f(hi) - f(lo)) <= 1 - EDGE
        if hit:
            out.append(k)
    return out


def _load(model):
    from src.data import load_raw
    df = load_raw(verbose=False)
    if model == "nn":
        from src import nn
        return df, nn.prepare(df)  # train-only statistics; no knob or seed changes them, so prepare once
    return df, None


def _write(name, obj):
    METRICS.mkdir(exist_ok=True)
    (METRICS / name).write_text(json.dumps(obj, indent=2))


def _read(name):
    return json.loads((METRICS / name).read_text())


def tune(model):
    df, data = _load(model)
    trials = []
    for i, cfg in enumerate(sample_configs(model), 1):
        r = run(model, df, cfg, TUNE_SEED, data)
        trials.append({"trial": i, "config": cfg, **r})
        print(f"{model} trial {i:2d}/{N_TRIALS}  val ROC-AUC {r['val']['roc_auc']:.4f}  PR-AUC {r['val']['pr_auc']:.4f}"
              f"  stop {r['stop_at']}{' CAP' if r['hit_cap'] else ''}  {r['seconds']:.0f} s", flush=True)
    best = max(trials, key=lambda t: t["val"]["roc_auc"])
    _write(f"phase4_tune_{model}.json", {
        "model": model, "search": "random", "n_trials": N_TRIALS, "search_seed": SEARCH_SEED,
        "tune_seed": TUNE_SEED, "select_on": "val roc_auc", "space": SPACE[model], "rows": fingerprint(df),
        "best_trial": best["trial"], "best_config": best["config"], "trials": trials,
    })
    print(f"best: trial {best['trial']}  val ROC-AUC {best['val']['roc_auc']:.4f}  {best['config']}")


def seeds(model):
    cfg = _read(f"phase4_tune_{model}.json")["best_config"]
    df, data = _load(model)
    runs = []
    for s in SEEDS:
        runs.append({"seed": s, **run(model, df, cfg, s, data)})
        print(f"{model} seed {s}  val ROC-AUC {runs[-1]['val']['roc_auc']:.4f}  PR-AUC {runs[-1]['val']['pr_auc']:.4f}",
              flush=True)
    _write(f"phase4_seeds_{model}.json", {
        "model": model, "config": cfg, "seeds": list(SEEDS), "rows": fingerprint(df), "runs": runs,
        "val": {m: _stats([r["val"][m] for r in runs]) for m in ("roc_auc", "pr_auc")},
    })


def report():
    t = {m: _read(f"phase4_tune_{m}.json") for m in SPACE}
    s = {m: _read(f"phase4_seeds_{m}.json") for m in SPACE}
    # Refuse to compare unless the comparison is like-for-like.
    for ok, msg in [
        (all(len(t[m]["trials"]) == N_TRIALS for m in SPACE), "trial counts differ"),
        (len({json.dumps(f[m]["rows"]) for f in (t, s) for m in SPACE}) == 1, "train/validation rows differ"),
        (all(s[m]["seeds"] == list(SEEDS) for m in SPACE), "final seeds differ"),
        (len(set(SEEDS)) == len(SEEDS) and TUNE_SEED not in SEEDS, "final seeds repeat or reuse the tuning seed"),
        (all(s[m]["config"] == t[m]["best_config"] for m in SPACE), "seed runs did not use the tuned winner"),
    ]:
        if not ok:
            sys.exit(f"refusing to compare: {msg}")

    out = {"verdict": verdict(*[[r["val"]["roc_auc"] for r in s[m]["runs"]] for m in SPACE]), "models": {}}
    for m in SPACE:
        trials, best = t[m]["trials"], t[m]["trials"][t[m]["best_trial"] - 1]
        st = s[m]["val"]
        notes = {}
        for k in ("roc_auc", "pr_auc"):
            if st[k]["std"] == 0:
                notes[k] = ("std 0: the config uses all rows and all columns, so there is no randomness and every "
                            "seed builds the same model" if m == "lgbm" else "std 0: unexpected, check the seeds")
        out["models"][m] = {
            "baseline_val": trials[0]["val"], "best_trial": t[m]["best_trial"], "best_trial_val": best["val"],
            "best_config": t[m]["best_config"], "seeds_val": st, "std_notes": notes,
            "trials_hit_cap": sum(r["hit_cap"] for r in trials), "seeds_hit_cap": sum(r["hit_cap"] for r in s[m]["runs"]),
            "tune_seconds": round(sum(r["seconds"] for r in trials)),
            "edge_knobs": edges(m, t[m]["best_config"]),
        }
    g = out["models"]
    out["gap_lgbm_minus_nn"] = {k: g["lgbm"]["seeds_val"][k]["mean"] - g["nn"]["seeds_val"][k]["mean"]
                                for k in ("roc_auc", "pr_auc")}
    out["caveats"] = [
        "Validation chose both the settings and the stopping point, so both validation scores are slightly optimistic.",
        "All seeds share one time split and one validation month; the spread does not cover a different period.",
        "The spread does not cover the search itself: a different search seed could pick a different winner.",
        "An edge knob means a better value may lie outside the range: that model's score is a lower bound.",
        "Training seconds: LightGBM includes its per-run data conversion; network preprocessing (once per process) is not.",
    ]
    _write("phase4_compare.json", out)

    row = lambda label, f: print(f"{label:30}" + "".join(f"{f(m):>24}" for m in SPACE))
    row("", lambda m: NAMES[m])
    for k, name in (("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")):
        row(f"baseline (trial 1) {name}", lambda m: f"{g[m]['baseline_val'][k]:.4f}")
        row(f"best trial {name} (seed {TUNE_SEED})", lambda m: f"{g[m]['best_trial_val'][k]:.4f} (#{g[m]['best_trial']})")
        row(f"{len(SEEDS)} seeds {name} mean +- std",
            lambda m: f"{g[m]['seeds_val'][k]['mean']:.4f} +- {g[m]['seeds_val'][k]['std']:.4f}")
        row(f"{len(SEEDS)} seeds {name} min-max",
            lambda m: f"{g[m]['seeds_val'][k]['min']:.4f}-{g[m]['seeds_val'][k]['max']:.4f}")
    row("tuning compute (s)", lambda m: g[m]["tune_seconds"])
    row("trials / seeds hitting cap", lambda m: f"{g[m]['trials_hit_cap']} / {g[m]['seeds_hit_cap']}")
    row("winner at range edge", lambda m: ", ".join(g[m]["edge_knobs"]) or "none")
    for m in SPACE:
        for k, note in g[m]["std_notes"].items():
            print(f"{NAMES[m]} {k}: {note}")
        if g[m]["edge_knobs"]:
            print(f"{NAMES[m]}: winner at range edge ({', '.join(g[m]['edge_knobs'])}); its score is a lower bound."
                  " Ranges are not widened afterwards.")
    gap = out["gap_lgbm_minus_nn"]
    print(f"\nmean gap (LightGBM - neural net): ROC-AUC {gap['roc_auc']:+.4f}  PR-AUC {gap['pr_auc']:+.4f}")
    print(f"VERDICT (rule fixed in DECISIONS.md before running): {out['verdict']}")
    for c in out["caveats"]:
        print(f"- {c}")


if __name__ == "__main__":
    cmd, model = (sys.argv[1:] + [None])[:2]
    if cmd == "report":
        report()
    elif cmd in ("tune", "seeds") and model in SPACE:
        {"tune": tune, "seeds": seeds}[cmd](model)
    else:
        sys.exit(__doc__)
