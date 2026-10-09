"""Phase 8: scheduled retraining with delayed labels. EXPLORATORY and DESCRIPTIVE: the test set is spent, and the
evaluation window (days 98-181) reuses months seen before (validation and test). Nothing here changes a reported
model, threshold or config, and nothing here picks a model for the service.

A fraud team deploys on day R0 and judges each following 7-day block with the model its policy has active then.
A retrain on day r may learn only from rows whose label has arrived: s + D days <= start of day r (train_mask).
LightGBM and PyTorch clash in one process, so every step is its own process:

    python -m src.retrain fit lgbm   recipe check, then every distinct training set x 5 seeds -> models/phase8/
    python -m src.retrain fit nn     the same for the net
    python -m src.retrain report     savings, AUC, client segments, bootstrap, verdicts   (no ML library)
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.business import REF_C, day_draws, day_of, flags, outcome
from src.compare import SEEDS, _read, _stats, _write, nn_knobs
from src.drift import INK, INK2, NAMES, P5, REPORTS, SURFACE, _style, ci, model_hashes
from src.split import DAY, DT_START, split_by_time

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "phase8"  # git-ignored: per-row scores
MODELS = ("lgbm", "nn")
R0, BLOCK, N_BLOCKS = 98, 7, 12   # first deployment day; blocks [r, r + 7) cover days 98-181
BLOCKS = tuple(R0 + BLOCK * k for k in range(N_BLOCKS))  # the decision days
DELAYS = (0, 7, 30, 60)           # label delay in days; 0 is an unrealistic upper bound, never in a verdict
EVERY = {"never": None, "4-weekly": 28, "2-weekly": 14, "weekly": 7}
WEEKLY_DELAYS = (0, 7)            # compute budget (DECISIONS.md): weekly only where it is almost free
HEADLINE_D = 30
BAND = 1000                       # prediction (a) amendment: a mean gain within +-$1,000 counts as "no gain"
PREDICTED = {"a": "neither", "b": "more", "c": "LightGBM"}
STATUS = ("Exploratory and descriptive. The test set is spent; days 98-181 contain validation and test, seen in "
          "earlier phases. No reported model, threshold or config is changed. C = $10 is an assumption.")


# ---------- the clock: no LightGBM, no PyTorch ----------

def train_mask(dt, r, d):
    """Rows a retrain on day r may learn from: the label has arrived (s + d days <= R) and the payment is before R."""
    R = DT_START + r * DAY
    dt = np.asarray(dt)
    return (dt + d * DAY <= R) & (dt < R)


def schedule(policy):
    every = EVERY[policy]
    return [R0] if every is None else list(range(R0, R0 + BLOCK * N_BLOCKS, every))


def active_day(sched, block):
    """The latest retrain on or before the block's start (ValueError before the first deployment)."""
    return max(r for r in sched if r <= block)


def block_start(day):
    """Decision day of each row's evaluation block; -1 outside the window."""
    day = np.asarray(day)
    return np.where(day >= R0, R0 + (day - R0) // BLOCK * BLOCK, -1)


def cells():
    return [(d, p) for d in DELAYS for p in EVERY if p != "weekly" or d in WEEKLY_DELAYS]


def plan(dt):
    """The shared grid. models = {key: {"mask", "uses": [(d, r)], "blocks": [block days it is active for]}};
    active = {(d, policy, block): key}. A model is its training-row set: equal masks share one key (and one fit)."""
    models, active, key_of = {}, {}, {}
    for d, p in cells():
        sched = schedule(p)
        for b in BLOCKS:
            r = active_day(sched, b)
            if (d, r) not in key_of:
                m = train_mask(dt, r, d)
                k = hashlib.sha256(np.packbits(m).tobytes()).hexdigest()[:12]
                g = models.setdefault(k, {"mask": m, "uses": [], "blocks": set()})
                assert np.array_equal(g["mask"], m), "hash collision"
                g["uses"].append((d, r))
                key_of[(d, r)] = k
            models[key_of[(d, r)]]["blocks"].add(b)
            active[(d, p, b)] = key_of[(d, r)]
    for g in models.values():
        g["uses"], g["blocks"] = sorted(g["uses"]), sorted(g["blocks"])
    return models, active


def client_key(df):
    """card1 + addr1 + (day - D1), the usual IEEE-CIS client id (audit A1). Missing where addr1 or D1 is missing."""
    start = pd.Series(day_of(df["TransactionDT"]) - df["D1"].to_numpy("float64"), index=df.index)
    key = df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_" + start.astype(str)
    return key.where(df["addr1"].notna() & df["D1"].notna())


def returning(dt, key, d):
    """Per row: does its client have a row whose label has arrived by the row's decision day (the train_mask rule)?
    Fixed per delay, so every policy and model is compared on the same rows. Missing key = new; outside window = False."""
    blk, has = block_start(day_of(dt)), key.notna().to_numpy()
    out = np.zeros(len(key), bool)
    for b in BLOCKS:
        seen = key[train_mask(dt, b, d) & has].unique()
        rows = (blk == b) & has
        out[rows] = key[rows].isin(seen).to_numpy()
    return out


def contribution(p, y, amt, c=REF_C):
    """Each row's savings vs flag-nothing under the Phase 5 raw rule (flag if p x amount > C): a flag earns the
    fraud amount it catches and costs C. Summed over rows it equals business.outcome()["savings"]."""
    return np.where(flags("raw", p, amt, None, c), (y == 1) * amt - c, 0.0)


# ---------- fitting: fixed recipes, nothing tuned ----------

def recipe(model):
    """Phase 4 winning config and each seed's Phase 4 stopping point (trees for LightGBM, best epoch for the net)."""
    s = _read(f"phase4_seeds_{model}.json")
    return s["config"], {r["seed"]: r["stop_at"] for r in s["runs"]}


def fit_lgbm(df, feats, train, seed, trees, cfg, **over):
    import lightgbm as lgb
    from src.lgbm import LEARNING_RATE
    m = lgb.LGBMClassifier(n_estimators=trees, learning_rate=LEARNING_RATE, random_state=seed, deterministic=True,
                           force_row_wise=True, n_jobs=4, verbose=-1)
    m.set_params(**{**cfg, "subsample_freq": 1, **over})  # `over`: test caps only
    return m.fit(df.loc[train, feats], df.loc[train, "isFraud"])


def lgbm_scores(df, feats, train, ev, seeds, counts, cfg, **over):
    """One model per seed on `train` rows, fixed tree count, no early stopping; scores of the `ev` rows."""
    X = df.loc[ev, feats]
    return np.array([fit_lgbm(df, feats, train, s, counts[s], cfg, **over).predict_proba(X)[:, 1] for s in seeds])


def fit_nn(xn, xc, y, prep, seed, epochs, cfg):
    """nn.train without the validation check: the same calls in the same order, for a fixed number of epochs."""
    import torch
    from src import nn
    nn.setup(seed)
    net = nn.Net(xn.shape[1], nn.slots(prep), cfg["dropout"], cfg["hidden"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        nn.run_epoch(net, opt, xn, xc, y, gen, cfg["batch"])
    return net


def nn_scores(df, feats, train, ev, seeds, counts, cfg):
    """Preprocessing fitted on this model's own training rows (shared by its seeds), then one net per seed."""
    import torch
    from src import nn
    rows = df[train]
    prep = nn.fit_prep(rows, feats)
    xn, xc = nn.transform(rows, prep)
    y = torch.from_numpy(rows["isFraud"].to_numpy("float32"))
    en, ec = nn.transform(df[ev], prep)
    return np.array([nn.predict(fit_nn(xn, xc, y, prep, s, counts[s], cfg), en, ec) for s in seeds])


def recipe_check(model, df, feats, cfg, counts):
    """Seed 1 on Phase 5's own train rows must rebuild the frozen model's validation scores exactly."""
    tr, val, _ = split_by_time(df)
    train, ev = df.index.isin(tr.index), df.index.isin(val.index)
    p = (lgbm_scores if model == "lgbm" else nn_scores)(df, feats, train, ev, SEEDS[:1], counts, cfg)[0]
    ref = np.load(P5 / f"{model}_val.npz")
    assert (ref["ids"] == df.loc[ev, "TransactionID"].to_numpy()).all()
    diff = float(np.abs(p - ref["p"][0]).max())
    print(f"recipe check: {model} seed {SEEDS[0]} on Phase 5 train rows vs frozen val scores, max difference {diff:.1e}")
    if diff > 1e-9:
        sys.exit("the fixed-count recipe does not rebuild the frozen model; refusing")
    return diff


def fit(model):
    if model == "nn":
        import torch  # noqa: F401  must load before LightGBM's library (src.lgbm below): the other order crashes
    from src.data import load_raw
    from src.lgbm import feature_names
    before = model_hashes()
    df = load_raw(verbose=False)
    feats, dt = feature_names(df), df["TransactionDT"].to_numpy()
    config, counts = recipe(model)
    cfg = nn_knobs(config) if model == "nn" else config
    score = lgbm_scores if model == "lgbm" else nn_scores
    diff = recipe_check(model, df, feats, cfg, counts)
    models, _ = plan(dt)
    blk = block_start(day_of(dt))
    OUT.mkdir(parents=True, exist_ok=True)
    todo = {k: g for k, g in models.items() if not (OUT / f"{model}_{k}.npz").exists()}  # resumable
    total, done, t0 = sum(int(g["mask"].sum()) for g in todo.values()), 0, time.time()
    print(f"{len(models)} training sets, {len(todo)} to fit, {total:,} training rows x {len(SEEDS)} seeds", flush=True)
    for k, g in sorted(todo.items(), key=lambda kv: kv[1]["mask"].sum()):
        ev = np.isin(blk, g["blocks"])
        p = score(df, feats, g["mask"], ev, SEEDS, counts, cfg)
        np.savez(OUT / f"{model}_{k}.npz", ids=df.loc[ev, "TransactionID"].to_numpy(), p=p)
        done += int(g["mask"].sum())
        el = time.time() - t0
        print(f"{model} {k}  train rows {int(g['mask'].sum()):,}  (D, r) {g['uses']}  blocks {len(g['blocks'])}  "
              f"{el / 60:.0f} min, projected {el * total / done / 60:.0f} min", flush=True)
    assert model_hashes() == before, "a frozen file changed during fitting"
    (OUT / f"{model}_fit.json").write_text(json.dumps({"recipe_check_max_diff": diff, "counts": counts,
                                                        "config": config, "model_hashes": before}))


# ---------- report: no LightGBM, no PyTorch ----------

def classify(rng, pos, neg):
    return pos if rng[0] > 0 else neg if rng[1] < 0 else "cannot tell"


def score_a(mean, rng):
    """Prediction (a) 'neither', amended rule (DECISIONS.md), checked in this order."""
    if rng[1] < 0 or abs(mean) <= BAND:
        return "confirmed"
    if mean > BAND and rng[0] > 0:
        return "wrong"
    return "not confirmed either way"


def load_cells(df, models, active, E):
    """{(model, d, policy): scores (seeds, evaluation rows)}, each block taken from its policy's active model."""
    rel = np.full(len(df), -1)
    rel[E] = np.arange(len(E))
    pos = pd.Index(df["TransactionID"])
    be = block_start(day_of(df["TransactionDT"].to_numpy()[E]))
    S = {}
    for m in MODELS:
        files = {}
        for k, g in models.items():
            f = OUT / f"{m}_{k}.npz"
            if not f.exists():
                sys.exit(f"missing {f.name}: run `python -m src.retrain fit {m}` first")
            z = np.load(f)
            i = pos.get_indexer(z["ids"])
            if (i < 0).any() or not np.array_equal(rel[i], np.flatnonzero(np.isin(be, g["blocks"]))):
                sys.exit(f"{f.name}: scored rows do not match the model's blocks; refusing")
            files[k] = (rel[i], z["p"])
        for d, p in cells():
            s = np.full((len(SEEDS), len(E)), np.nan)
            for b in BLOCKS:
                r, P = files[active[(d, p, b)]]
                sel = be[r] == b
                s[:, r[sel]] = P[:, sel]
            assert not np.isnan(s).any()
            S[(m, d, p)] = s
    return S


def report():
    from src.data import load_raw
    fits = {}
    for m in MODELS:
        f = OUT / f"{m}_fit.json"
        if not f.exists():
            sys.exit(f"missing {f.name}: run `python -m src.retrain fit {m}` first")
        fits[m] = json.loads(f.read_text())
        if fits[m]["model_hashes"] != model_hashes():
            sys.exit("a frozen Phase 5 file changed since `fit`; refusing")
    df = load_raw(verbose=False)
    dt, y, amt = df["TransactionDT"].to_numpy(), df["isFraud"].to_numpy(), df["TransactionAmt"].to_numpy("float64")
    day = day_of(dt)
    E = np.flatnonzero(block_start(day) >= 0)
    models, active = plan(dt)
    S = load_cells(df, models, active, E)
    ye, ae, de = y[E], amt[E], day[E]
    n_days = len(np.unique(de))
    _, inv = np.unique(de, return_inverse=True)
    _, draws = day_draws(de)  # paired: every cell and segment uses these same drawn days
    tot = lambda c: np.bincount(inv, weights=c, minlength=n_days)[draws].sum(axis=1)
    key = client_key(df)
    ret = {d: returning(dt, key, d)[E] for d in DELAYS}

    out, D, seeds_sav = {}, {}, {}  # D: bootstrap totals per (model, d, policy, segment)
    for (m, d, p), s in S.items():
        C = np.array([contribution(q, ye, ae) for q in s])
        o = [outcome(ye, ae, flags("raw", q, ae, None, REF_C), REF_C, n_days) for q in s]
        assert np.allclose(C.sum(axis=1), [x["savings"] for x in o]), "contribution sum != business.outcome"
        cm = C.mean(axis=0)
        segs = {"all": np.ones(len(E), bool), "returning": ret[d], "new": ~ret[d]}
        for g, mask in segs.items():
            D[(m, d, p, g)] = tot(np.where(mask, cm, 0.0))
            seeds_sav[(m, d, p, g)] = C[:, mask].sum(axis=1)
        out[(m, d, p)] = {
            "savings": _stats(seeds_sav[(m, d, p, "all")]), "savings_95": ci(D[(m, d, p, "all")]),
            **{k: _stats([x[k] for x in o]) for k in ("dollar_recall", "flags_per_day", "precision")},
            "roc_auc": _stats([roc_auc_score(ye, q) for q in s]), "pr_auc": _stats([average_precision_score(ye, q) for q in s]),
            "segments": {g: {"savings": _stats(seeds_sav[(m, d, p, g)]), "savings_95": ci(D[(m, d, p, g)])}
                         for g in ("returning", "new")},
            "retrain_days": schedule(p), "n_retrains_after_r0": len(schedule(p)) - 1,
            "train_rows": [int(models[active[(d, p, b)]]["mask"].sum()) for b in BLOCKS],
        }

    # Sanity ordering (a check, not a verdict): savings must not rise as the delay grows.
    bad = []
    for m in MODELS:
        for p in EVERY:
            ds = [d for d in DELAYS if (m, d, p) in out]
            for a, b in zip(ds, ds[1:]):
                rng = ci(D[(m, b, p, "all")] - D[(m, a, p, "all")])
                if rng[0] > 0:
                    bad.append(f"{m} {p}: D={b} saves more than D={a} (95% {rng[0]:,.0f} to {rng[1]:,.0f})")
    if bad:
        sys.exit("sanity ordering violated; investigate before writing any result:\n  " + "\n  ".join(bad))

    def gain(m, d, p, g="all"):
        seeds = seeds_sav[(m, d, p, g)] - seeds_sav[(m, d, "never", g)]
        return seeds, D[(m, d, p, g)] - D[(m, d, "never", g)]

    for (m, d, p), c in out.items():
        if p == "never":
            continue
        n = c["n_retrains_after_r0"]
        sd, dr = gain(m, d, p)
        c["gain_vs_never"] = {"mean": float(sd.mean()), "seeds": _stats(sd), "range_95": ci(dr),
                              "verdict": classify(ci(dr), "retraining pays", "retraining costs money")}
        c["extra_per_retrain"] = {"mean": float(sd.mean()) / n, "range_95": [v / n for v in ci(dr)],
                                  "note": "the cost of a retrain (compute, engineering, checking) is not priced"}
        c["gain_by_segment"] = {g: {"mean": float(gain(m, d, p, g)[0].mean()), "range_95": ci(gain(m, d, p, g)[1])}
                                for g in ("returning", "new")}

    pred = {"a": {}, "b": {}}
    for m in MODELS:
        g = out[(m, HEADLINE_D, "4-weekly")]["gain_vs_never"]
        pred["a"][m] = {"gain_mean": g["mean"], "range_95": g["range_95"], "pays_verdict": g["verdict"],
                        "score": score_a(g["mean"], g["range_95"])}
        (s60, d60), (s7, d7) = gain(m, 60, "4-weekly"), gain(m, 7, "4-weekly")
        rng = ci(d60 - d7)
        pred["b"][m] = {"diff_mean": float((s60 - s7).mean()), "range_95": rng,
                        "class": classify(rng, "more", "less")}
    sc = [pred["a"][m]["score"] for m in MODELS]
    pred["a"]["overall"] = ("confirmed" if all(x == "confirmed" for x in sc) else
                            "wrong" if "wrong" in sc else "not confirmed either way")
    (sn, dn), (sl, dl) = gain("nn", HEADLINE_D, "4-weekly"), gain("lgbm", HEADLINE_D, "4-weekly")
    rng = ci(dn - dl)
    pred["c"] = {"diff_net_minus_lgbm_mean": float((sn - sl).mean()), "range_95": rng,
                 "class": classify(rng, "neural net", "LightGBM")}
    for k in pred:
        pred[k]["predicted"] = PREDICTED[k]

    res = {"status": STATUS, "r0": R0, "blocks": list(BLOCKS), "delays": list(DELAYS), "headline_d": HEADLINE_D,
           "c": REF_C, "eval_rows": len(E), "eval_days": n_days, "eval_fraud_dollars": float(ae[ye == 1].sum()),
           "boot": {"draws": len(draws), "unit": "whole days", "paired": True},
           "distinct_training_sets": len(models),
           "recipe": {m: {"counts": f["counts"], "recipe_check_max_diff": f["recipe_check_max_diff"]} for m, f in fits.items()},
           "segments_share": {str(d): {"returning_rows": float(ret[d].mean()),
                                       "missing_key_rows": float(key.iloc[E].isna().mean())} for d in DELAYS},
           "cells": {m: {str(d): {p: out[(m, d, p)] for p in EVERY if (m, d, p) in out} for d in DELAYS} for m in MODELS},
           "sanity_ordering": "passed", "predictions": pred, "model_hashes": model_hashes()}
    _write("phase8_retrain.json", res)
    chart(res)
    print_report(res)


def print_report(res):
    print(f"\n{res['status']}\nsavings at C=${res['c']} over {res['eval_days']} days (fraud dollars "
          f"${res['eval_fraud_dollars']:,.0f}); mean +- std over 5 seeds; gain = vs never, 95% day-bootstrap range")
    for m in MODELS:
        print(f"\n{NAMES[m]}")
        for d in DELAYS:
            for p, c in res["cells"][m][str(d)].items():
                g = c.get("gain_vs_never")
                extra = (f"  gain {g['mean']:+10,.0f} ({g['range_95'][0]:+,.0f} to {g['range_95'][1]:+,.0f}) {g['verdict']}"
                         f"  per retrain {c['extra_per_retrain']['mean']:+,.0f}") if g else ""
                print(f"  D={d:<3}{p:9} ${c['savings']['mean']:>9,.0f} +- {c['savings']['std']:>6,.0f}  ROC "
                      f"{c['roc_auc']['mean']:.4f}  flags/day {c['flags_per_day']['mean']:.0f}{extra}")
    pr = res["predictions"]
    print(f"\n(a) predicted '{pr['a']['predicted']}': " + ", ".join(
        f"{NAMES[m]} {pr['a'][m]['score']} (gain {pr['a'][m]['gain_mean']:+,.0f}, {pr['a'][m]['pays_verdict']})"
        for m in MODELS) + f"; overall {pr['a']['overall']}")
    print(f"(b) predicted '{pr['b']['predicted']}': " + ", ".join(f"{NAMES[m]} {pr['b'][m]['class']}" for m in MODELS))
    print(f"(c) predicted '{pr['c']['predicted']}': {pr['c']['class']} (net - LightGBM gain "
          f"{pr['c']['range_95'][0]:+,.0f} to {pr['c']['range_95'][1]:+,.0f})")
    print("The cost of a retrain is not priced.")


DELAY_COLOR = {0: "#8a8984", 7: "#86b6ef", 30: "#2a78d6", 60: "#104281"}  # ordinal blue ramp (validated); 0 = grey


def chart(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pol = list(EVERY)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, facecolor=SURFACE)
    for ax, m in zip(axes, MODELS):
        _style(ax)
        ax.axhline(0, color=INK2, lw=1)
        for d in DELAYS:
            cs = res["cells"][m][str(d)]
            xs = [i for i, p in enumerate(pol) if p in cs]
            g = [cs[pol[i]].get("gain_vs_never", {"mean": 0, "range_95": [0, 0]}) for i in xs]
            mid = np.array([v["mean"] for v in g]) / 1e3
            err = np.array([[v["mean"] - v["range_95"][0], v["range_95"][1] - v["mean"]] for v in g]).T / 1e3
            label = "D = 0 (upper bound)" if d == 0 else f"D = {d} days"
            ax.errorbar(xs, mid, yerr=err, color=DELAY_COLOR[d], lw=2, marker="o", ms=6, capsize=3,
                        ls="--" if d == 0 else "-", label=label, zorder=3)
            ax.annotate(f"D={d}", (xs[-1], mid[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                        color=INK2, fontsize=8)
        ax.set_xticks(range(len(pol)), pol)
        ax.set_xlim(-0.3, len(pol) - 0.5)
        ax.set_title(NAMES[m], color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel(f"extra savings vs never retraining ($k)\nC = ${res['c']}, 12 weeks, mean of 5 seeds",
                       color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    fig.suptitle("Retraining with delayed labels: extra dollars by policy (bars = 95% day-bootstrap range; "
                 "retrain cost not priced)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    REPORTS.mkdir(exist_ok=True)
    fig.savefig(REPORTS / "retrain_savings.png", dpi=130, facecolor=SURFACE)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["fit"] and args[1:2] and args[1] in MODELS:
        fit(args[1])
    elif args == ["report"]:
        report()
    else:
        sys.exit(__doc__)
