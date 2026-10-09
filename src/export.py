"""Phase 7: write the serving artifact (model + spec) for the ONE frozen model.

The model is Phase 5 LightGBM seed 1 with its Phase 5 calibrator, copied unchanged. Nothing is trained, refitted or
scored here, and the test block is never read. Feature list and category levels are taken from the code and the
model file itself, never retyped.

    python -m src.export [out_dir]      default out_dir: models/service/
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import lightgbm as lgb

from src.lgbm import feature_names
from src.split import BOUNDS, DT_START

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "models" / "phase5"
SEED = 1  # fixed by the rule "the first Phase 5 seed", not by looking at scores
VERSION = f"lgbm-phase5-seed{SEED}-calibrated"


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_json(obj):  # same recipe as src.business.sha (not imported: that module pulls in the neural-net code path)
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def build(df, out_dir=None):
    out = Path(out_dir) if out_dir else ROOT / "models" / "service"
    out.mkdir(parents=True, exist_ok=True)

    frozen = json.loads((SRC / "frozen.json").read_text())
    val_meta = json.loads((ROOT / "metrics" / "phase5_val.json").read_text())
    assert sha_json(frozen) == val_meta["frozen_sha256"], "frozen.json is not the one Phase 5 recorded"
    cal = frozen["lgbm"]["seeds"][SEED - 1]["calibrator"]
    assert cal["kind"] == "isotonic" and frozen["lgbm"]["method"] == "isotonic"

    model_file = SRC / f"lgbm_seed{SEED}.txt"
    booster = lgb.Booster(model_file=str(model_file))
    feats = feature_names(df)
    assert booster.feature_name() == feats, "model feature order differs from the training code's feature list"

    cat_cols = [c for c in feats if str(df[c].dtype) == "category"]
    levels = dict(zip(cat_cols, booster.pandas_categorical))
    assert len(cat_cols) == len(booster.pandas_categorical)
    for c in cat_cols:
        assert levels[c] == list(df[c].cat.categories), f"category levels differ for {c}"

    test = json.loads((ROOT / "metrics" / "phase5_test.json").read_text())
    h, auc = test["headline"], test["auc"]["lgbm"]
    spec = {
        "version": VERSION,
        "model_sha256": sha_file(model_file),
        "calibrator_sha256": sha_json(cal),
        "n_features": len(feats),
        "features": [{"name": c, "kind": "category" if c in levels else "number",
                      "levels": levels.get(c)} for c in feats],
        "calibrator": cal,
        "training_period": {"first_day": 0, "last_train_day": (BOUNDS["train_end"] - DT_START) / 86400,
                            "note": "days counted from the first payment in the data; train ends before the 7-day gap"},
        "test_metrics": {
            "source": "metrics/phase5_test.json (one locked run, 30 days, 5 seeds; the served model is seed 1)",
            "lgbm_roc_auc_mean_over_5_seeds": auc["roc_auc"]["mean"],
            "lgbm_pr_auc_mean_over_5_seeds": auc["pr_auc"]["mean"],
            "headline": {k: h[k] for k in ("c", "savings", "savings_pct", "dollar_recall", "recall", "precision",
                                           "flags_per_day", "fraud_dollars")},
        },
        "known_limits": [
            "The review cost C is an assumed number ($10 by default), not measured; every money figure depends on it.",
            "Part of the model's edge is card memory: it recognises cards and addresses that had fraud labels in the "
            "training data. On clients it has not seen, its lead over the neural network was only about 0.006 ROC-AUC "
            "on validation (audit A1). Real chargebacks arrive weeks late, so the edge would be smaller still.",
            "Drift: the model is frozen. Scores fell 0.02-0.04 ROC-AUC after the training period and nothing here "
            "retrains or monitors it (Phase 6, audit A3).",
            "No label delay is modelled; the evaluation used final labels.",
            "Test metrics are from one 30-day window of one dataset.",
        ],
    }
    shutil.copyfile(model_file, out / "model.txt")
    assert sha_file(out / "model.txt") == spec["model_sha256"]
    (out / "spec.json").write_text(json.dumps(spec))
    return out


if __name__ == "__main__":
    from src.data import load_raw

    dest = build(load_raw(verbose=False), sys.argv[1] if len(sys.argv) > 1 else None)
    s = json.loads((dest / "spec.json").read_text())
    print(f"wrote {dest}  version {s['version']}  features {s['n_features']}  model sha256 {s['model_sha256']}")
