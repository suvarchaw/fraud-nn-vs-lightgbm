"""Phase 2: LightGBM baseline. Fit on TRAIN, early-stop and score on VALIDATION. Test is never touched."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.split import split_by_time

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
EXCLUDED = {"isFraud", "TransactionID", "TransactionDT"}  # answer, row number, calendar position
# Stopping rule: Phase 3 (neural net) must reuse the same metric and patience.
STOP_METRIC = "auc"      # validation ROC-AUC
PATIENCE = 100           # stop after this many rounds without improvement; best round is kept
MAX_TREES = 2000
LEARNING_RATE = 0.05


def feature_names(df):
    return [c for c in df.columns if c not in EXCLUDED]


def train(df, seed=SEED, max_trees=MAX_TREES):
    """Fit on train rows only. Returns (model, features, train_index, val_x, val_y)."""
    tr, val, _ = split_by_time(df)  # test block is discarded, never named
    feats = feature_names(df)
    model = lgb.LGBMClassifier(
        n_estimators=max_trees, learning_rate=LEARNING_RATE, random_state=seed,
        deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1,  # fixed 4 threads: repeatable
    )
    model.fit(
        tr[feats], tr["isFraud"],
        eval_X=val[feats], eval_y=val["isFraud"], eval_metric=STOP_METRIC,
        callbacks=[lgb.early_stopping(PATIENCE, verbose=False)],
    )
    return model, feats, tr.index, val[feats], val["isFraud"]


def scores(model, val_x, val_y):
    p = model.predict_proba(val_x, num_iteration=model.best_iteration_)[:, 1]
    return {"roc_auc": roc_auc_score(val_y, p), "pr_auc": average_precision_score(val_y, p)}


def no_skill(val_y):
    """Same score for every row: ROC-AUC 0.5 exactly, PR-AUC = fraud rate."""
    p = np.zeros(len(val_y))
    return {"roc_auc": roc_auc_score(val_y, p), "pr_auc": average_precision_score(val_y, p)}


def top_importance(model, feats, n=10):
    gain = model.booster_.feature_importance("gain")
    share = 100 * gain / gain.sum()
    order = np.argsort(-share)[:n]
    return {feats[i]: round(float(share[i]), 2) for i in order}


if __name__ == "__main__":
    from src.data import load_raw

    df = load_raw()
    model, feats, tr_idx, val_x, val_y = train(df)
    m, ns = scores(model, val_x, val_y), no_skill(val_y)
    imp = top_importance(model, feats)
    out = {
        "seed": SEED, "n_features": len(feats), "n_train": len(tr_idx), "n_val": len(val_y),
        "val_fraud_rate": float(val_y.mean()), "best_iteration": int(model.best_iteration_),
        "stop_metric": STOP_METRIC, "patience": PATIENCE,
        "model": m, "no_skill": ns, "top10_gain_share_%": imp,
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "metrics").mkdir(exist_ok=True)
    model.booster_.save_model(str(ROOT / "models" / f"lgbm_seed{SEED}.txt"))
    (ROOT / "metrics" / "phase2_lgbm.json").write_text(json.dumps(out, indent=2))
    print(f"best_iteration={out['best_iteration']}  features={len(feats)}")
    print(f"{'':10}{'ROC-AUC':>10}{'PR-AUC':>10}")
    print(f"{'model':10}{m['roc_auc']:10.4f}{m['pr_auc']:10.4f}")
    print(f"{'no-skill':10}{ns['roc_auc']:10.4f}{ns['pr_auc']:10.4f}")
    print("top 10 features by gain share (%):")
    for k, v in imp.items():
        print(f"  {k:20}{v:6.2f}")
