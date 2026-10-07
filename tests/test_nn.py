import numpy as np
import pandas as pd
import pytest
import torch

import src.compare as cmp
import src.lgbm
import src.nn as nnet
from src.split import split_by_time

NEW_LABEL = "never-seen.example"


pytestmark = pytest.mark.usefixtures("one_thread")  # fixture in conftest.py


def test_features_identical_to_lightgbm(df, prepared):
    prep = prepared[0]
    assert nnet.feature_names is src.lgbm.feature_names  # imported, not copied
    assert sorted(prep["num"] + prep["cat"]) == sorted(src.lgbm.feature_names(df))
    assert len(prep["num"]) + len(prep["cat"]) == len(src.lgbm.feature_names(df))


def test_statistics_come_from_train_rows_only(df, prepared):
    """Recompute every statistic here, independently with pandas, from train rows only."""
    prep = prepared[0]
    tr, _, _ = split_by_time(df)
    for c in prep["num"]:
        s = tr[c].astype("float64")
        s = np.sign(s) * np.log1p(s.abs())
        med = s.median()
        filled = s.fillna(med)
        std = filled.std(ddof=0)
        assert np.isclose(prep["median"][c], med, rtol=1e-6, atol=1e-9), c
        assert np.isclose(prep["mean"][c], filled.mean(), rtol=1e-6, atol=1e-9), c
        assert np.isclose(prep["std"][c], std if std > 1e-6 else 1.0, rtol=1e-6, atol=1e-9), c
    assert prep["flag"] == [c for c in prep["num"] if tr[c].isna().any()]
    for c in prep["cat"]:
        counts = tr[c].value_counts()
        assert prep["vocab"][c] == sorted(str(k) for k, n in counts.items() if n >= nnet.MIN_COUNT), c


def test_changing_validation_does_not_change_statistics(df, prepared):
    prep = prepared[0]
    _, val, _ = split_by_time(df)
    d = df.copy()
    for c in prep["num"]:
        d.loc[val.index, c] = d.loc[val.index, c] * 10
    for c in prep["cat"][:5]:
        d[c] = d[c].cat.add_categories([NEW_LABEL])
        d.loc[val.index, c] = NEW_LABEL
    assert nnet.prepare(d)[0] == prep


def test_unseen_category_maps_to_unknown(df, prepared):
    prep = prepared[0]
    _, val, _ = split_by_time(df)
    c = "P_emaildomain"
    v = val.iloc[:100].copy()
    v[c] = v[c].cat.add_categories([NEW_LABEL])
    v.loc[v.index[:10], c] = NEW_LABEL
    xn, xc = nnet.transform(v, prep)
    assert (xc[:10, prep["cat"].index(c)] == nnet.UNKNOWN).all()
    p = nnet.predict(nnet.Net(xn.shape[1], nnet.slots(prep)), xn, xc)
    assert np.isfinite(p).all()


def test_no_nan_or_inf_reaches_the_network(df, prepared):
    prep, (xn, xc, _), (vn, vc, _) = prepared
    for x in (xn, vn):
        assert torch.isfinite(x).all()
    for codes in (xc, vc):
        assert (codes >= 0).all() and (codes < torch.tensor(nnet.slots(prep))).all()
    _, val, _ = split_by_time(df)
    v = val.iloc[:100].copy()
    v.loc[v.index[:4], "TransactionAmt"] = [np.nan, np.inf, -np.inf, 1e30]
    v.loc[v.index[:4], "D1"] = [np.inf, np.nan, 1e30, -np.inf]
    assert torch.isfinite(nnet.transform(v, prep)[0]).all()


def test_same_seed_same_score(df):
    a = nnet.train(df, seed=7, max_epochs=1, n_train=20_000, verbose=False)[2]["val"]
    b = nnet.train(df, seed=7, max_epochs=1, n_train=20_000, verbose=False)[2]["val"]
    assert a == b


def test_sees_same_rows_as_lightgbm(df, prepared):
    """Twin of test_lgbm's row check: both models get exactly split_by_time's train and validation rows, in order."""
    tr, val, _ = split_by_time(df)
    prep, (_, _, y), (vn, vc, vy) = prepared
    assert np.array_equal(y.numpy(), tr["isFraud"].to_numpy("float32"))
    assert np.array_equal(vy.numpy(), val["isFraud"].to_numpy("float32"))
    xn, xc = nnet.transform(val, prep)
    assert torch.equal(xn, vn) and torch.equal(xc, vc)


def test_phase4_trial_never_touches_test_rows(df, scrambled):
    cfg, caps = cmp.sample_configs("nn")[1], {"max_epochs": 1, "n_train": 20_000}
    assert cmp.run("nn", df, cfg, 7, **caps)["val"] == cmp.run("nn", scrambled, cfg, 7, **caps)["val"]
