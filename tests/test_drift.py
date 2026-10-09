"""Phase 6 checks. Nothing here trains a model."""
import numpy as np
import pandas as pd
import pytest

import src.drift as dr
from src.split import split_by_time


def test_psi_identical_is_zero_and_known_shift_is_known_value():
    e = np.array([0.5, 0.5])
    assert dr.psi(e, e) == 0
    # by hand: (0.25 - 0.5) ln(0.25 / 0.5) + (0.75 - 0.5) ln(0.75 / 0.5) = 0.1733 + 0.1014
    assert dr.psi(e, np.array([0.25, 0.75])) == pytest.approx(0.2747, abs=1e-4)


def test_numeric_bins_from_reference_deciles_missing_has_own_bin():
    ref = pd.Series(np.arange(100.0))
    assert dr.psi(*dr.shares(ref, ref)[:2]) == 0
    e, a, k = dr.shares(ref, pd.Series(np.r_[np.arange(50.0), [np.nan] * 50]))
    assert np.allclose(e[:-1], 0.1) and e[-1] == 0  # 10 equal decile bins, no missing in the reference
    assert a[-1] == 0.5 and a[:-1].sum() == 0.5     # half the compared rows are missing: their own bin
    assert k == 11


def test_unseen_category_counts_with_floored_share():
    cats = ["a", "b", "new"]
    ref = pd.Series(pd.Categorical(["a"] * 50 + ["b"] * 50, categories=cats))
    cmp = pd.Series(pd.Categorical(["a"] * 50 + ["new"] * 25 + [None] * 25, categories=cats))
    e, a, k = dr.shares(ref, cmp)
    assert e.tolist() == [0, 0.5, 0.5, 0] and a.tolist() == [0.25, 0.5, 0, 0.25]  # bins: missing, a, b, new
    f = dr.FLOOR
    by_hand = 2 * (0.25 - f) * np.log(0.25 / f) + (f - 0.5) * np.log(f / 0.5)
    assert dr.psi(e, a) == pytest.approx(by_hand)
    assert k == 4


def test_chance_level_separates_noise_from_a_real_shift():
    rng = np.random.default_rng(0)
    a, b, c = (pd.Series(rng.normal(mu, size=5000)) for mu in (0, 0, 0.3))
    e, s, k = dr.shares(a, b)
    assert dr.psi(e, s) < dr.chance_psi(k, 5000, 5000)
    e, s, k = dr.shares(a, c)
    assert dr.psi(e, s) > dr.chance_psi(k, 5000, 5000)


def test_every_row_in_one_week_blocks_match_the_split(df):
    wk, blk = dr.week_of(df["TransactionDT"]), dr.block_of(df["TransactionDT"])
    assert len(wk) == len(df) and wk.min() == 0 and wk.max() == 25
    assert np.bincount(wk).sum() == len(df)
    tr, val, test = split_by_time(df)
    for name, part in (("train", tr), ("val", val), ("test", test)):
        assert df.index[blk == name].equals(part.index)
    gaps = ~df.index.isin(tr.index.union(val.index).union(test.index))
    assert set(blk[gaps]) == {"gap1", "gap2"} and (np.isin(blk, ["gap1", "gap2"]) == gaps).all()
    # the weeks pre-registered in DECISIONS.md
    assert dr.weeks_made_of(wk, blk, ["val", "gap2", "test"]) == list(range(17, 26))
    assert dr.weeks_made_of(wk, blk, ["val"]) == [17, 18, 19]
    assert dr.weeks_made_of(wk, blk, ["test"]) == [22, 23, 24, 25]


def test_adversarial_uses_no_label_no_time_and_day_block_folds(df):
    a, b = dr.pairs(df)["train_vs_test"]
    X, t, d = dr.adversarial_xy(df, a, b)
    assert not {"isFraud", "TransactionDT", "TransactionID"} & set(X.columns)
    assert (t == np.r_[np.zeros(len(a)), np.ones(len(b))]).all()      # target = "is a test row"
    assert (t != pd.concat([a, b])["isFraud"].to_numpy()).any()        # ... and not the fraud label
    for period in (0, 1):
        days, f = d[t == period], dr.folds(d[t == period])
        assert set(f) == set(range(dr.N_FOLDS))
        spans = [(days[f == k].min(), days[f == k].max()) for k in range(dr.N_FOLDS)]
        assert all(spans[k][1] < spans[k + 1][0] for k in range(dr.N_FOLDS - 1))  # contiguous, no day in two folds


def test_scrambled_labels_leave_every_psi_unchanged(df):
    d = df.copy()
    d["isFraud"] = np.random.default_rng(0).permutation(d["isFraud"].to_numpy())
    assert dr.psi_table(d) == dr.psi_table(df)


def test_label_delay_flag_and_trend_rules():
    weeks = list(range(17, 26))
    steady = {w: 0.95 - 0.005 * (w - 17) for w in weeks}
    early_mean = np.mean([steady[w] for w in weeks[:-2]])
    assert not dr.label_delay_flag(True, (-0.007, -0.003), {w: v + 0.004 for w, v in steady.items()}, early_mean,
                                   [24, 25])
    late = {w: 0.93 if w < 24 else 0.90 for w in weeks}
    assert dr.label_delay_flag(True, (-0.001, 0.001), {w: v + 0.004 for w, v in late.items()}, 0.93, [24, 25])
    assert dr.label_delay_flag(False, (-0.004, 0.002), {w: v + 0.004 for w, v in late.items()}, 0.93, [24, 25])
    widths = [0.02] * 9
    assert dr.trend_label(-0.005, (-0.007, -0.003), list(steady.values()), widths, 9) == "falls steadily"
    assert dr.trend_label(0.0005, (-0.002, 0.003), [0.90 + 0.001 * (w % 2) for w in weeks], widths, 9) == "stays flat"
    assert dr.trend_label(0.0, (-0.004, 0.004), [0.90 + 0.05 * (w % 2) for w in weeks], widths, 9) == "bounces around"
