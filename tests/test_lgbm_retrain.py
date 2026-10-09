"""Phase 8 checks: the label-availability rule, the policy clock, the money maths, and LightGBM leakage.
Named to sort before test_nn: LightGBM must train before PyTorch starts in this process."""
import numpy as np
import pandas as pd
import pytest

import src.business as biz
import src.retrain as rt
from src.lgbm import feature_names
from src.split import DAY, DT_START

SMALL = {"n_estimators": 20}  # test cap: changes how much a model learns, not which rows it sees


@pytest.fixture(scope="module")
def grid(df):
    return rt.plan(df["TransactionDT"].to_numpy())


def key_of(grid, d, r):
    return next(k for k, g in grid[0].items() if (d, r) in g["uses"])


def lgbm(df, g, ev, seed=1):
    cfg, counts = rt.recipe("lgbm")
    return rt.lgbm_scores(df, feature_names(df), g["mask"], ev, [seed], counts, cfg, **SMALL)


def flipped(df, rows):
    return df.assign(isFraud=np.where(rows, 1 - df["isFraud"], df["isFraud"]))


# ---------- (a) the rule itself ----------

def test_no_model_trains_on_a_row_whose_label_is_not_back(df, grid):
    """For every model in the grid, recomputed here with its own formula (not train_mask)."""
    dt = df["TransactionDT"].to_numpy()
    models, _ = grid
    assert len(models) == 25  # the training sets the compute estimate in DECISIONS.md counted
    for g in models.values():
        used = dt[g["mask"]]
        assert len(used)
        for d, r in g["uses"]:
            R = DT_START + r * DAY
            assert (used + d * DAY <= R).all() and (used < R).all()
            assert len(used) == ((dt + d * DAY <= R) & (dt < R)).sum()  # and nothing else is dropped


def test_lightgbm_fit_sees_exactly_the_mask_rows(df, grid):
    """The winning config bags rows (subsample ~0.81), so tree 0 sees a subset. A throwaway no-bagging fit of the
    same mask must see every one of its rows, read from the model's own first tree."""
    g = grid[0][key_of(grid, 60, rt.R0)]
    cfg, _ = rt.recipe("lgbm")
    m = rt.fit_lgbm(df, feature_names(df), g["mask"], 1, 2, cfg, subsample=1.0)
    seen = int(m.booster_.trees_to_dataframe().query("tree_index == 0").iloc[0]["count"])
    assert seen == g["mask"].sum()


# ---------- (b) the key leakage test, (c) evaluation labels, (d) determinism ----------

def test_labels_not_yet_back_cannot_change_lgbm(df, grid):
    d, r = 60, rt.R0
    g = grid[0][key_of(grid, d, r)]
    dt = df["TransactionDT"].to_numpy()
    late = dt + d * DAY > DT_START + r * DAY  # own formula
    assert (late & (dt < DT_START + r * DAY)).any()  # rows whose features exist but whose label is still out
    ev = rt.block_start(biz.day_of(dt)) == g["blocks"][0]
    assert np.array_equal(lgbm(df, g, ev), lgbm(flipped(df, late), g, ev))


def test_evaluation_labels_never_enter_lgbm(df, grid):
    """The never-retrain model at D = 0 (tightest boundary) is scored on all 12 blocks: flip exactly those labels."""
    g = grid[0][grid[1][(0, "never", rt.R0)]]
    assert g["blocks"] == list(rt.BLOCKS)
    ev = np.isin(rt.block_start(biz.day_of(df["TransactionDT"])), g["blocks"])
    assert np.array_equal(lgbm(df, g, ev), lgbm(flipped(df, ev), g, ev))


def test_lgbm_same_seed_same_scores(df, grid):
    g = grid[0][key_of(grid, 60, rt.R0)]
    ev = rt.block_start(biz.day_of(df["TransactionDT"])) == rt.R0
    assert np.array_equal(lgbm(df, g, ev, seed=3), lgbm(df, g, ev, seed=3))


# ---------- (e) the policy clock ----------

def test_policy_picks_the_latest_retrain_on_or_before_each_block():
    assert rt.BLOCKS == tuple(range(98, 176, 7))
    want = {"never": [98] * 12,
            "4-weekly": [98] * 4 + [126] * 4 + [154] * 4,
            "2-weekly": [98, 98, 112, 112, 126, 126, 140, 140, 154, 154, 168, 168],
            "weekly": list(rt.BLOCKS)}
    for p, days in want.items():
        assert rt.schedule(p) == sorted(set(days))
        assert [rt.active_day(rt.schedule(p), b) for b in rt.BLOCKS] == days
    assert rt.active_day([10, 20], 19) == 10 and rt.active_day([10, 20], 20) == 20  # toy timeline
    with pytest.raises(ValueError):
        rt.active_day([10], 5)  # before the first deployment there is no model
    assert rt.block_start([97, 98, 104, 105, 181]).tolist() == [-1, 98, 98, 105, 175]


def test_every_block_has_exactly_its_policy_model(grid):
    models, active = grid
    assert {p for d, p in rt.cells() if d in (30, 60)} == {"never", "4-weekly", "2-weekly"}  # compute budget
    for d, p in rt.cells():
        for b in rt.BLOCKS:
            g = models[active[(d, p, b)]]
            assert b in g["blocks"] and (d, rt.active_day(rt.schedule(p), b)) in g["uses"]
        assert models[active[(d, "never", rt.R0)]]["blocks"] == list(rt.BLOCKS)


# ---------- money and client segments ----------

def test_contribution_sums_to_business_savings():
    y, amt, p = np.array([1, 1, 0, 0]), np.array([100.0, 5.0, 50.0, 1.0]), np.array([0.5, 0.5, 0.5, 0.0])
    c = rt.contribution(p, y, amt)  # p x amt = 50, 2.5, 25, 0 vs C = 10: rows 0 and 2 flagged
    assert c.tolist() == [90, 0, -10, 0]
    assert c.sum() == 80 == biz.outcome(y, amt, p * amt > 10, 10, 1)["savings"]


def test_returning_client_needs_a_label_that_has_arrived():
    day = np.array([50, 100, 95, 100, 100, 100, 60])
    toy = pd.DataFrame({"TransactionDT": DT_START + day * DAY + 100,
                        "card1": [1, 1, 2, 2, 3, 4, 5],
                        "addr1": [10.0, 10, 20, 20, 30, np.nan, 50],
                        "D1": [10.0, 60, 0, 5, 0, 0, 0]})  # rows 0/1: one client (start day 40); rows 2/3 too
    key = rt.client_key(toy)
    assert key[0] == key[1] and key[2] == key[3] and pd.isna(key[5])
    dt = toy["TransactionDT"].to_numpy()
    # D = 7, decision day 98: client 1's day-50 label is back; client 2's day-95 label is not; 3 is new; 4 no key
    assert rt.returning(dt, key, 7).tolist() == [False, True, False, False, False, False, False]
    assert rt.returning(dt, key, 60).tolist() == [False] * 7  # 50 + 60 > 98: not back yet
