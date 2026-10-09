"""Phase 8 checks for the net: the key leakage test, evaluation labels, determinism. Runs after test_nn, on one
thread (LightGBM has already trained in this process; see conftest.py)."""
import numpy as np
import pytest

import src.business as biz
import src.retrain as rt
from src.compare import SEEDS, nn_knobs
from src.lgbm import feature_names
from src.split import DAY, DT_START

pytestmark = pytest.mark.usefixtures("one_thread")
ONE = {s: 1 for s in SEEDS}  # test cap: one epoch changes how much a net learns, not which rows it sees


@pytest.fixture(scope="module")
def grid(df):
    return rt.plan(df["TransactionDT"].to_numpy())


def net(df, g, ev, seed=1):
    cfg, _ = rt.recipe("nn")
    return rt.nn_scores(df, feature_names(df), g["mask"], ev, [seed], ONE, nn_knobs(cfg))


def flipped(df, rows):
    return df.assign(isFraud=np.where(rows, 1 - df["isFraud"], df["isFraud"]))


def smallest(grid):
    return grid[0][grid[1][(60, "never", rt.R0)]]


def test_labels_not_yet_back_cannot_change_the_net(df, grid):
    g = smallest(grid)
    dt = df["TransactionDT"].to_numpy()
    late = dt + 60 * DAY > DT_START + rt.R0 * DAY  # own formula, not train_mask
    ev = rt.block_start(biz.day_of(dt)) == rt.R0
    assert np.array_equal(net(df, g, ev), net(flipped(df, late), g, ev))


def test_evaluation_labels_never_enter_the_net(df, grid):
    g = grid[0][grid[1][(0, "never", rt.R0)]]
    blk = rt.block_start(biz.day_of(df["TransactionDT"]))
    ev = np.isin(blk, g["blocks"])  # all 12 blocks: flip their labels, score the first one
    first = blk == rt.R0
    assert np.array_equal(net(df, g, first), net(flipped(df, ev), g, first))


def test_net_same_seed_same_scores(df, grid):
    g = smallest(grid)
    ev = rt.block_start(biz.day_of(df["TransactionDT"])) == rt.R0
    assert np.array_equal(net(df, g, ev, seed=3), net(df, g, ev, seed=3))
