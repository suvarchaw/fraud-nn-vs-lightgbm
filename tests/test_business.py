"""Phase 5 checks. Nothing here trains a model."""
import json

import numpy as np
import pytest

import src.business as biz
from src.split import split_by_time


def test_gap_and_test_rows_cannot_move_the_choices(df, scrambled):
    """Calibrators, thresholds and the top-1% cut-off must be identical when every gap and test row has scrambled
    features, amounts, labels AND scores. Validation scores are the same in both runs."""
    tr, val, _ = split_by_time(df)
    out = ~df.index.isin(tr.index.union(val.index))  # gap + test positions
    rng = np.random.default_rng(0)
    y = df["isFraud"].to_numpy()
    base = {m: np.clip(0.3 * y + 0.7 * rng.random((len(biz.SEEDS), len(df))), 0, 1) for m in biz.MODELS}
    a, b = {}, {}
    for m, s in base.items():
        a[m], b[m] = s.copy(), s.copy()
        a[m][:, out] = np.nan                      # what `choose` really passes: nothing outside validation
        b[m][:, out] = rng.random((len(biz.SEEDS), out.sum()))
    assert biz.freeze(df, a) == biz.freeze(scrambled, b)


def test_cost_maths_by_hand():
    y = np.array([1, 1, 0, 0, 0])
    amt = np.array([100.0, 5.0, 50.0, 1.0, 1.0])
    nothing = biz.outcome(y, amt, np.zeros(5, bool), 2, n_days=1)
    assert nothing["cost"] == 105 and nothing["savings"] == 0             # all fraud dollars lost
    everything = biz.outcome(y, amt, np.ones(5, bool), 2, n_days=1)
    assert everything["cost"] == 2 * 5 and everything["savings"] == 95   # C x rows
    one = biz.outcome(y, amt, np.array([1, 0, 0, 0, 0], bool), 2, n_days=1)
    assert one["cost"] == 2 + 5
    assert one["recall"] == 0.5 and one["dollar_recall"] == 100 / 105 and one["precision"] == 1


def test_calibrated_rule_is_expected_loss_over_c():
    p, amt = np.array([0.5, 0.5, 0.01, 0.2]), np.array([30.0, 10.0, 2000.0, 40.0])
    identity = {"calibrator": {"kind": "isotonic", "x": [0.0, 1.0], "y": [0.0, 1.0]}}
    assert biz.flags("calibrated", p, amt, identity, 10).tolist() == [True, False, True, False]  # 15, 5, 20, 8 vs 10


def test_best_threshold_by_hand():
    p = np.array([0.9, 0.8, 0.1])
    y = np.array([1, 0, 0])
    amt = np.array([100.0, 1.0, 1.0])
    assert biz.best_threshold(p, y, amt, 5) == 0.9              # flag only the fraud
    assert biz.best_threshold(p, y, amt, 500) == float("inf")   # a review costs more than any loss: flag nothing


def test_brier_and_ece():
    assert biz.brier(np.array([1, 0]), np.array([0.8, 0.4])) == pytest.approx((0.04 + 0.16) / 2)
    y = np.array([0] * 9 + [1] + [0] * 5 + [1] * 5)               # group 1: 10% fraud, group 2: 50%
    p = np.array([0.1] * 10 + [0.5] * 10)
    assert biz.ece(y, p, bins=2) == pytest.approx(0)             # percentages match reality exactly
    assert biz.ece(y, p * 2, bins=2) == pytest.approx(0.3)       # all doubled: gaps 0.1 and 0.5


def test_platt_and_isotonic_keep_the_order():
    rng = np.random.default_rng(1)
    p = rng.random(2000)
    y = (rng.random(2000) < p ** 2).astype(int)
    for kind in ("platt", "isotonic"):
        q = biz.calibrate(np.sort(p), biz.fit_calibrator(p, y, kind))
        assert (np.diff(q) >= 0).all(), kind


def test_bootstrap_draws_whole_days():
    days = np.repeat(np.arange(30), 7)           # 30 days, 7 payments each
    rows_of, draws = biz.day_draws(days, n=50)
    assert draws.shape == (50, 30)               # each draw has as many days as the block
    for dr in draws:
        per_day = np.bincount(days[biz.rows_in(rows_of, dr)], minlength=30)
        assert (per_day % 7 == 0).all()          # a day is in a draw 0, 1, 2... times, always with all its rows
        assert (per_day // 7 == np.bincount(dr, minlength=30)).all()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Lock, DECISIONS.md and frozen choices in a temporary folder, never the real files."""
    monkeypatch.setattr(biz, "LOCK", tmp_path / "lock.json")
    monkeypatch.setattr(biz, "DECISIONS", tmp_path / "DECISIONS.md")
    monkeypatch.setattr(biz, "OUT", tmp_path)
    (tmp_path / "frozen.json").write_text(json.dumps({"x": 1}))
    (tmp_path / "DECISIONS.md").write_text("# log\n")
    return tmp_path


def test_second_test_run_refused_unless_override_logged(sandbox):
    first = biz.start_test_run()
    with pytest.raises(SystemExit, match="refusing"):
        biz.start_test_run()
    second = biz.start_test_run(override="first run crashed before any result")
    assert second != first
    log = (sandbox / "DECISIONS.md").read_text()
    assert "OVERRIDE" in log and first in log and "crashed before any result" in log


def test_test_rows_scored_only_inside_the_locked_run(sandbox):
    with pytest.raises(SystemExit, match="refusing"):
        biz.score("lgbm", "test", "made-up-run-id")
    biz.start_test_run()
    with pytest.raises(SystemExit, match="refusing"):
        biz.score("lgbm", "test", "made-up-run-id")
