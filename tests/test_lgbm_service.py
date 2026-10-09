"""Phase 7 checks: the service must give exactly the offline scores. Only VALIDATION rows are ever scored here.

The file name keeps it between test_lgbm and test_nn: pytest runs files alphabetically, and LightGBM must be used
before PyTorch in one process (see DECISIONS.md, Phase 4)."""
import json
import logging
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.business as biz
import src.export as export
from service import app as svc
from src.lgbm import feature_names
from src.split import split_by_time

ROOT = export.ROOT
JSON = {"content-type": "application/json"}


@pytest.fixture(scope="module")
def artifact(df, tmp_path_factory):
    return export.build(df, tmp_path_factory.mktemp("service"))


@pytest.fixture(scope="module")
def spec(artifact):
    return json.loads((artifact / "spec.json").read_text())


@pytest.fixture(scope="module")
def client(artifact):
    return TestClient(svc.create_app(artifact))


def post(client, path, payload, **kw):
    return client.post(path, content=json.dumps(payload), headers=JSON, **kw)


def as_payload(row):
    """What a client would send: missing values omitted, numbers as JSON numbers, labels as strings."""
    return {k: (v if isinstance(v, str) else float(v)) for k, v in row.items() if not pd.isna(v)}


def offline(rows, spec):
    """The training-side path: the Phase 5 model file and business.calibrate, on a table with training dtypes."""
    raw = lgb.Booster(model_file=str(export.SRC / "lgbm_seed1.txt")).predict(rows)
    return biz.calibrate(raw, spec["calibrator"])


# ---------- the key test ----------

def test_service_equals_offline_on_validation_rows(df, spec, client):
    _, val, _ = split_by_time(df)  # the test block is never named
    feats = feature_names(df)
    pick = np.sort(np.random.default_rng(0).choice(len(val), size=2000, replace=False))
    rows = val.iloc[pick]
    want = offline(rows[feats], spec)
    records = [as_payload(r) for r in rows[feats].to_dict("records")]
    got = []
    for i in range(0, len(records), svc.MAX_BATCH):
        r = post(client, "/score_batch", records[i:i + svc.MAX_BATCH])
        assert r.status_code == 200, r.text
        got += r.json()["results"]
    p = np.array([g["p"] for g in got])
    assert np.abs(p - want).max() < 1e-9
    amt = rows["TransactionAmt"].to_numpy("float64")
    assert np.abs(np.array([g["expected_loss"] for g in got]) - want * amt).max() < 1e-6
    assert [g["flag"] for g in got] == list(want * amt > 10)
    # the same rows one at a time
    for i in range(0, 2000, 40):
        r = post(client, "/score", records[i])
        assert abs(r.json()["p"] - want[i]) < 1e-9
    # and against the scores Phase 5 saved after training
    saved = np.load(export.SRC / "lgbm_val.npz")
    where = pd.Series(np.arange(len(saved["ids"])), index=saved["ids"])[rows["TransactionID"].to_numpy()]
    assert np.abs(biz.calibrate(saved["p"][0][where.to_numpy()], spec["calibrator"]) - p).max() < 1e-9
    assert (p != p[0]).any() and len(set(p)) > 30  # not a constant (isotonic has few steps)


def test_missing_and_unseen_values_match_training_treatment(df, spec, client):
    _, val, _ = split_by_time(df)
    feats = feature_names(df)
    row = val[feats].iloc[[5]].copy()
    cat = next(f["name"] for f in spec["features"] if f["kind"] == "category" and pd.notna(row[f["name"]].iloc[0]))
    # unseen label: offline, LightGBM sets it to missing, so the service must equal offline-with-missing
    sent = as_payload(row.iloc[0].to_dict()) | {cat: "never-seen.example"}
    got = post(client, "/score", sent).json()["p"]
    as_missing = row.copy()
    as_missing[cat] = pd.Categorical([None], categories=as_missing[cat].cat.categories)
    unseen = row.copy()
    unseen[cat] = unseen[cat].cat.add_categories(["never-seen.example"])
    unseen[cat] = pd.Categorical(["never-seen.example"], categories=unseen[cat].cat.categories)
    assert abs(got - offline(as_missing, spec)[0]) < 1e-9
    assert abs(got - offline(unseen, spec)[0]) < 1e-9
    # only the amount: every other field missing
    only = pd.DataFrame({c: pd.Categorical([None], dtype=val[c].dtype) if str(val[c].dtype) == "category"
                         else np.array([np.nan], dtype="float32") for c in feats})
    only["TransactionAmt"] = np.float32(25.5)
    r = post(client, "/score", {"TransactionAmt": 25.5}).json()
    assert abs(r["p"] - offline(only, spec)[0]) < 1e-9


# ---------- inputs ----------

def test_bad_inputs_are_rejected_without_echoing_them(spec, client):
    num = next(f["name"] for f in spec["features"] if f["kind"] == "number" and f["name"] != "TransactionAmt")
    cat = next(f["name"] for f in spec["features"] if f["kind"] == "category")
    marker = "zz-marker-not-a-number"
    for bad in ({"TransactionAmt": 10, num: marker}, {"TransactionAmt": 10, cat: 5}, {"TransactionAmt": 10, "nope": 1},
                {"TransactionAmt": "10"}, {"TransactionAmt": 0}, {"TransactionAmt": -3}, {"TransactionAmt": True},
                {"TransactionAmt": 1e12}, {num: 1.0}, {"TransactionAmt": 10, cat: "x" * 101}, [], 7):
        r = post(client, "/score", bad)
        assert r.status_code == 422, (bad, r.text)
        assert marker not in r.text and "xxxxxxxx" not in r.text
    for text in ('{"TransactionAmt": NaN}', '{"TransactionAmt": 10, "%s": Infinity}' % num, "{not json"):
        assert client.post("/score", content=text, headers=JSON).status_code == 422
    assert post(client, "/score_batch", []).status_code == 422
    assert post(client, "/score_batch", [{"TransactionAmt": 10}, {"TransactionAmt": "x"}]).status_code == 422
    assert post(client, "/score", {"TransactionAmt": 10}, params={"review_cost": 0}).status_code == 422
    assert post(client, "/score", {"TransactionAmt": 10}, params={"review_cost": "abc"}).status_code == 422


def test_size_limits(client):
    one = {"TransactionAmt": 10}
    assert post(client, "/score_batch", [one] * svc.MAX_BATCH).status_code == 200
    assert post(client, "/score_batch", [one] * (svc.MAX_BATCH + 1)).status_code == 413
    big = client.post("/score", content=b" " * (svc.MAX_BODY_BYTES + 1), headers=JSON)
    assert big.status_code == 413


def test_missing_fields_work_and_c_is_configurable(client):
    r = post(client, "/score", {"TransactionAmt": 100}).json()
    assert 0 <= r["p"] <= 1 and r["review_cost_used"] == 10 and r["expected_loss"] == pytest.approx(r["p"] * 100)
    assert post(client, "/score", {"TransactionAmt": 100}, params={"review_cost": 1e8}).json()["flag"] is False
    cheap = post(client, "/score", {"TransactionAmt": 100}, params={"review_cost": 1e-9}).json()
    assert cheap["flag"] is True and cheap["review_cost_used"] == 1e-9


def test_decision_rule_on_a_hand_made_table():
    p, amt = np.array([0.5, 0.5, 0.01, 0.2, 0.25]), np.array([30.0, 10.0, 2000.0, 40.0, 40.0])
    loss, flag = svc.decide(p, amt, 10)
    assert list(loss) == [15, 5, 20, 8, 10]
    assert list(flag) == [True, False, True, False, False]  # equal to C is not flagged
    assert list(svc.decide(p, amt, 4.99)[1]) == [True] * 5
    assert list(svc.decide(p, amt, 20)[1]) == [False] * 5


# ---------- skew guards ----------

def test_spec_comes_from_the_training_code_and_model(df, spec):
    assert [f["name"] for f in spec["features"]] == feature_names(df)
    booster = lgb.Booster(model_file=str(export.SRC / "lgbm_seed1.txt"))
    cats = [f["levels"] for f in spec["features"] if f["kind"] == "category"]
    assert cats == booster.pandas_categorical == [list(df[f["name"]].cat.categories) for f in spec["features"]
                                                   if f["kind"] == "category"]
    frozen = json.loads((export.SRC / "frozen.json").read_text())
    assert spec["calibrator"] == frozen["lgbm"]["seeds"][0]["calibrator"]
    assert export.sha_json(frozen) == biz.sha(frozen) == json.loads(
        (ROOT / "metrics" / "phase5_val.json").read_text())["frozen_sha256"]
    assert spec["n_features"] == len(feature_names(df)) == 431


def test_calibrator_copy_equals_business_calibrate(spec):
    grid = np.concatenate([[0, 1e-9, 1], np.linspace(0, 1, 2001), np.array(spec["calibrator"]["x"])])
    assert (svc.calibrate(grid, spec["calibrator"]) == biz.calibrate(grid, spec["calibrator"])).all()
    assert svc.sha_json({"a": 1}) == export.sha_json({"a": 1}) == biz.sha({"a": 1})


def test_service_refuses_a_changed_model(artifact, tmp_path):
    (tmp_path / "spec.json").write_text((artifact / "spec.json").read_text())
    (tmp_path / "model.txt").write_text((artifact / "model.txt").read_text() + "\n")
    with pytest.raises(RuntimeError, match="sha256"):
        svc.create_app(tmp_path)


def test_service_does_not_import_torch_or_training_code():
    code = ("import sys, service.app; bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'src')]; "
            "assert not bad, bad")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_test_block_is_never_named():
    for path in (ROOT / "service" / "app.py", ROOT / "src" / "export.py"):
        text = path.read_text()
        assert "split_by_time" not in text and "test_start" not in text and "phase5_test_lock" not in text


# ---------- privacy ----------

def test_logs_hold_no_payload_values(spec, client, caplog):
    cat = next(f["name"] for f in spec["features"] if f["kind"] == "category")
    with caplog.at_level(logging.DEBUG):
        post(client, "/score", {"TransactionAmt": 123.456789, cat: "zz-secret-marker"})
        post(client, "/score", {"TransactionAmt": 98.7654321}, params={"review_cost": 7.7777})
        post(client, "/score_batch", [{"TransactionAmt": 55.5555555}] * 3)
        post(client, "/score", {"TransactionAmt": "zz-secret-marker"})
    lines = [r.getMessage() for r in caplog.records if r.name == "service"]  # the test client's own log is not ours
    for secret in ("zz-secret-marker", "123.456789", "98.7654321", "55.5555555", "7.7777"):
        assert secret not in "\n".join(lines)
    assert len(lines) == 4 and all("status=" in s and "latency_ms=" in s for s in lines)
    assert "score=" in lines[0] and "max_score=" in lines[2] and "score=" not in lines[3]


def test_model_info_and_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    m = client.get("/model-info").json()
    assert m["n_features"] == 431 and len(m["model_sha256"]) == 64 and m["review_cost_default"] == 10
    assert m["test_metrics"]["headline"]["c"] == 10 and m["known_limits"] and "assumed" in m["review_cost_note"]
