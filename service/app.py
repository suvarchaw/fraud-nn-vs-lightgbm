"""Phase 7: fraud scoring API for ONE frozen model (LightGBM seed 1 + its Phase 5 calibrator).

Serves exactly what Phase 5 evaluated: no refit, no ensemble. The feature list, column order and category levels
come from spec.json (written by `python -m src.export`), never from code in this file. Imports no PyTorch and nothing
from the training code. Request payloads and feature values are never logged.

    MODEL_DIR=models/service uvicorn service.app:create_app --factory
"""
import hashlib
import itertools
import json
import logging
import os
import time
from pathlib import Path
from typing import Annotated

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

MAX_BATCH = 500                # rows per /score_batch call: a hard limit
MAX_BODY_BYTES = 8_000_000     # largest accepted request body (a full 500-row batch is about 4 MB)
MAX_LABEL_CHARS = 100          # longest accepted category label
MAX_AMOUNT = 10_000_000
DEFAULT_REVIEW_COST = 10.0     # dollars; an ASSUMPTION, see REVIEW_COST_NOTE
REVIEW_COST_NOTE = ("The review cost C is an assumed number, not a measured one. A transaction is flagged when "
                    "calibrated probability x amount > C; change C to match your real cost of one review.")
log = logging.getLogger("service")


def sha_json(obj):  # same recipe as src.export.sha_json (a test checks the two agree)
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def calibrate(p, cal):
    """Isotonic step function, exactly as src.business.calibrate (a test checks the two agree)."""
    return np.interp(p, cal["x"], cal["y"])


def decide(p, amount, c):
    """Expected loss = p x amount; flag when it exceeds the review cost C."""
    loss = p * amount
    return loss, loss > c


def request_model(spec):
    """One optional field per model feature (missing = NaN, as in training); unknown fields are rejected."""
    fields = {}
    for f in spec["features"]:
        if f["name"] == "TransactionAmt":  # the amount is a model feature AND the amount in p x amount
            fields[f["name"]] = (Annotated[float, Field(strict=True, gt=0, le=MAX_AMOUNT, allow_inf_nan=False,
                                                        description="Transaction amount (required).")], ...)
        elif f["kind"] == "category":
            fields[f["name"]] = (Annotated[str, Field(strict=True, max_length=MAX_LABEL_CHARS)] | None, None)
        else:
            fields[f["name"]] = (Annotated[float, Field(strict=True, allow_inf_nan=False)] | None, None)
    return create_model("Transaction", __config__=ConfigDict(extra="forbid"), **fields)


def build_frame(spec, rows):
    """The one place a feature table is made: spec column order, float32 numbers (as in training), category columns
    with the training levels (an unseen label becomes missing, exactly as LightGBM treated it in training)."""
    cols = {}
    for f in spec["features"]:
        vals = [getattr(r, f["name"]) for r in rows]
        if f["kind"] == "category":
            known = f.setdefault("_known", set(f["levels"]))
            cols[f["name"]] = pd.Categorical([v if v in known else None for v in vals], categories=f["levels"])
        else:
            cols[f["name"]] = np.array([np.nan if v is None else v for v in vals], dtype="float32")
    return pd.DataFrame(cols)


class Result(BaseModel):
    p: float = Field(description="Calibrated fraud probability.")
    expected_loss: float = Field(description="p x amount, in dollars.")
    flag: bool = Field(description="True when expected_loss > review_cost_used.")


class Meta(BaseModel):
    review_cost_used: float = Field(description=REVIEW_COST_NOTE)
    model_version: str
    model_sha256: str


class ScoreOut(Result, Meta):
    pass


class BatchOut(Meta):
    results: list[Result]


def create_app(model_dir=None):
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    d = Path(model_dir or os.environ.get("MODEL_DIR", "models/service"))
    spec = json.loads((d / "spec.json").read_text())
    if hashlib.sha256((d / "model.txt").read_bytes()).hexdigest() != spec["model_sha256"]:
        raise RuntimeError("model.txt does not match the sha256 recorded in spec.json; refusing to start")
    if sha_json(spec["calibrator"]) != spec["calibrator_sha256"]:
        raise RuntimeError("calibrator does not match the sha256 recorded in spec.json; refusing to start")
    booster = lgb.Booster(model_file=str(d / "model.txt"))
    assert booster.feature_name() == [f["name"] for f in spec["features"]]
    default_c = float(os.environ.get("REVIEW_COST", DEFAULT_REVIEW_COST))
    if not (0 < default_c < 1e9):
        raise RuntimeError("REVIEW_COST must be a positive number")
    Transaction = request_model(spec)
    counter = itertools.count(1)

    app = FastAPI(title="Fraud scoring (frozen LightGBM, Phase 5 seed 1)", version=spec["version"],
                  description="Scores card payments with one frozen model. " + REVIEW_COST_NOTE)

    def score(rows, c):
        raw = booster.predict(build_frame(spec, rows), num_threads=1)  # per-row maths: thread count cannot change it
        p = calibrate(raw, spec["calibrator"])
        amt = np.array([r.TransactionAmt for r in rows], dtype="float64")
        loss, flag = decide(p, amt, c)
        return [Result(p=float(a), expected_loss=float(b), flag=bool(f)) for a, b, f in zip(p, loss, flag)]

    meta = lambda c: dict(review_cost_used=c, model_version=spec["version"], model_sha256=spec["model_sha256"])

    @app.middleware("http")
    async def guard_and_log(request: Request, call_next):
        t0, n = time.perf_counter(), next(counter)
        if request.method == "POST":
            length = request.headers.get("content-length")
            if length is None or not length.isdigit():
                resp = JSONResponse({"detail": "Content-Length required"}, 411)
            elif int(length) > MAX_BODY_BYTES:
                resp = JSONResponse({"detail": f"body larger than {MAX_BODY_BYTES} bytes"}, 413)
            else:
                resp = await call_next(request)
        else:
            resp = await call_next(request)
        # only: request number, route, status, latency, score. Never the body, query string, headers or client address.
        log.info("req=%d %s %s status=%d latency_ms=%.1f%s", n, request.method, request.url.path, resp.status_code,
                 1000 * (time.perf_counter() - t0), getattr(request.state, "log_score", ""))
        return resp

    @app.exception_handler(RequestValidationError)
    async def invalid(_, exc):  # field path and error type only: never echo what the client sent
        return JSONResponse({"detail": [{"loc": list(e["loc"]), "type": e["type"]} for e in exc.errors()]}, 422)

    def cost(c):
        return default_c if c is None else c

    CostQ = Annotated[float | None, Query(gt=0, lt=1e9, allow_inf_nan=False, description=REVIEW_COST_NOTE)]

    @app.post("/score", response_model=ScoreOut)
    async def score_one(tx: Transaction, request: Request, review_cost: CostQ = None):
        c = cost(review_cost)
        (r,) = await run_in_threadpool(score, [tx], c)
        request.state.log_score = f" score={r.p:.6f}"
        return {**r.model_dump(), **meta(c)}

    @app.post("/score_batch", response_model=BatchOut)
    async def score_many(txs: list[Transaction], request: Request, review_cost: CostQ = None):
        if len(txs) > MAX_BATCH:
            raise HTTPException(413, f"at most {MAX_BATCH} transactions per call")
        if not txs:
            raise HTTPException(422, "empty batch")
        c = cost(review_cost)
        rs = await run_in_threadpool(score, txs, c)
        request.state.log_score = f" n={len(rs)} max_score={max(r.p for r in rs):.6f}"
        return {"results": rs, **meta(c)}

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": True}

    @app.get("/model-info")
    def model_info():
        return {"model_version": spec["version"], "model_sha256": spec["model_sha256"],
                "calibrator_sha256": spec["calibrator_sha256"], "n_features": spec["n_features"],
                "training_period": spec["training_period"], "test_metrics": spec["test_metrics"],
                "known_limits": spec["known_limits"], "review_cost_default": default_c,
                "review_cost_note": REVIEW_COST_NOTE, "max_batch": MAX_BATCH}

    return app
