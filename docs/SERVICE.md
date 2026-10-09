# Fraud scoring service (Phase 7)

Serves ONE frozen model: the Phase 5 LightGBM seed-1 model with its Phase 5 isotonic calibrator, exactly as
evaluated. No refit, no ensemble, no PyTorch. The model comes from competition data and is not redistributable, so
it is never committed and the image is never pushed to a registry.

## Build the artifact, run it
```bash
.venv/bin/python -m src.export                       # writes models/service/ (model.txt + spec.json), scores nothing
MODEL_DIR=models/service .venv/bin/uvicorn service.app:create_app --factory --no-access-log
```
Docker (model mounted read-only, not baked in):
```bash
docker build -t fraud-score:local .
docker run --rm -p 8000:8000 -v "$PWD/models/service:/models:ro" fraud-score:local
```
`scripts/smoke_docker.sh` does build, run, `/health`, `/score`, stop. `scripts/latency.py` measures local latency.

## Endpoints
| | |
|---|---|
| `POST /score` | One transaction. Every model feature is optional (missing = NaN, as in training); `TransactionAmt` is required (it is both a feature and the amount in p x amount). Unknown fields, wrong types, NaN/inf, labels over 100 characters -> 422. |
| `POST /score_batch` | A JSON list of the same objects. At most 500 rows and 8 MB; over either -> 413. |
| `GET /health` | `{"status": "ok", "model_loaded": true}` |
| `GET /model-info` | Version, sha256, feature count, training period, headline test metrics, known limits. |

Both score endpoints take an optional `?review_cost=` (dollars). The default is `$10`, set by the `REVIEW_COST`
environment variable. **C is an assumed number, not a measured one**: a transaction is flagged when
`calibrated probability x amount > C`.

Example (invented values, not a dataset row):
```bash
curl -s -X POST localhost:8000/score -H 'content-type: application/json' \
  -d '{"TransactionAmt": 149.5, "ProductCD": "W", "card4": "visa", "card6": "debit", "C1": 2, "C2": 1}'
```
```json
{"p": 0.2031, "expected_loss": 30.37, "flag": true, "review_cost_used": 10.0,
 "model_version": "lgbm-phase5-seed1-calibrated", "model_sha256": "b5827c33...6f7ceb"}
```
Feature names are the dataset's column names (`GET /openapi.json` lists all 431). A label the model never saw in
training is treated as missing, exactly as LightGBM treated it.

## Privacy
Request payloads, query strings and feature values are never logged. One line per request:
request number, route, status, latency, score (batch: row count and highest score). Validation errors return the
field path and error type, never the submitted value.

## What this does and does not show
- The service reproduces the offline scores of the frozen model on validation rows to 1e-9 (a test). That proves no
  training/serving skew in the feature table; it does not show that real traffic looks like these columns.
- Test metrics in `/model-info` are copied from the one locked Phase 5 run (30 days, assumed C = $10), not recomputed.
- Known limits: assumed review cost; part of the edge is card memory (smaller on new clients, smaller again with real
  chargeback delays); the model is frozen and scores fell 0.02-0.04 ROC-AUC after the training period; no label delay.
