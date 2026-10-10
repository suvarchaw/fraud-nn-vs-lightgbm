#!/usr/bin/env bash
# Recording-friendly demo of the scoring service: start, /health, one /score, /model-info, stop.
# Needs Docker running, the image fraud-score:local (built here if missing) and models/service/ from `python -m src.export`.
# The payload is invented (same as docs/SERVICE.md), not a dataset row. Set PAUSE=0 to skip the pauses.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PATH:/Applications/Docker.app/Contents/Resources/bin"
IMAGE=fraud-score:local
NAME=fraud-score-demo
URL=http://127.0.0.1:8000
PAUSE=${PAUSE:-2}

trap 'docker stop "$NAME" >/dev/null 2>&1 || true' EXIT
step() { echo; echo "=== $1"; sleep "$PAUSE"; }
pretty() { python3 -m json.tool; }

docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" .

step "1. Start the container (model folder mounted read-only)"
docker run -d --rm --name "$NAME" -p 8000:8000 -v "$PWD/models/service:/models:ro" "$IMAGE" >/dev/null
echo "container $NAME started"

step "2. Wait until /health says ok"
for _ in $(seq 1 60); do
  curl -sf "$URL/health" >/dev/null && break
  sleep 1
done
curl -s "$URL/health" | pretty

step "3. Score one invented payment (POST /score)"
curl -s -X POST "$URL/score" -H 'content-type: application/json' \
  -d '{"TransactionAmt": 149.5, "ProductCD": "W", "card4": "visa", "card6": "debit", "C1": 2, "C2": 1}' | pretty

step "4. Show what model is loaded (GET /model-info)"
curl -s "$URL/model-info" | pretty

step "5. Stop and remove the container"
docker stop "$NAME" >/dev/null
echo "stopped and removed"
