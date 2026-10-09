#!/usr/bin/env bash
# Manual smoke test (not part of pytest): build the image, run it with the model mounted, call /health and /score, stop.
# Needs `python -m src.export` to have written models/service/. Example payload values are invented.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PATH:/Applications/Docker.app/Contents/Resources/bin"
IMAGE=fraud-score:local
NAME=fraud-score-smoke
trap 'docker stop "$NAME" >/dev/null 2>&1 || true' EXIT

docker build -t "$IMAGE" .
docker images "$IMAGE" --format 'image size: {{.Size}}'
docker run -d --rm --name "$NAME" -p 8000:8000 -v "$PWD/models/service:/models:ro" "$IMAGE" >/dev/null
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  sleep 1
done
echo "--- GET /health"
curl -s http://127.0.0.1:8000/health; echo
echo "--- POST /score (invented payment)"
curl -s -X POST http://127.0.0.1:8000/score -H 'content-type: application/json' \
  -d '{"TransactionAmt": 149.5, "ProductCD": "W", "card4": "visa", "card6": "debit", "C1": 2, "C2": 1}'; echo
echo "--- container health status"
docker inspect --format '{{.State.Health.Status}}' "$NAME" 2>/dev/null || true
echo "--- container log (no payload values expected)"
docker logs "$NAME" 2>&1 | tail -5
docker stop "$NAME" >/dev/null
echo "stopped"
