#!/usr/bin/env bash
# Build, run, and smoke-test the serving container locally.
# Usage: ./scripts_docker_test.sh
set -euo pipefail

IMAGE="heart-disease-api:latest"
NAME="heart-api-test"
PORT=8000

echo ">> Building image..."
docker build -t "$IMAGE" .

echo ">> Starting container..."
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" -p "${PORT}:8000" "$IMAGE"

echo ">> Waiting for health..."
for i in $(seq 1 20); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null; then
    echo "   healthy."
    break
  fi
  sleep 1
done

echo ">> /health:"
curl -s "http://localhost:${PORT}/health" | python -m json.tool

echo ">> /predict (sample_request.json):"
curl -s -X POST "http://localhost:${PORT}/predict" \
  -H "Content-Type: application/json" \
  -d @sample_request.json | python -m json.tool

echo ">> Logs (last 20 lines):"
docker logs --tail 20 "$NAME"

echo ">> Cleanup: docker rm -f ${NAME}"
