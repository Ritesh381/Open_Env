#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Running unit/API tests"
pytest -q

echo "[2/5] Validating OpenEnv package structure"
openenv validate

echo "[3/5] Checking required endpoints (local server must be running at :8000)"
curl -fsS "http://localhost:8000/health" >/dev/null
curl -fsS "http://localhost:8000/tasks" >/dev/null
curl -fsS "http://localhost:8000/baseline" >/dev/null || true
curl -fsS -X POST "http://localhost:8000/reset" \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task1_security_basic"}' >/dev/null
curl -fsS -X POST "http://localhost:8000/grader" \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task1_security_basic","action":{"inline_comments":[{"file_path":"app/auth.py","line_number":11,"comment":"SQL injection risk in query construction.","severity":"critical","category":"security"}],"general_comments":[],"decision":{"decision":"request_changes","summary":"Critical security issue."},"submit":true}}' >/dev/null

echo "[4/5] Checking Docker build"
docker build -t pr-review-env:check .

echo "[5/5] Optional baseline refresh"
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  python3 baseline/baseline_inference.py --provider openai --temperature 0.0 --seed 42 --output baseline_results_latest.json
elif [[ -n "${GROQ_API_KEY:-}" ]]; then
  python3 baseline/baseline_inference.py --provider groq --model openai/gpt-oss-120b --temperature 0.0 --seed 42 --output baseline_results_latest.json
else
  echo "No provider API key set; skipping live baseline run."
fi

echo "Pre-submission checks completed."
