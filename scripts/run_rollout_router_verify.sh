#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- Required inputs (override via env vars) ----
INPUT="${INPUT:-data/apigen_with_instruction_full.json.with_gt.filtered.json}"
OUTPUT="${OUTPUT:-outputs/router_rollouts/sft_router_verify.jsonl}"
INTERACTION_CONFIG="${INTERACTION_CONFIG:-config/apigen_router_interaction_grpo.yaml}"

# ---- Rollout settings ----
N="${N:-8}"
WORKERS="${WORKERS:-16}"
START="${START:-0}"
END="${END:--1}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

# ---- Router settings (tau-bench aligned defaults) ----
ROUTER_BASE_URL="${ROUTER_BASE_URL:-http://127.0.0.1:30001/v1}"
ROUTER_API_KEY="${ROUTER_API_KEY:-EMPTY}"
ROUTER_MODEL="${ROUTER_MODEL:-default}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-0.0}"
ROUTER_MAX_TOKENS="${ROUTER_MAX_TOKENS:-512}"
ROUTER_TOP_P="${ROUTER_TOP_P:-}"
ROUTER_TOP_K="${ROUTER_TOP_K:-}"
ROUTER_TIMEOUT="${ROUTER_TIMEOUT:-}"
ROUTER_MAX_RETRIES="${ROUTER_MAX_RETRIES:-10}"
ROUTER_CONFIG="${ROUTER_CONFIG:-}"

# ---- User simulator override (optional) ----
USER_MODEL="${USER_MODEL:-}"
USER_PROVIDER="${USER_PROVIDER:-}"
OSS_TEMPERATURE="${OSS_TEMPERATURE:-1.0}"

CMD=(
  python "${ROOT_DIR}/scripts/rollout_router_verify.py"
  --input "${INPUT}"
  --output "${OUTPUT}"
  --interaction-config "${INTERACTION_CONFIG}"
  --n "${N}"
  --workers "${WORKERS}"
  --start "${START}"
  --end "${END}"
  --effort-mode router
  --router-base-url "${ROUTER_BASE_URL}"
  --router-api-key "${ROUTER_API_KEY}"
  --router-model "${ROUTER_MODEL}"
  --router-temperature "${ROUTER_TEMPERATURE}"
  --router-max-tokens "${ROUTER_MAX_TOKENS}"
  --router-max-retries "${ROUTER_MAX_RETRIES}"
)

if [[ -n "${ROUTER_TOP_P}" ]]; then
  CMD+=(--router-top-p "${ROUTER_TOP_P}")
fi
if [[ -n "${ROUTER_TOP_K}" ]]; then
  CMD+=(--router-top-k "${ROUTER_TOP_K}")
fi
if [[ -n "${ROUTER_TIMEOUT}" ]]; then
  CMD+=(--router-timeout "${ROUTER_TIMEOUT}")
fi
if [[ -n "${ROUTER_CONFIG}" ]]; then
  CMD+=(--router-config "${ROUTER_CONFIG}")
fi
if [[ -n "${USER_MODEL}" ]]; then
  CMD+=(--user-model "${USER_MODEL}")
fi
if [[ -n "${USER_PROVIDER}" ]]; then
  CMD+=(--user-provider "${USER_PROVIDER}")
fi
if [[ -n "${OSS_TEMPERATURE}" ]]; then
  CMD+=(--oss-temperature "${OSS_TEMPERATURE}")
fi
if [[ "${SKIP_EXISTING}" == "true" ]]; then
  CMD+=(--skip-existing)
fi

echo "[run_rollout_router_verify] ${CMD[*]}"
exec "${CMD[@]}"
