#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /anaconda/etc/profile.d/conda.sh
conda activate rl
source "${SCRIPT_DIR}/env_setup.sh"

export NVTE_CUDA_INCLUDE_DIR=/usr/local/cuda/include
export TAU_BENCH_ROOT="/home/azureuser/cloudfiles/code/Users/jingbo.yang/tau-bench"
export PYTHONPATH="${TAU_BENCH_ROOT}:${SCRIPT_DIR}:${PYTHONPATH}"
export RAY_DISABLE_DASHBOARD=1
export RAY_DEDUP_LOGS=0
export RAY_raylet_start_wait_time_s=120
export HF_HOME="${SCRIPT_DIR}/outputs/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export XDG_CACHE_HOME="${HF_HOME}"
RAY_TMP_TARGET="${NVME_RAY_DIR:-/tmp/ray}"

# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_SOCKET_IFNAME=^lo,docker0


mkdir -p "${RAY_TMP_TARGET}"
mkdir -p "${HF_DATASETS_CACHE}"
mkdir -p "${TRANSFORMERS_CACHE}"

export RAY_TMPDIR="${RAY_TMP_TARGET}"
export TMPDIR="${RAY_TMP_TARGET}"
export RAY_OBJECT_SPILLING_CONFIG='{"type":"filesystem","params":{"directory_path":"'"${RAY_TMP_TARGET}"'"}}'

ray stop --force >/dev/null 2>&1 || true

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="2,3,4,5"
fi

ray start --head --num-gpus=4 --temp-dir="${RAY_TMP_TARGET}" >/dev/null
export RAY_ADDRESS="auto"

cd "${SCRIPT_DIR}"

DATA_OUT_DIR="${SCRIPT_DIR}/data/apigen_router_gt50"
VAL_COUNT=10

if [ -f "${DATA_OUT_DIR}/train.parquet" ] || [ -f "${DATA_OUT_DIR}/val.parquet" ]; then
  echo "[train_router] Found existing parquet files in ${DATA_OUT_DIR}; skip regeneration."
else
  python "${SCRIPT_DIR}/scripts/generate_apigen_router_data.py" \
    --out_dir "${DATA_OUT_DIR}" \
    --input_path "${SCRIPT_DIR}/data/apigen_with_instruction_full.json.with_gt.filtered.json" \
    --val_count "${VAL_COUNT}"
fi
python -m verl.trainer.main_ppo \
  --config-path "${SCRIPT_DIR}/config" \
  --config-name apigen_router_ppo \
  data.train_files="${DATA_OUT_DIR}/train.parquet" \
  data.val_files="${DATA_OUT_DIR}/val.parquet" \
  +ray_kwargs.ray_init.runtime_env.working_dir="${SCRIPT_DIR}"
