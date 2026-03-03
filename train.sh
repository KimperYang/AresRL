#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /anaconda/etc/profile.d/conda.sh
conda activate rl

export NVTE_CUDA_INCLUDE_DIR=/usr/local/cuda/include
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

cd "${SCRIPT_DIR}"

python "${SCRIPT_DIR}/scripts/generate_maze_data.py" --out_dir "${SCRIPT_DIR}/data/maze"
python -m verl.trainer.main_ppo \
  --config-path "${SCRIPT_DIR}/config" \
  --config-name maze_ppo \
  +ray_kwargs.ray_init.runtime_env.working_dir="${SCRIPT_DIR}"
