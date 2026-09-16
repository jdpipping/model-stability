#!/usr/bin/env bash
#SBATCH --job-name=zoo-mig90
#SBATCH --account=ajw-wharton
#SBATCH --qos=wharton-dgx-b200
#SBATCH --partition=b200-mig90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/ajw/wharton/jpipping/zoo-logs/%x-%j.out
#SBATCH --error=/vast/projects/ajw/wharton/jpipping/zoo-logs/%x-%j.err

set -euo pipefail
source "${ZOO_BETTY_PROJECT_HOST:-/vast/projects/ajw/wharton/jpipping/zoo}/scripts/betty/common.sh"

if (( $# == 0 )); then
  set -- python3 scripts/betty/benchmark_models.py --device gpu
fi

srun \
  --export=ALL \
  --container-image="${ZOO_BETTY_IMAGE}" \
  --container-mounts="${ZOO_BETTY_STORAGE_ROOT}:${ZOO_BETTY_MOUNT}" \
  bash -lc 'export TF_USE_LEGACY_KERAS=0; cd "$ZOO_BETTY_PROJECT" && exec "$@"' bash "$@"
