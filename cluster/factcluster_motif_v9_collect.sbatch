#!/bin/bash
#SBATCH --job-name=cgra-ii-v9-largeop
#SBATCH --partition=eda
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/motif-v9-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
DATA_ROOT=/fact_data/yibozhang/cgra-ii-model2/multi-cgra-v9
CORPUS=${DATA_ROOT}/motif-v9-largeop-seed-20260913-r3
MANIFEST=${CORPUS}/corpus-manifest.json
V8_MANIFEST=/fact_data/yibozhang/cgra-ii-model2/multi-cgra-v8/motif-v8-formal-seed-20260911/corpus-manifest.json
COMBINED_PLACEMENT=${DATA_ROOT}/placement-supervision-v8-v9.json
AUDIT_REPORT=${DATA_ROOT}/motif-v9-largeop-audit.json
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
OPT=/fact_data/yibozhang/cgra-ii-model2/tools/mlir-neura-opt-47b7e3a6
EXPECTED_OPT_SHA256=7c8b0753609c9045fd4dabd3041f5a5311ce3922f431526c9a61ea4d794e6d49
EXPECTED_ARCHITECTURE_SHA256=f244f15be30604eb32eb96e4837a4bf1ce5c34961c3a46299b90931505cc97e6
EXPECTED_V8_MANIFEST_SHA256=33ef4a2dc8385b8245234cb4cc95817ab321c35291a2663f0f09103f0888823f
ARCHITECTURE=/fact_data/yibozhang/cgra-ii-model2/tools/neura-architecture-f244f15b.yaml
SAMPLES_PER_FAMILY=${SAMPLES_PER_FAMILY:-50}
SEED=20260913
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-60}
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK}}

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=${SEED}

mkdir -p "${DATA_ROOT}"
cd "${PROJECT_ROOT}"

actual_opt_sha256=$(sha256sum "${OPT}" | awk '{print $1}')
actual_architecture_sha256=$(sha256sum "${ARCHITECTURE}" | awk '{print $1}')
actual_v8_sha256=$(sha256sum "${V8_MANIFEST}" | awk '{print $1}')
if [[ "${actual_opt_sha256}" != "${EXPECTED_OPT_SHA256}" ]]; then
  echo "mlir-neura-opt hash mismatch" >&2
  exit 2
fi
if [[ "${actual_architecture_sha256}" != "${EXPECTED_ARCHITECTURE_SHA256}" ]]; then
  echo "Neura architecture hash mismatch" >&2
  exit 2
fi
if [[ "${actual_v8_sha256}" != "${EXPECTED_V8_MANIFEST_SHA256}" ]]; then
  echo "frozen v8 manifest hash mismatch" >&2
  exit 2
fi

echo "job_id=${SLURM_JOB_ID}"
echo "generator=motif-v9"
echo "training_role=training_only_augmentation"
echo "operation_bands=24-29,30-35,36-40"
echo "samples_per_family=${SAMPLES_PER_FAMILY}"
echo "seed=${SEED}"
echo "timeout_seconds=${TIMEOUT_SECONDS}"
echo "jobs=${JOBS}"

if [[ ! -f "${MANIFEST}" ]]; then
  srun --unbuffered "${PYTHON}" adapters/neura_experiment.py \
    --opt "${OPT}" \
    --real-architecture "${ARCHITECTURE}" \
    --output-dir "${CORPUS}" \
    --motif-generator-version motif-v9 \
    --motif-samples-per-family "${SAMPLES_PER_FAMILY}" \
    --seed "${SEED}" \
    --timeout "${TIMEOUT_SECONDS}" \
    --motif-jobs "${JOBS}" \
    --motif-checkpoint-every 64 \
    --motif-predeclare-only
fi

srun --unbuffered "${PYTHON}" adapters/neura_experiment.py \
  --opt "${OPT}" \
  --real-architecture "${ARCHITECTURE}" \
  --output-dir "${CORPUS}" \
  --motif-generator-version motif-v9 \
  --motif-samples-per-family "${SAMPLES_PER_FAMILY}" \
  --seed "${SEED}" \
  --timeout "${TIMEOUT_SECONDS}" \
  --motif-jobs "${JOBS}" \
  --motif-checkpoint-every 64 \
  --motif-resume \
  --motif-collect-only

"${PYTHON}" adapters/neura_placement_supervision.py \
  --manifest "${V8_MANIFEST}" \
  --manifest "${MANIFEST}" \
  --output "${COMBINED_PLACEMENT}"

"${PYTHON}" adapters/report_motif_v8_corpus.py \
  --manifest "${MANIFEST}" \
  --output "${AUDIT_REPORT}"

sha256sum "${MANIFEST}" "${COMBINED_PLACEMENT}" "${AUDIT_REPORT}"
