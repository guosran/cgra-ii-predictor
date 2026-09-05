#!/bin/bash
#SBATCH --job-name=cgra-ii-v8-labels
#SBATCH --partition=eda
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/motif-v8-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
DATA_ROOT=/fact_data/yibozhang/cgra-ii-model2/multi-cgra-v8
CORPUS=${DATA_ROOT}/motif-v8-formal-seed-20260911
MANIFEST=${CORPUS}/corpus-manifest.json
PLACEMENT=${DATA_ROOT}/placement-supervision.json
AUDIT_REPORT=${DATA_ROOT}/motif-v8-formal-audit.json
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
OPT=/fact_data/yibozhang/cgra-ii-model2/tools/mlir-neura-opt-47b7e3a6
EXPECTED_OPT_SHA256=7c8b0753609c9045fd4dabd3041f5a5311ce3922f431526c9a61ea4d794e6d49
ARCHITECTURE=/fact_data/yibozhang/cgra-ii-model2/tools/neura-architecture-f244f15b.yaml
EXPECTED_ARCHITECTURE_SHA256=f244f15be30604eb32eb96e4837a4bf1ce5c34961c3a46299b90931505cc97e6
SAMPLES_PER_FAMILY=${SAMPLES_PER_FAMILY:-250}
SEED=20260911
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-60}
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK}}

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=${SEED}

mkdir -p "${DATA_ROOT}" "$(dirname "${OPT}")"
cd "${PROJECT_ROOT}"

actual_opt_sha256=$(sha256sum "${OPT}" | awk '{print $1}')
actual_architecture_sha256=$(sha256sum "${ARCHITECTURE}" | awk '{print $1}')
if [[ "${actual_opt_sha256}" != "${EXPECTED_OPT_SHA256}" ]]; then
  echo "mlir-neura-opt hash mismatch" >&2
  exit 2
fi
if [[ "${actual_architecture_sha256}" != "${EXPECTED_ARCHITECTURE_SHA256}" ]]; then
  echo "Neura architecture hash mismatch" >&2
  exit 2
fi

echo "job_id=${SLURM_JOB_ID}"
echo "host=${SLURMD_NODENAME:-unknown}"
echo "generator=motif-v8"
echo "shape_protocol=amoeba-static-rectangles-4x4-tiles-v1"
echo "seed=${SEED}"
echo "samples_per_family=${SAMPLES_PER_FAMILY}"
echo "mapper=neura-heuristic"
echo "mapper_revision=47b7e3a68c321075293e6fcb45fb3b1cabb93b88"
echo "mapper_sha256=${actual_opt_sha256}"
echo "architecture_sha256=${actual_architecture_sha256}"
echo "timeout_seconds=${TIMEOUT_SECONDS}"
echo "jobs=${JOBS}"
echo "corpus=${CORPUS}"

if [[ ! -f "${MANIFEST}" ]]; then
  srun --unbuffered "${PYTHON}" adapters/neura_experiment.py \
    --opt "${OPT}" \
    --real-architecture "${ARCHITECTURE}" \
    --output-dir "${CORPUS}" \
    --motif-generator-version motif-v8 \
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
  --motif-generator-version motif-v8 \
  --motif-samples-per-family "${SAMPLES_PER_FAMILY}" \
  --seed "${SEED}" \
  --timeout "${TIMEOUT_SECONDS}" \
  --motif-jobs "${JOBS}" \
  --motif-checkpoint-every 64 \
  --motif-resume \
  --motif-collect-only

"${PYTHON}" adapters/neura_placement_supervision.py \
  --manifest "${MANIFEST}" \
  --output "${PLACEMENT}"

"${PYTHON}" adapters/report_motif_v8_corpus.py \
  --manifest "${MANIFEST}" \
  --output "${AUDIT_REPORT}"

sha256sum \
  "${MANIFEST}" \
  "${CORPUS}/corpus-manifest.predeclared.json" \
  "${PLACEMENT}" \
  "${AUDIT_REPORT}"
