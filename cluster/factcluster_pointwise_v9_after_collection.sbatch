#!/bin/bash
#SBATCH --job-name=cgra-ii-v9-launch
#SBATCH --partition=eda
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/pointwise-v9-launch-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
V9_ROOT=/fact_data/yibozhang/cgra-ii-model2/multi-cgra-v9
MANIFEST=${V9_ROOT}/motif-v9-largeop-seed-20260913-r3/corpus-manifest.json
PLACEMENT=${V9_ROOT}/placement-supervision-v8-v9.json
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python

cd "${PROJECT_ROOT}"
"${PYTHON}" adapters/report_motif_v8_corpus.py \
  --manifest "${MANIFEST}" \
  --output "${V9_ROOT}/motif-v9-largeop-audit.json"

manifest_sha256=$(sha256sum "${MANIFEST}" | awk '{print $1}')
placement_sha256=$(sha256sum "${PLACEMENT}" | awk '{print $1}')

train=$(sbatch --parsable \
  --export="ALL,EXPECTED_V9_MANIFEST_SHA256=${manifest_sha256},EXPECTED_PLACEMENT_SHA256=${placement_sha256}" \
  cluster/factcluster_pointwise_v9_largeop_train.sbatch)

printf 'manifest_sha256=%s\nplacement_sha256=%s\ntraining=%s\n' \
  "${manifest_sha256}" "${placement_sha256}" "${train}"
