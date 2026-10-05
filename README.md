# ORBIT per-CGRA 2×2 II predictor

This package preserves the original C0 checkpoint for a fixed 2×2-tile-per-CGRA heuristic mapper. It predicts the mapper’s compiled initiation interval (II) for one DFG and one mapper shape; it does not predict mapper success, placement quality, or whole-program performance.

## Frozen model and input contract

The checkpoint in `models/candidates/per-cgra-2x2/mapper.pt` is the original C0 artifact, SHA-256 `bb7196d5b37d54b1e5245a5e47cb6323cebfb10deea6d902e2ac6425f74c155e`. Its architecture SHA-256 is `6f4a9a1815dcc0d97c00fd6ee20424fa9420ace90654da29e70d283cba7a611f`. The checkpoint and architecture are verified by the package loader.

The packaged training matrix reproduces the original **v1 raw-kernel 148-feature** training cohort and uses the frozen 61-feature mask. Its shape roster, in tie order, is `2×2, 2×4, 4×2, 2×6, 6×2, 2×8, 8×2, 4×4`. Separately, the same unchanged C0 checkpoint has been evaluated under the validated v2 input contract with its repaired feature extraction. That v2 evaluation binding does not change the original v1 cohort or imply that training on this packaged matrix reproduces the v2 feature pipeline. Use the v1 predictor command below for the original raw-kernel contract; use v2 features only with their separately validated v2 feature extraction and binding.

The packaged data is a portable feature matrix; raw DFGs, mapped IR, mapper binaries, and training checkpoints beyond C0 are not included. `dataset/manifest.json` binds the matrix, source-group assignments, exclusion file, shape roster, feature roster, split, architecture, and native label provenance with SHA-256 digests. Censored mapping outcomes retain a JSON `null` II. The 3 whole-source-group exclusions remain explicitly marked and are not used for training or score summaries.

| Partition | DFGs | Queries | Successful II labels | Censored / null |
|---|---:|---:|---:|---:|
| Train | 185 | 1,480 | 1,469 | 11 |
| Validation | 39 | 312 | 312 | 0 |
| Test | 38 | 304 | 304 | 0 |
| Excluded | 61 | 488 | 484 | 4 |

The eligible set has 262 DFGs, 2,085 successful labels, and 11 null labels. The full frozen collection has 323 DFGs and 2,584 query outcomes: 2,569 success and 15 censored. Splits use the original deterministic 70/15/15 source-group assignment (seed 20261004). The four original training seeds are 17, 41, 113, and 239; training uses 80 epochs, the 64/32 MLP, AdamW, and the original weighted SmoothL1 plus pairwise and set-top-1 losses.

## Install and use

Python 3.8+ and PyTorch 2.x are required. Install the package with `python -m pip install .`.

Evaluate the frozen C0 checkpoint against the packaged train, validation, and test matrices without fitting:

```sh
python adapters/train_per_cgra_2x2_model.py --mode evaluate
```

Run inference from one pre-mapper route-expanded DFG and its analytical bounds:

```sh
python adapters/predict_per_cgra_2x2.py \
  --dfg task.mlir --rows 2 --cols 4 --rec-mii 1 --res-mii 3
```

Train a separate output checkpoint from the packaged matrix with the frozen recipe; this command does not overwrite C0:

```sh
python adapters/train_per_cgra_2x2_model.py \
  --mode train --dataset models/candidates/per-cgra-2x2/dataset \
  --output-dir /tmp/per-cgra-2x2-reproduction
```

No retraining was run to assemble this package, and matrix-based training is not claimed to reproduce the frozen weights byte for byte. The original training report and source-group/exclusion records are retained beside C0. These task-local results do not establish scheduler or hardware performance.

Run the focused package checks with `python -m pytest -q`.
