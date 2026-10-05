#!/usr/bin/env python3
"""Predict one compiled II from an original-contract pre-mapper DFG query."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from cgra_ii_predictor.checkpoint import (  # noqa: E402
    EXPECTED_ARCHITECTURE_SHA256,
    load_checkpoint,
    load_published_candidate,
    sha256_file,
)
from cgra_ii_predictor.dfg import require_neura_route_expanded_dfg  # noqa: E402
from cgra_ii_predictor.mapper_model import mapper_feature_vector  # noqa: E402
from cgra_ii_predictor.shape_protocol import SHAPE_PROTOCOL_2X2, SHAPE_PROTOCOL_2X2_ID  # noqa: E402


PACKAGE = ROOT / "models/candidates/per-cgra-2x2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfg", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True, help="mapper tile rows")
    parser.add_argument("--cols", type=int, required=True, help="mapper tile columns")
    parser.add_argument("--rec-mii", type=int, required=True)
    parser.add_argument("--res-mii", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    args = parser.parse_args()
    shape = SHAPE_PROTOCOL_2X2.validate_mapper_shape(args.rows, args.cols)
    if args.rec_mii < 0 or args.res_mii <= 0:
        parser.error("RecMII must be nonnegative and ResMII must be positive")
    lower_bound = max(args.rec_mii, args.res_mii)
    if lower_bound > 20:
        parser.error("analytical lower bound exceeds the original mapper II ceiling")

    if args.checkpoint is None:
        model, _config, metadata = load_published_candidate(PACKAGE)
        checkpoint = PACKAGE / metadata["checkpoint"]["path"]
        checkpoint_sha = metadata["checkpoint"]["sha256"]
        architecture_sha = metadata["architecture"]["sha256"]
    else:
        checkpoint = args.checkpoint.resolve()
        model, _config, artifact = load_checkpoint(
            checkpoint, expected_architecture_sha256=EXPECTED_ARCHITECTURE_SHA256,
        )
        checkpoint_sha = sha256_file(checkpoint)
        architecture_sha = artifact["architecture_sha256"]

    dfg_path = args.dfg.resolve()
    dfg_text = dfg_path.read_text()
    graph = require_neura_route_expanded_dfg(dfg_text)
    features = mapper_feature_vector(
        graph, shape[0], shape[1], float(args.rec_mii), float(args.res_mii),
        float(lower_bound), shape_protocol=SHAPE_PROTOCOL_2X2_ID,
    )
    tensor = torch.tensor([features], dtype=torch.float32)
    lower = torch.tensor([float(lower_bound)], dtype=torch.float32)
    with torch.inference_mode():
        predicted = float(model(tensor, lower)[0].item())
    result = {
        "schema": "cgra-ii-per-cgra-2x2-prediction-v1",
        "input_contract": "original v1 raw-kernel 148-feature contract; not current v2",
        "dfg_sha256": sha256_file(dfg_path),
        "shape_protocol_id": SHAPE_PROTOCOL_2X2_ID,
        "shape": list(shape),
        "rec_mii": args.rec_mii,
        "res_mii": args.res_mii,
        "lower_bound": lower_bound,
        "predicted_ii": predicted,
        "checkpoint_sha256": checkpoint_sha,
        "architecture_sha256": architecture_sha,
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
