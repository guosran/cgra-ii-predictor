# Deployment adapters

The retained adapters implement only the final Amoeba path:

- consume the frozen static-shape manifest emitted by Amoeba;
- extract task DFGs and obtain analysis-only RecMII/ResMII;
- generate the direct heuristic-mapper II cost catalog;
- score frozen candidates and compare them with true mapper oracles.

The model never chooses a program. Amoeba's C++ manifest reader owns candidate
enumeration, canonical ordering, and fixed-orientation packing validation. The
predictor validates the frozen records and derives only the exact task/shape
queries referenced by those records; it does not reconstruct the candidate
space. Compiler output containing mapper labels is rejected by the analysis-only
feature parser.

TODO: a future analytical spatial-temporal scheduler may model cross-time tile
reuse. That scheduler will be a separate search scope and manifest contract;
the current adapter consumes only the validated static manifest.

The external candidate, score, and cost schema identifiers are dictated by
the current Amoeba executable. Keep those wire identifiers unchanged unless
both repositories are migrated together.

Candidate enumeration also writes `amoeba.source_task_body_sha256` onto each
source task in its MLIR output. Run `extract_amoeba_task_dfgs.py` on that bound
output, not on the pre-enumeration input. Pass `--function NAME` when the input
module contains more than one function; the extractor then selects that one
function region before collecting its tasks. The extractor carries the
identity into each standalone DFG; feature generation, catalog generation, and
Amoeba all reject a missing or mismatched identity before scoring.
