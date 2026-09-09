# Deployment adapters

The retained adapters implement only the final Amoeba path:

- consume the packing-pruned static-shape manifest emitted by Amoeba;
- extract task DFGs and obtain analysis-only RecMII/ResMII;
- generate the direct heuristic-mapper II cost catalog;
- score frozen candidates and compare them with true mapper oracles.

The model never chooses a program. Amoeba's current candidate protocol guarantees
that all fixed-orientation task rectangles in a candidate can simultaneously
fit without overlap on the physical grid. Temporal reuse does not relax this
capacity constraint. Compiler output containing mapper labels is rejected by
the analysis-only feature parser.

TODO: a future analytical spatial-temporal scheduler may model cross-time tile
reuse. That scheduler will be a separate search scope; the current adapter
must continue to reject over-capacity candidates.

The external candidate, score, and cost schema identifiers are dictated by
the current Amoeba executable. Keep those wire identifiers unchanged unless
both repositories are migrated together.

Candidate enumeration also writes `amoeba.source_task_body_sha256` onto each
source task in its MLIR output. Run `extract_amoeba_task_dfgs.py` on that bound
output, not on the pre-enumeration input. The extractor carries the identity
into each standalone DFG; feature generation, catalog generation, and Amoeba
all reject a missing or mismatched identity before scoring.
