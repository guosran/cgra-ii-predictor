# Deployment adapters

The retained adapters implement only the final Amoeba path:

- enumerate exact-packable static shape candidates;
- extract task DFGs and obtain analysis-only RecMII/ResMII;
- generate the ordinary ensemble cost catalog;
- score frozen candidates and compare them with true mapper oracles.

The model never chooses a program or bypasses Amoeba's exact packing checks.
Compiler output containing mapper labels is rejected by the analysis-only
feature parser.

The external candidate, score, and cost schema identifiers are dictated by
the current Amoeba executable. Keep those wire identifiers unchanged unless
both repositories are migrated together.
