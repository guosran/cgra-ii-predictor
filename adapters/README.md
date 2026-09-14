# Deployment adapters

The retained adapters implement only the final Amoeba path:

- consume the packing-pruned static-shape manifest emitted by Amoeba;
- extract task DFGs and obtain analysis-only RecMII/ResMII;
- generate the ordinary ensemble cost catalog;
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

## Hash and provenance contract

The adapters use SHA-256 for two related but different jobs. `sha256_file()`
hashes the exact bytes of an artifact, including formatting and line endings.
`canonical_json_sha256()` hashes a JSON value after sorting keys and removing
insignificant whitespace; it is used only for semantic JSON contracts. A
canonical JSON hash must not replace a raw file hash when a later stage needs
to prove that it consumed the same file bytes.

The producer/consumer chain is:

| Identity | Produced by | Checked or consumed by | Purpose |
| --- | --- | --- | --- |
| `task.body_sha256` / `source_task_body_sha256` | Amoeba candidate enumeration | DFG extractor, feature generation, catalog generation, frozen pipeline | Binds task `A`'s standalone DFG to the original Taskflow task body used to enumerate candidates. It is a task identity, not a hash of the extracted DFG file. |
| `task_dfg_sha256` | Query-feature generation from each standalone DFG | Catalog generation and mapper replay | Binds analytical facts and predictions to the exact DFG bytes passed to Neura. This catches a regenerated or edited DFG even when its embedded task-body identity is unchanged. |
| `architecture.spec_sha256` / `architecture_sha256` | Amoeba and query-feature generation from the architecture YAML bytes | Query generation, ensemble architecture validation, catalog generation, mapper replay | Binds shape enumeration, Neura analysis, model labels, and mapper replay to one hardware specification. |
| `neura_opt_sha256` | Query-feature generation and frozen-pipeline validation | Catalog provenance and mapper replay | Identifies the exact Neura executable that produced analysis facts or runs the mapper. |
| `candidate_manifest_sha256` | Candidate-manifest loader | Query generation, catalog generation, scoring pipeline | Binds all downstream data to the exact JSONL bytes, including task order, candidate order, and task/shape query set. |
| checkpoint `sha256` | Catalog generation / ensemble report | Ensemble loader and catalog generation | Identifies the exact serialized model weights and metadata selected for inference. |
| checkpoint `config_sha256` | Catalog generation | Catalog metadata and audit tooling | Canonically identifies the validated model configuration, independently of checkpoint JSON formatting and independently of the raw checkpoint bytes. |
| `ensemble_report_sha256` | Ensemble loader | Cost-catalog namespace | Makes changes to ensemble weights, architecture support, or validation metadata create a new predictor identity. |
| `analytical_input_sha256` | Catalog generation | Cost-catalog namespace | Identifies the exact analysis-only RecMII/ResMII JSON consumed by the catalog builder. |
| catalog `namespace` | Canonical hash of the catalog namespace contract | Frozen pipeline score header and catalog consumer | Names the complete semantic input contract: manifest, analytical input, task bodies/DFGs, architecture, Neura executable, checkpoints/configs, ensemble report, shape protocol, and ranking policy. |
| `source_dfg_sha256`, `mapped_artifact_sha256`, `stdout_sha256`, `stderr_sha256` | Mapper replay | Pipeline report and audit tooling | Identifies the input DFG, mapped MLIR output, and logs for each `(task, mapper shape)` attempt. |
| `oracle_sha256` | Oracle evaluation | Pipeline report | Identifies the exact mapper-result corpus used for recall and regret comparisons. |

The source-task hash is read from the bound MLIR rather than recomputed from a
standalone DFG because extraction changes the module and kernel signature. The
standalone DFG hash is therefore retained separately. Likewise, the
architecture, optimizer, manifest, checkpoint, analytical input, ensemble
report, and mapper artifacts use raw file hashes because their exact bytes are
part of the reproducibility boundary; only the model configuration and the
catalog namespace use canonical JSON hashing.
