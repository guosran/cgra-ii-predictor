# Evaluation protocol

This project distinguishes exploratory model development from a frozen blind
evaluation. A nested holdout can reduce hyperparameter leakage, but it does not
make a corpus blind when the same corpus informed feature or model-class
choices.

## Development evaluation

Model 1 is predeclared as residual Ridge regression above
`max(RecMII, ResMII)`. The floor and its two components are excluded from the
primary model feature vector. The split unit is generated base-DFG lineage,
not an architecture row. Ridge strength and residual dead zone are selected
only inside each outer training fold.

Primary training gives every generated lineage total weight one; distinct
observations are balanced within each lineage. Weights still sum to the number
of lineages, preserving the meaning of the Ridge penalty. Rows
sharing the same observation identity under the active model projection share
one observation's weight.

Each outer or inner validation invocation independently uses
leave-one-lineage-out when its current group set has at most 20 groups;
otherwise it uses up to 10 deterministic stratified group folds. Every
candidate of a base DFG stays in one fold.

Reports include two kinds of diagnostics:

- point-estimation metrics, including MAE, exact rate, within-one rate, signed
  error, and maximum error;
- tie-aware within-`ranking_query_id` pairwise concordance for architecture/DSE
  ranking.

A base-DFG query with one candidate or constant `compiled_ii` has no ranking
signal and is excluded with an explicit reason. Candidate IDs are scoped to
their query. A query spanning multiple leakage groups is invalid because its
predictions came from different outer models. A prediction tie receives half
credit; an actual-II tie creates no comparable pair.

Use an explicit lineage alias when legacy family names share source ancestry:

```sh
PYTHONPATH=src python3 -m cgra_ii_predictor.cli dataset.json \
  --group-alias old-name-a=common-lineage \
  --group-alias old-name-b=common-lineage
```

If every sample has the requested identity, additional architecture- or
suite-domain holdouts can be run without changing source-lineage training
weights. At least three distinct metadata domains are required. These nested
runs are opt-in because their cost grows quickly with the number of distinct
domain identities:

```sh
PYTHONPATH=src python3 -m cgra_ii_predictor.cli dataset.json \
  --holdout-metadata-key architecture_id \
  --holdout-metadata-key suite
```

Generated motif runs use a separate predeclared corpus manifest.  The primary
generated split is by base-DFG lineage
(`generated/<generator_version>/<motif>/<base_id>`); use
`--holdout-metadata-key generator_family` for a motif-family holdout when at
least three distinct generator families are present. Generated rows are the
only primary fitting data and do not count as real-program evidence. A motif
manifest must be written
atomically before mapper execution and must retain every timeout/nonzero exit
as a censored candidate.

All reports produced by this command are marked `exploratory`, with labels
available at evaluation time and `frozen_test=false`.

## Frozen blind evaluation

A frozen claim requires a separate workflow, not a command-line status flag:

1. Freeze the code revision, generated-only model class, feature names,
   hyperparameter grid, grouping policy, training dataset hash, and mapper
   configuration with `machsuite_frozen.py freeze-model`.
2. Run `preflight` against the fixed 19-entry MachSuite inventory. It records
   sample IDs, source/dependency hashes, lineages, architecture identity,
   candidate IDs, and every unsupported frontend case without running mapper.
3. Verify that train and test source-algorithm lineages and canonical DFG hashes
   are disjoint. For a topology transfer claim, also make the relevant
   architecture families disjoint.
4. Train once, serialize the model, and record the model artifact SHA-256.
5. Run `predict`; it rejects labels, exploratory models, identity warnings, and
   non-primary feature contracts, re-derives features from the lowered/cost
   artifacts, then hashes the predictions into a seal.
6. Only after that seal exists, run `reveal`; it verifies the model, manifest,
   prediction, tool, source, DFG, and architecture hashes before invoking the
   mapper. Failures/timeouts remain censored and labels are written separately.
7. Report all predeclared metrics and every exclusion. Do not select a new
   feature set, model class, or threshold from the frozen results.

`freeze-model` validates archived training labels against deterministic source,
mapped-artifact, clean Neura revision, and mapper-binary hashes and reproduces
the point model; it deliberately does not pay the cost of rerunning every
training mapping. Preserve the corpus manifest and mapped artifacts as the
label-provenance record.

The repository now contains the pinned inventory and executable three-stage
workflow. The label-free compatibility preflight has 11/19 ready and 8/19
lowering-censored candidates. No final generated-only model has yet been
frozen and no MachSuite labels have been revealed, so there is still no frozen
accuracy claim.

The local seal is tamper-evident, not a trusted timestamp: all files could in
principle be regenerated after labels were seen. Publish the seal hash to Git,
an immutable artifact store, or an independent timestamp service before label
reveal when making a blind-test claim.

## What LISA contributes to the roadmap

LISA is useful inspiration for a later mapping-aware model, but it is not a
direct compiled-II regressor. The paper trains separate GNNs to predict node
and edge labels that guide placement and routing. It retrains for each target
accelerator and reports 1,000 randomly generated training DFGs per accelerator.

Accordingly, a Model 2 in this project should first collect auditable
mapping-derived auxiliary labels—such as scheduling priority, spatial and
temporal distance, or routing pressure—and test whether they improve the
unchanged mapper. A GNN is justified only after there are enough independent
graphs and architecture domains for lineage, suite, and architecture holdouts.
The present 54-row corpus has 15 declared kernel names but at most 12 possible
lineages under the conservative alias audit. It is not sufficient evidence for
that step.
