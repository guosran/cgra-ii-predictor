# Adapters

Adapters own all compiler- and mapper-specific behavior. They may invoke an
external toolchain to lower programs, read architecture descriptions, collect
pre-mapping features, and obtain labels from an unchanged mapper.

The Neura adapter is migrated from the original analytical-cost-model
experiment. It remains useful for corpus reproduction, but its textual MLIR
features are not part of the generic package API.

An adapter must never turn a timeout into a compiled-II label, and must record
the source family, architecture identity, compiler revision, mapper revision,
and mapper configuration needed to reproduce every successful label.

