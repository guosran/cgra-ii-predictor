"""CGRA initiation-interval prediction and static-shape ranking.

Shape and collection utilities are usable without the optional model runtime.
Model classes remain available from this package and are imported on demand.
"""

from .shape_protocol import SHAPE_PROTOCOL, SHAPE_PROTOCOL_ID


_MODEL_EXPORTS = {
    "CategoricalMapperIIModel",
    "DirectMapperIIEnsemble",
    "DirectMapperIIModel",
    "RidgeMapperIIModel",
    "MapperModelConfig",
}


def __getattr__(name: str):
    if name not in _MODEL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import mapper_model
    value = getattr(mapper_model, name)
    globals()[name] = value
    return value

__all__ = [
    "CategoricalMapperIIModel", "DirectMapperIIEnsemble",
    "DirectMapperIIModel", "RidgeMapperIIModel",
    "MapperModelConfig",
    "SHAPE_PROTOCOL", "SHAPE_PROTOCOL_ID",
]
