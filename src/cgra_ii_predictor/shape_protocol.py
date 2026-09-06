"""Static physical-CGRA shapes and mapper feature normalizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


MapperShape = Tuple[int, int]
PhysicalShape = Tuple[int, int]
SHAPE_PROTOCOL_ID = "amoeba-static-rectangles-4x4-tiles"


@dataclass(frozen=True)
class ShapeProtocol:
    """Finite validated shape domain plus fixed feature normalizers."""

    protocol_id: str
    mapper_shapes: Tuple[MapperShape, ...]
    max_mapper_rows: int
    max_mapper_cols: int
    max_mapper_tiles: int
    max_directed_links: int
    max_memory_tiles: int
    max_bisection_links: int
    max_manhattan_distance: int
    physical_to_mapper: Tuple[Tuple[PhysicalShape, MapperShape], ...]

    def validate_mapper_shape(self, rows: int, cols: int) -> MapperShape:
        if isinstance(rows, bool) or isinstance(cols, bool):
            raise ValueError("mapper tile dimensions must be integers")
        if not isinstance(rows, int) or not isinstance(cols, int):
            raise ValueError("mapper tile dimensions must be integers")
        shape = (rows, cols)
        if shape not in self.mapper_shapes:
            raise ValueError(
                f"mapper tile shape {rows}x{cols} is outside protocol "
                f"{self.protocol_id}"
            )
        return shape

    def mapper_for_physical(self, rows: int, cols: int) -> MapperShape:
        try:
            return dict(self.physical_to_mapper)[(rows, cols)]
        except KeyError as error:
            raise ValueError(
                f"physical CGRA shape {rows}x{cols} is outside protocol "
                f"{self.protocol_id}"
            ) from error

    def physical_for_mapper(self, rows: int, cols: int) -> PhysicalShape:
        self.validate_mapper_shape(rows, cols)
        reverse = {mapper: physical for physical, mapper in self.physical_to_mapper}
        return reverse[(rows, cols)]

    def to_dict(self) -> Dict[str, object]:
        return {
            "protocol_id": self.protocol_id,
            "supported_mapper_tile_shapes": [
                {"rows": rows, "cols": cols}
                for rows, cols in self.mapper_shapes
            ],
            "physical_to_mapper": [
                {
                    "physical_cgra_rows": physical[0],
                    "physical_cgra_cols": physical[1],
                    "mapper_tile_rows": mapper[0],
                    "mapper_tile_cols": mapper[1],
                }
                for physical, mapper in self.physical_to_mapper
            ],
            "normalization": {
                "mapper_rows": self.max_mapper_rows,
                "mapper_cols": self.max_mapper_cols,
                "mapper_tiles": self.max_mapper_tiles,
                "directed_links": self.max_directed_links,
                "memory_tiles": self.max_memory_tiles,
                "bisection_links": self.max_bisection_links,
                "manhattan_distance": self.max_manhattan_distance,
            },
            "orientation_equivalent": False,
            "shape_scope": "static_rectangular_only",
        }


SHAPE_PROTOCOL = ShapeProtocol(
    protocol_id=SHAPE_PROTOCOL_ID,
    mapper_shapes=(
        (4, 4), (4, 8), (8, 4), (4, 12),
        (12, 4), (4, 16), (8, 8), (16, 4),
    ),
    max_mapper_rows=16,
    max_mapper_cols=16,
    max_mapper_tiles=64,
    max_directed_links=224,
    max_memory_tiles=19,
    max_bisection_links=16,
    max_manhattan_distance=18,
    physical_to_mapper=(
        ((1, 1), (4, 4)),
        ((1, 2), (4, 8)),
        ((2, 1), (8, 4)),
        ((1, 3), (4, 12)),
        ((3, 1), (12, 4)),
        ((1, 4), (4, 16)),
        ((2, 2), (8, 8)),
        ((4, 1), (16, 4)),
    ),
)


def get_shape_protocol(protocol_id: str = SHAPE_PROTOCOL_ID) -> ShapeProtocol:
    """Resolve the single supported deployment shape protocol."""
    if protocol_id != SHAPE_PROTOCOL_ID:
        raise ValueError(f"unsupported shape protocol: {protocol_id!r}")
    return SHAPE_PROTOCOL


__all__ = [
    "MapperShape", "PhysicalShape", "SHAPE_PROTOCOL", "SHAPE_PROTOCOL_ID",
    "ShapeProtocol", "get_shape_protocol",
]
