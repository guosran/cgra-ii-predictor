"""Static physical-CGRA shapes and mapper feature normalizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


MapperShape = Tuple[int, int]
PhysicalShape = Tuple[int, int]
SHAPE_PROTOCOL_4X4_ID = "amoeba-static-rectangles-4x4-tiles"
SHAPE_PROTOCOL_2X2_ID = "amoeba-static-rectangles-2x2-per-cgra-max4"
# Preserve the existing protocol as the default for current model consumers.
SHAPE_PROTOCOL_ID = SHAPE_PROTOCOL_4X4_ID


@dataclass(frozen=True)
class ShapeProtocol:
    """Finite validated shape domain plus fixed feature normalizers."""

    protocol_id: str
    mapper_shapes: Tuple[MapperShape, ...]
    max_mapper_rows: int
    max_mapper_cols: int
    max_mapper_tiles: int
    max_directed_links: int
    max_north_or_west_boundary_tiles: int
    max_bisection_links: int
    max_manhattan_distance: int
    physical_to_mapper: Tuple[Tuple[PhysicalShape, MapperShape], ...]

    @property
    def per_cgra_rows(self) -> int:
        """Mapper tile rows contributed by one physical CGRA core."""
        return self.mapper_for_physical(1, 1)[0]

    @property
    def per_cgra_cols(self) -> int:
        """Mapper tile columns contributed by one physical CGRA core."""
        return self.mapper_for_physical(1, 1)[1]

    @property
    def max_physical_cgras(self) -> int:
        """Largest physical CGRA count represented by this protocol."""
        return max(rows * cols for (rows, cols), _ in self.physical_to_mapper)

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
        physical = self.validate_physical_shape(rows, cols)
        try:
            return dict(self.physical_to_mapper)[physical]
        except KeyError as error:
            raise ValueError(
                f"physical CGRA shape {rows}x{cols} is outside protocol "
                f"{self.protocol_id}"
            ) from error

    def validate_physical_shape(self, rows: int, cols: int) -> PhysicalShape:
        if isinstance(rows, bool) or isinstance(cols, bool):
            raise ValueError("physical CGRA dimensions must be integers")
        if not isinstance(rows, int) or not isinstance(cols, int):
            raise ValueError("physical CGRA dimensions must be integers")
        shape = (rows, cols)
        if shape not in dict(self.physical_to_mapper):
            raise ValueError(
                f"physical CGRA shape {rows}x{cols} is outside protocol "
                f"{self.protocol_id}"
            )
        return shape

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
                "north_or_west_boundary_tiles": (
                    self.max_north_or_west_boundary_tiles
                ),
                "bisection_links": self.max_bisection_links,
                "manhattan_distance": self.max_manhattan_distance,
            },
            "orientation_equivalent": False,
            "shape_scope": "static_rectangular_only",
        }


SHAPE_PROTOCOL_4X4 = ShapeProtocol(
    protocol_id=SHAPE_PROTOCOL_4X4_ID,
    mapper_shapes=(
        (4, 4), (4, 8), (8, 4), (4, 12),
        (12, 4), (4, 16), (8, 8), (16, 4),
        (8, 12), (12, 8), (8, 16), (16, 8),
        (12, 12), (12, 16), (16, 12), (16, 16),
    ),
    max_mapper_rows=16,
    max_mapper_cols=16,
    max_mapper_tiles=256,
    max_directed_links=960,
    max_north_or_west_boundary_tiles=31,
    max_bisection_links=32,
    max_manhattan_distance=30,
    physical_to_mapper=(
        ((1, 1), (4, 4)),
        ((1, 2), (4, 8)),
        ((2, 1), (8, 4)),
        ((1, 3), (4, 12)),
        ((3, 1), (12, 4)),
        ((1, 4), (4, 16)),
        ((2, 2), (8, 8)),
        ((4, 1), (16, 4)),
        ((2, 3), (8, 12)),
        ((3, 2), (12, 8)),
        ((2, 4), (8, 16)),
        ((4, 2), (16, 8)),
        ((3, 3), (12, 12)),
        ((3, 4), (12, 16)),
        ((4, 3), (16, 12)),
        ((4, 4), (16, 16)),
    ),
)

SHAPE_PROTOCOL_FULL16 = SHAPE_PROTOCOL_4X4

SHAPE_PROTOCOL_2X2 = ShapeProtocol(
    protocol_id=SHAPE_PROTOCOL_2X2_ID,
    mapper_shapes=(
        (2, 2), (2, 4), (4, 2), (2, 6),
        (6, 2), (2, 8), (8, 2), (4, 4),
    ),
    max_mapper_rows=8,
    max_mapper_cols=8,
    max_mapper_tiles=16,
    max_directed_links=48,
    max_north_or_west_boundary_tiles=9,
    max_bisection_links=8,
    max_manhattan_distance=8,
    physical_to_mapper=(
        ((1, 1), (2, 2)),
        ((1, 2), (2, 4)),
        ((2, 1), (4, 2)),
        ((1, 3), (2, 6)),
        ((3, 1), (6, 2)),
        ((1, 4), (2, 8)),
        ((4, 1), (8, 2)),
        ((2, 2), (4, 4)),
    ),
)

# Compatibility alias: existing imports retain the 4x4-tile full-16 domain.
SHAPE_PROTOCOL = SHAPE_PROTOCOL_4X4

_SHAPE_PROTOCOLS = {
    protocol.protocol_id: protocol
    for protocol in (SHAPE_PROTOCOL_4X4, SHAPE_PROTOCOL_2X2)
}


def get_shape_protocol(protocol_id: str = SHAPE_PROTOCOL_ID) -> ShapeProtocol:
    """Resolve a supported protocol by its stable identifier."""
    try:
        return _SHAPE_PROTOCOLS[protocol_id]
    except KeyError as error:
        raise ValueError(
            f"unsupported shape protocol: {protocol_id!r}"
        ) from error


def get_shape_protocol_for_per_cgra(
    per_cgra_rows: int, per_cgra_cols: int,
) -> ShapeProtocol:
    """Resolve a protocol from the loaded architecture's per-CGRA tile size."""
    if (isinstance(per_cgra_rows, bool) or isinstance(per_cgra_cols, bool)
            or not isinstance(per_cgra_rows, int)
            or not isinstance(per_cgra_cols, int)):
        raise ValueError("per-CGRA tile dimensions must be integers")
    for protocol in _SHAPE_PROTOCOLS.values():
        if (protocol.per_cgra_rows, protocol.per_cgra_cols) == (
            per_cgra_rows, per_cgra_cols,
        ):
            return protocol
    raise ValueError(
        "unsupported per-CGRA tile shape "
        f"{per_cgra_rows}x{per_cgra_cols}"
    )


__all__ = [
    "MapperShape", "PhysicalShape", "SHAPE_PROTOCOL", "SHAPE_PROTOCOL_ID",
    "SHAPE_PROTOCOL_2X2", "SHAPE_PROTOCOL_2X2_ID", "SHAPE_PROTOCOL_4X4",
    "SHAPE_PROTOCOL_4X4_ID", "SHAPE_PROTOCOL_FULL16", "ShapeProtocol",
    "get_shape_protocol", "get_shape_protocol_for_per_cgra",
]
