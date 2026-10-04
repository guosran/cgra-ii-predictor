import pytest

from cgra_ii_predictor.shape_protocol import (
    SHAPE_PROTOCOL,
    SHAPE_PROTOCOL_2X2,
    SHAPE_PROTOCOL_2X2_ID,
    SHAPE_PROTOCOL_4X4,
    SHAPE_PROTOCOL_FULL16,
    SHAPE_PROTOCOL_ID,
    get_shape_protocol,
    get_shape_protocol_for_per_cgra,
)


EXPECTED_2X2_MAPPINGS = (
    ((1, 1), (2, 2)),
    ((1, 2), (2, 4)),
    ((2, 1), (4, 2)),
    ((1, 3), (2, 6)),
    ((3, 1), (6, 2)),
    ((1, 4), (2, 8)),
    ((4, 1), (8, 2)),
    ((2, 2), (4, 4)),
)


def test_2x2_protocol_exposes_exact_max4_oriented_domain():
    protocol = SHAPE_PROTOCOL_2X2

    assert protocol.protocol_id == SHAPE_PROTOCOL_2X2_ID
    assert protocol.per_cgra_rows == 2
    assert protocol.per_cgra_cols == 2
    assert protocol.max_physical_cgras == 4
    assert protocol.physical_to_mapper == EXPECTED_2X2_MAPPINGS
    assert protocol.mapper_shapes == tuple(
        mapper for _, mapper in EXPECTED_2X2_MAPPINGS
    )

    for physical, mapper in EXPECTED_2X2_MAPPINGS:
        assert protocol.validate_physical_shape(*physical) == physical
        assert protocol.mapper_for_physical(*physical) == mapper
        assert protocol.physical_for_mapper(*mapper) == physical


def test_2x2_normalizers_are_maxima_of_the_supported_domain():
    protocol = SHAPE_PROTOCOL_2X2
    shapes = protocol.mapper_shapes

    assert protocol.max_mapper_rows == max(rows for rows, _ in shapes) == 8
    assert protocol.max_mapper_cols == max(cols for _, cols in shapes) == 8
    assert protocol.max_mapper_tiles == max(
        rows * cols for rows, cols in shapes
    ) == 16
    assert protocol.max_directed_links == max(
        2 * (rows * (cols - 1) + cols * (rows - 1))
        for rows, cols in shapes
    ) == 48
    assert protocol.max_north_or_west_boundary_tiles == max(
        rows + cols - 1 for rows, cols in shapes
    ) == 9
    assert protocol.max_bisection_links == max(
        0 if rows * cols == 1 else 2 * min(rows, cols)
        for rows, cols in shapes
    ) == 8
    assert protocol.max_manhattan_distance == max(
        rows + cols - 2 for rows, cols in shapes
    ) == 8


def test_protocol_lookup_accepts_id_or_architecture_core_dimensions():
    assert get_shape_protocol() is SHAPE_PROTOCOL
    assert get_shape_protocol(SHAPE_PROTOCOL_ID) is SHAPE_PROTOCOL_4X4
    assert get_shape_protocol(SHAPE_PROTOCOL_2X2_ID) is SHAPE_PROTOCOL_2X2
    assert get_shape_protocol_for_per_cgra(2, 2) is SHAPE_PROTOCOL_2X2
    assert get_shape_protocol_for_per_cgra(4, 4) is SHAPE_PROTOCOL_4X4
    assert SHAPE_PROTOCOL_FULL16 is SHAPE_PROTOCOL_4X4


def test_legacy_4x4_default_keeps_its_full16_domain():
    assert SHAPE_PROTOCOL is SHAPE_PROTOCOL_4X4
    assert SHAPE_PROTOCOL.protocol_id == "amoeba-static-rectangles-4x4-tiles"
    assert SHAPE_PROTOCOL.per_cgra_rows == 4
    assert SHAPE_PROTOCOL.per_cgra_cols == 4
    assert SHAPE_PROTOCOL.max_physical_cgras == 16
    assert SHAPE_PROTOCOL.mapper_for_physical(4, 4) == (16, 16)
    assert len(SHAPE_PROTOCOL.mapper_shapes) == 16


@pytest.mark.parametrize(
    "physical_shape",
    ((0, 1), (1, 0), (1, 5), (5, 1), (2, 3), (3, 2), (4, 4)),
)
def test_2x2_protocol_rejects_unsupported_or_out_of_grid_physical_shapes(
    physical_shape,
):
    with pytest.raises(ValueError, match="outside protocol"):
        SHAPE_PROTOCOL_2X2.mapper_for_physical(*physical_shape)


@pytest.mark.parametrize("bad_shape", ((1, 2), (2, 1), (6, 6), (8, 8)))
def test_2x2_protocol_rejects_mapper_shapes_outside_its_domain(bad_shape):
    with pytest.raises(ValueError, match="outside protocol"):
        SHAPE_PROTOCOL_2X2.validate_mapper_shape(*bad_shape)


def test_lookups_reject_unknown_protocols_and_tile_dimensions():
    with pytest.raises(ValueError, match="unsupported shape protocol"):
        get_shape_protocol("amoeba-static-rectangles-3x3-per-cgra")
    with pytest.raises(ValueError, match="unsupported per-CGRA tile shape"):
        get_shape_protocol_for_per_cgra(3, 3)
    with pytest.raises(ValueError, match="must be integers"):
        get_shape_protocol_for_per_cgra(True, 2)
