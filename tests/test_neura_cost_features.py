import pytest

from adapters.neura_cost_features import parse_cost_features


def test_parse_analysis_only_cost_features():
    text = """
    module attributes {rec_res_mii_info, rec_mii = 3 : i32,
                       res_mii = 5 : i32} {}
    """
    assert parse_cost_features(text) == {"rec_mii": 3, "res_mii": 5}


def test_reject_mapper_labels():
    with pytest.raises(ValueError, match="mapping/label tokens"):
        parse_cost_features(
            "rec_res_mii_info rec_mii = 1 : i32 res_mii = 2 : i32 "
            "compiled_ii = 4 : i32"
        )
