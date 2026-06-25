import pytest

from bittensor.core.chain_data.subnet_hyperparameters import SubnetHyperparameters

_HYPERPARAMS_V3 = [
    {"name": "kappa", "value": {"U16": 32767}},
    {"name": "tempo", "value": {"U16": 100}},
    {"name": "max_weights_limit", "value": {"U16": 65535}},
    {"name": "weights_rate_limit", "value": {"U64": 100}},
    {"name": "registration_allowed", "value": {"Bool": True}},
    {"name": "liquid_alpha_enabled", "value": {"Bool": False}},
    {"name": "min_burn", "value": {"TaoBalance": 500000}},
    {"name": "alpha_sigmoid_steepness", "value": {"I32F32": {"bits": 4294967296000}}},
    {"name": "activity_cutoff_factor", "value": {"U32": 13889}},
]


@pytest.mark.parametrize(
    "tag",
    ["U8", "U16", "U32", "U64", "U128", "I8", "I16", "I32", "I64"],
)
def test_decode_value_integer_tags(tag):
    """Verify integer SCALE tags decode to plain ints."""
    result = SubnetHyperparameters._decode_value({tag: 42})
    assert result == 42
    assert type(result) is int


def test_decode_value_bool_tag():
    """Verify Bool SCALE tag decodes to bool."""
    assert SubnetHyperparameters._decode_value({"Bool": True}) is True
    assert SubnetHyperparameters._decode_value({"Bool": False}) is False


def test_from_any_decodes_v3_list():
    """Verify v3 Vec<HyperparamEntry> decoding into typed fields."""
    params = SubnetHyperparameters.from_any(_HYPERPARAMS_V3)
    assert params.kappa == 32767
    assert params.tempo == 100
    assert params.max_weight_limit == 65535
    assert params.weights_rate_limit == 100
    assert params.registration_allowed is True
    assert params.liquid_alpha_enabled is False
    assert params.min_burn == 500000
    assert params.activity_cutoff_factor == 13889
    assert isinstance(params.alpha_sigmoid_steepness, float)


def test_from_any_decodes_v2_dict():
    """Verify v2 struct dict decoding into typed fields."""
    params = SubnetHyperparameters.from_any(
        {
            "rho": 10,
            "tempo": 360,
            "max_weights_limit": 49151,
            "registration_allowed": True,
            "alpha_sigmoid_steepness": {"bits": 4294967296000},
        }
    )
    assert params.rho == 10
    assert params.tempo == 360
    assert params.max_weight_limit == 49151
    assert params.registration_allowed is True
    assert isinstance(params.alpha_sigmoid_steepness, float)


def test_from_any_unknown_field_goes_to_spillover():
    """Verify unknown v3 entries are accessible via spillover mapping."""
    params = SubnetHyperparameters.from_any(
        _HYPERPARAMS_V3 + [{"name": "future_param", "value": {"U128": 7}}]
    )
    assert params.future_param == 7
    assert "future_param" in params


def test_from_any_bytes_name():
    """Verify byte-string entry names are decoded."""
    params = SubnetHyperparameters.from_any([{"name": b"tempo", "value": {"U16": 360}}])
    assert params.tempo == 360


def test_subnet_hyperparameters_access():
    """Verify attribute, item, and get access patterns."""
    params = SubnetHyperparameters.from_any(_HYPERPARAMS_V3)
    assert params.tempo == 100
    assert params["tempo"] == 100
    assert params.get("missing", "default") == "default"


def test_from_any_passthrough_existing_instance():
    """Verify from_any returns the same instance when already decoded."""
    original = SubnetHyperparameters.from_any(_HYPERPARAMS_V3)
    assert SubnetHyperparameters.from_any(original) is original
