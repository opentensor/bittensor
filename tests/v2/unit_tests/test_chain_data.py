import pytest
import bittensor
import torch
from bittensor.chain_data import AxonInfo, ChainDataType, DelegateInfo, NeuronInfo

SS58_FORMAT = bittensor.__ss58_format__
RAOPERTAO = 10**18


@pytest.mark.parametrize(
    "ip, expected, test_case",
    [
        ("0.0.0.0", False, "ID_is_serving_false"),
        ("127.0.0.1", True, "ID_is_serving_true"),
    ],
)
def test_is_serving(ip, expected, test_case):
    # Arrange
    axon_info = AxonInfo(
        version=1, ip=ip, port=8080, ip_type=4, hotkey="", coldkey="cold"
    )

    # Act
    result = axon_info.is_serving

    # Assert
    assert result == expected, f"Test case: {test_case}"


@pytest.mark.parametrize(
    "ip_type, ip, port, expected, test_case",
    [
        (4, "127.0.0.1", 8080, "/ipv4/127.0.0.1:8080", "ID_ip_str_ipv4"),
        (6, "::1", 8080, "/ipv6/::1:8080", "ID_ip_str_ipv6"),
    ],
)
def test_ip_str(ip_type, ip, port, expected, test_case):
    # Arrange
    axon_info = AxonInfo(
        version=1, ip=ip, port=port, ip_type=ip_type, hotkey="hot", coldkey="cold"
    )

    # Act
    result = axon_info.ip_str()

    # Assert
    assert result == expected, f"Test case: {test_case}"


@pytest.mark.parametrize(
    "other, expected, test_case",
    [
        (None, False, "ID_eq_none"),
        (
            AxonInfo(
                version=1,
                ip="127.0.0.1",
                port=8080,
                ip_type=4,
                hotkey="hot",
                coldkey="cold",
            ),
            True,
            "ID_eq_equal",
        ),
        (
            AxonInfo(
                version=2,
                ip="127.0.0.1",
                port=8080,
                ip_type=4,
                hotkey="hot",
                coldkey="cold",
            ),
            False,
            "ID_eq_diff_version",
        ),
    ],
)
def test_eq(other, expected, test_case):
    # Arrange
    axon_info = AxonInfo(
        version=1, ip="127.0.0.1", port=8080, ip_type=4, hotkey="hot", coldkey="cold"
    )

    # Act
    result = axon_info == other

    # Assert
    assert result == expected, f"Test case: {test_case}"


@pytest.mark.parametrize(
    "axon_info, expected, test_case",
    [
        (
            AxonInfo(
                version=1,
                ip="127.0.0.1",
                port=8080,
                ip_type=4,
                hotkey="hot",
                coldkey="cold",
            ),
            '{"version": 1, "ip": "127.0.0.1", "port": 8080, "ip_type": 4, "hotkey": "hot", "coldkey": "cold", "protocol": 4, "placeholder1": 0, "placeholder2": 0}',
            "ID_to_string",
        ),
    ],
)
def test_to_string(axon_info, expected, test_case):
    # Act
    result = axon_info.to_string()

    # Assert
    assert result == expected, f"Test case: {test_case}"


# Test AxonInfo.from_string method
@pytest.mark.parametrize(
    "string, expected, test_case",
    [
        (
            '{"version": 1, "ip": "127.0.0.1", "port": 8080, "ip_type": 4, "hotkey": "hot", "coldkey": "cold"}',
            AxonInfo(
                version=1,
                ip="127.0.0.1",
                port=8080,
                ip_type=4,
                hotkey="hot",
                coldkey="cold",
            ),
            "ID_from_string_valid",
        ),
        ("invalid_json", AxonInfo(0, "", 0, 0, "", ""), "ID_from_string_invalid_json"),
    ],
)
def test_from_string(string, expected, test_case):
    # Act
    result = AxonInfo.from_string(string)

    # Assert
    assert result == expected, f"Test case: {test_case}"


# Test AxonInfo.from_neuron_info method
@pytest.mark.parametrize(
    "neuron_info, expected, test_case",
    [
        (
            {
                "axon_info": {
                    "version": 1,
                    "ip": 2130706433,
                    "port": 8080,
                    "ip_type": 4,
                },
                "hotkey": "hot",
                "coldkey": "cold",
            },
            AxonInfo(
                version=1,
                ip="127.0.0.1",
                port=8080,
                ip_type=4,
                hotkey="hot",
                coldkey="cold",
            ),
            "ID_from_neuron_info",
        ),
    ],
)
def test_from_neuron_info(neuron_info, expected, test_case):
    # Act
    result = AxonInfo.from_neuron_info(neuron_info)

    # Assert
    assert result == expected, f"Test case: {test_case}"


# Test AxonInfo.to_parameter_dict method
@pytest.mark.parametrize(
    "axon_info, test_case",
    [
        (
            AxonInfo(
                version=1,
                ip="127.0.0.1",
                port=8080,
                ip_type=4,
                hotkey="hot",
                coldkey="cold",
            ),
            "ID_to_parameter_dict",
        ),
    ],
)
def test_to_parameter_dict(axon_info, test_case):
    # Act
    result = axon_info.to_parameter_dict()

    # Assert
    assert isinstance(result, dict)
    for key, value in axon_info.__dict__.items():
        assert key in result
        assert result[key] == value, f"Test case: {test_case}"


@pytest.mark.parametrize(
    "axon_info, test_case",
    [
        (
            AxonInfo(
                version=1,
                ip="127.0.0.1",
                port=8080,
                ip_type=4,
                hotkey="hot",
                coldkey="cold",
            ),
            "ID_to_parameter_dict",
        ),
    ],
)
def test_to_parameter_dict_torch(
    axon_info,
    test_case,
    force_legacy_torch_compat_api,
):
    result = axon_info.to_parameter_dict()

    # Assert
    assert isinstance(result, torch.nn.ParameterDict)
    for key, value in axon_info.__dict__.items():
        assert key in result
        assert result[key] == value, f"Test case: {test_case}"


@pytest.mark.parametrize(
    "parameter_dict, expected, test_case",
    [
        (
            {
                "version": 1,
                "ip": "127.0.0.1",
                "port": 8080,
                "ip_type": 4,
                "hotkey": "hot",
                "coldkey": "cold",
            },
            AxonInfo(
                version=1,
                ip="127.0.0.1",
                port=8080,
                ip_type=4,
                hotkey="hot",
                coldkey="cold",
            ),
            "ID_from_parameter_dict",
        ),
    ],
)
def test_from_parameter_dict(parameter_dict, expected, test_case):
    # Act
    result = AxonInfo.from_parameter_dict(parameter_dict)

    # Assert
    assert result == expected, f"Test case: {test_case}"


@pytest.mark.parametrize(
    "parameter_dict, expected, test_case",
    [
        (
            torch.nn.ParameterDict(
                {
                    "version": 1,
                    "ip": "127.0.0.1",
                    "port": 8080,
                    "ip_type": 4,
                    "hotkey": "hot",
                    "coldkey": "cold",
                }
            ),
            AxonInfo(
                version=1,
                ip="127.0.0.1",
                port=8080,
                ip_type=4,
                hotkey="hot",
                coldkey="cold",
            ),
            "ID_from_parameter_dict",
        ),
    ],
)
def test_from_parameter_dict_torch(
    parameter_dict, expected, test_case, force_legacy_torch_compat_api
):
    # Act
    result = AxonInfo.from_parameter_dict(parameter_dict)

    # Assert
    assert result == expected, f"Test case: {test_case}"


def create_neuron_info_decoded(
    hotkey,
    coldkey,
    stake,
    weights,
    bonds,
    rank,
    emission,
    incentive,
    consensus,
    trust,
    validator_trust,
    dividends,
    uid,
    netuid,
    active,
    last_update,
    validator_permit,
    pruning_score,
    prometheus_info,
    axon_info,
):
    return {
        "hotkey": hotkey,
        "coldkey": coldkey,
        "stake": stake,
        "weights": weights,
        "bonds": bonds,
        "rank": rank,
        "emission": emission,
        "incentive": incentive,
        "consensus": consensus,
        "trust": trust,
        "validator_trust": validator_trust,
        "dividends": dividends,
        "uid": uid,
        "netuid": netuid,
        "active": active,
        "last_update": last_update,
        "validator_permit": validator_permit,
        "pruning_score": pruning_score,
        "prometheus_info": prometheus_info,
        "axon_info": axon_info,
    }


@pytest.mark.parametrize(
    "test_id, neuron_info_decoded,",
    [
        (
            "happy-path-1",
            create_neuron_info_decoded(
                hotkey=b"\x01" * 32,
                coldkey=b"\x02" * 32,
                stake=[(b"\x02" * 32, 1000)],
                weights=[(1, 2)],
                bonds=[(3, 4)],
                rank=100,
                emission=1000,
                incentive=200,
                consensus=300,
                trust=400,
                validator_trust=500,
                dividends=600,
                uid=1,
                netuid=2,
                active=True,
                last_update=1000,
                validator_permit=100,
                pruning_score=1000,
                prometheus_info={
                    "version": 1,
                    "ip": 2130706433,
                    "port": 8080,
                    "ip_type": 4,
                    "block": 100,
                },
                axon_info={
                    "version": 1,
                    "ip": 2130706433,
                    "port": 8080,
                    "ip_type": 4,
                },
            ),
        ),
    ],
)
def test_fix_decoded_values_happy_path(test_id, neuron_info_decoded):
    # Act
    result = NeuronInfo.fix_decoded_values(neuron_info_decoded)

    # Assert
    assert result.hotkey == neuron_info_decoded["hotkey"], f"Test case: {test_id}"
    assert result.coldkey == neuron_info_decoded["coldkey"], f"Test case: {test_id}"
    assert result.stake == neuron_info_decoded["stake"], f"Test case: {test_id}"
    assert result.weights == neuron_info_decoded["weights"], f"Test case: {test_id}"
    assert result.bonds == neuron_info_decoded["bonds"], f"Test case: {test_id}"
    assert result.rank == neuron_info_decoded["rank"], f"Test case: {test_id}"
    assert result.emission == neuron_info_decoded["emission"], f"Test case: {test_id}"
    assert result.incentive == neuron_info_decoded["incentive"], f"Test case: {test_id}"
    assert result.consensus == neuron_info_decoded["consensus"], f"Test case: {test_id}"
    assert result.trust == neuron_info_decoded["trust"], f"Test case: {test_id}"
    assert (
        result.validator_trust == neuron_info_decoded["validator_trust"]
    ), f"Test case: {test_id}"
    assert result.dividends == neuron_info_decoded["dividends"], f"Test case: {test_id}"
    assert result.uid == neuron_info_decoded["uid"], f"Test case: {test_id}"
    assert result.netuid == neuron_info_decoded["netuid"], f"Test case: {test_id}"
    assert result.active == neuron_info_decoded["active"], f"Test case: {test_id}"
    assert (
        result.last_update == neuron_info_decoded["last_update"]
    ), f"Test case: {test_id}"


@pytest.mark.parametrize(
    "test_id, neuron_info_decoded",
    [
        (
            "edge-1",
            create_neuron_info_decoded(
                hotkey=b"\x01" * 32,
                coldkey=b"\x02" * 32,
                stake=[],
                weights=[(1, 2)],
                bonds=[(3, 4)],
                rank=100,
                emission=1000,
                incentive=200,
                consensus=300,
                trust=400,
                validator_trust=500,
                dividends=600,
                uid=1,
                netuid=2,
                active=True,
                last_update=1000,
                validator_permit=100,
                pruning_score=1000,
                prometheus_info={
                    "version": 1,
                    "ip": 2130706433,
                    "port": 8080,
                    "ip_type": 4,
                    "block": 100,
                },
                axon_info={
                    "version": 1,
                    "ip": 2130706433,
                    "port": 8080,
                    "ip_type": 4,
                },
            ),
        ),
    ],
)
def test_fix_decoded_values_edge_cases(test_id, neuron_info_decoded):
    # Act
    result = NeuronInfo.fix_decoded_values(neuron_info_decoded)

    # Assert
    assert result.stake == 0, f"Test case: {test_id}"
    assert result.weights == neuron_info_decoded["weights"], f"Test case: {test_id}"


@pytest.mark.parametrize(
    "test_id, neuron_info_decoded, expected_exception",
    [
        (
            "error-1",
            create_neuron_info_decoded(
                hotkey="not_bytes",
                coldkey=b"\x02" * 32,
                stake=[(b"\x02" * 32, 1000)],
                weights=[(1, 2)],
                bonds=[(3, 4)],
                rank=100,
                emission=1000,
                incentive=200,
                consensus=300,
                trust=400,
                validator_trust=500,
                dividends=600,
                uid=1,
                netuid=2,
                active=True,
                last_update=1000,
                validator_permit=100,
                pruning_score=1000,
                prometheus_info={},
                axon_info={},
            ),
            ValueError,
        ),
    ],
)
def test_fix_decoded_values_error_cases(
    test_id, neuron_info_decoded, expected_exception
):
    # Arrange
    # (Omitted since all input values are provided via test parameters)

    # Act / Assert
    with pytest.raises(expected_exception):
        NeuronInfo.fix_decoded_values(neuron_info_decoded), f"Test case: {test_id}"


@pytest.fixture
def mock_from_scale_encoding(mocker):
    return mocker.patch("bittensor.chain_data.from_scale_encoding")


@pytest.fixture
def mock_fix_decoded_values(mocker):
    return mocker.patch(
        "bittensor.DelegateInfo.fix_decoded_values", side_effect=lambda x: x
    )


@pytest.mark.parametrize(
    "test_id, vec_u8, expected",
    [
        (
            "happy-path-1",
            [1, 2, 3],
            [
                DelegateInfo(
                    hotkey_ss58="hotkey",
                    total_stake=1000,
                    nominators=[
                        "nominator1",
                        "nominator2",
                    ],
                    owner_ss58="owner",
                    take=10.1,
                    validator_permits=[1, 2, 3],
                    registrations=[4, 5, 6],
                    return_per_1000=100,
                    total_daily_return=1000,
                )
            ],
        ),
        (
            "happy-path-2",
            [4, 5, 6],
            [
                DelegateInfo(
                    hotkey_ss58="hotkey",
                    total_stake=1000,
                    nominators=[
                        "nominator1",
                        "nominator2",
                    ],
                    owner_ss58="owner",
                    take=2.1,
                    validator_permits=[1, 2, 3],
                    registrations=[4, 5, 6],
                    return_per_1000=100,
                    total_daily_return=1000,
                )
            ],
        ),
    ],
)
def test_list_from_vec_u8_happy_path(
    mock_from_scale_encoding, mock_fix_decoded_values, test_id, vec_u8, expected
):
    # Arrange
    mock_from_scale_encoding.return_value = expected

    # Act
    result = DelegateInfo.list_from_vec_u8(vec_u8)

    # Assert
    mock_from_scale_encoding.assert_called_once_with(
        vec_u8, ChainDataType.DelegateInfo, is_vec=True
    )
    assert result == expected, f"Failed {test_id}"


@pytest.mark.parametrize(
    "test_id, vec_u8, expected",
    [
        ("edge_empty_list", [], []),
    ],
)
def test_list_from_vec_u8_edge_cases(
    mock_from_scale_encoding, mock_fix_decoded_values, test_id, vec_u8, expected
):
    # Arrange
    mock_from_scale_encoding.return_value = None

    # Act
    result = DelegateInfo.list_from_vec_u8(vec_u8)

    # Assert
    mock_from_scale_encoding.assert_called_once_with(
        vec_u8, ChainDataType.DelegateInfo, is_vec=True
    )
    assert result == expected, f"Failed {test_id}"


@pytest.mark.parametrize(
    "vec_u8, expected_exception",
    [
        ("not_a_list", TypeError),
    ],
)
def test_list_from_vec_u8_error_cases(
    vec_u8,
    expected_exception,
):
    # No Arrange section needed as input values are provided via test parameters

    # Act & Assert
    with pytest.raises(expected_exception):
        _ = DelegateInfo.list_from_vec_u8(vec_u8)
