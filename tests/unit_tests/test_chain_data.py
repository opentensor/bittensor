# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import pytest
import torch

from bittensor.core.chain_data import AxonInfo, DelegateInfo
from bittensor.core.chain_data.utils import ChainDataType

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
    force_legacy_torch_compatible_api,
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
    parameter_dict, expected, test_case, force_legacy_torch_compatible_api
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
