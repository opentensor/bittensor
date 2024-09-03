# The MIT License (MIT)
# Copyright © 2022 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Standard Lib
import argparse
import unittest.mock as mock
from unittest.mock import MagicMock

# 3rd Party
import pytest

# Application
import bittensor
from bittensor.subtensor import (
    Subtensor,
    _logger,
    Balance,
)
from bittensor.chain_data import SubnetHyperparameters
from bittensor.commands.utils import normalize_hyperparameters
from bittensor import subtensor_module
from bittensor.utils.balance import Balance

U16_MAX = 65535
U64_MAX = 18446744073709551615


def test_serve_axon_with_external_ip_set():
    internal_ip: str = "192.0.2.146"
    external_ip: str = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"

    mock_serve_axon = MagicMock(return_value=True)

    mock_subtensor = MagicMock(spec=bittensor.subtensor, serve_axon=mock_serve_axon)

    mock_add_insecure_port = mock.MagicMock(return_value=None)
    mock_wallet = MagicMock(
        spec=bittensor.wallet,
        coldkey=MagicMock(),
        coldkeypub=MagicMock(
            # mock ss58 address
            ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        ),
        hotkey=MagicMock(
            ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        ),
    )

    mock_config = bittensor.axon.config()
    mock_axon_with_external_ip_set = bittensor.axon(
        wallet=mock_wallet,
        ip=internal_ip,
        external_ip=external_ip,
        config=mock_config,
    )

    mock_subtensor.serve_axon(
        netuid=-1,
        axon=mock_axon_with_external_ip_set,
    )

    mock_serve_axon.assert_called_once()

    # verify that the axon is served to the network with the external ip
    _, kwargs = mock_serve_axon.call_args
    axon_info = kwargs["axon"].info()
    assert axon_info.ip == external_ip


def test_serve_axon_with_external_port_set():
    external_ip: str = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"

    internal_port: int = 1234
    external_port: int = 5678

    mock_serve = MagicMock(return_value=True)

    mock_serve_axon = MagicMock(return_value=True)

    mock_subtensor = MagicMock(
        spec=bittensor.subtensor,
        serve=mock_serve,
        serve_axon=mock_serve_axon,
    )

    mock_wallet = MagicMock(
        spec=bittensor.wallet,
        coldkey=MagicMock(),
        coldkeypub=MagicMock(
            # mock ss58 address
            ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        ),
        hotkey=MagicMock(
            ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        ),
    )

    mock_config = bittensor.axon.config()

    mock_axon_with_external_port_set = bittensor.axon(
        wallet=mock_wallet,
        port=internal_port,
        external_port=external_port,
        config=mock_config,
    )

    with mock.patch(
        "bittensor.utils.networking.get_external_ip", return_value=external_ip
    ):
        # mock the get_external_ip function to return the external ip
        mock_subtensor.serve_axon(
            netuid=-1,
            axon=mock_axon_with_external_port_set,
        )

    mock_serve_axon.assert_called_once()
    # verify that the axon is served to the network with the external port
    _, kwargs = mock_serve_axon.call_args
    axon_info = kwargs["axon"].info()
    assert axon_info.port == external_port


class ExitEarly(Exception):
    """Mock exception to exit early from the called code"""

    pass


def test_stake_multiple():
    mock_amount: bittensor.Balance = bittensor.Balance.from_tao(1.0)

    mock_wallet = MagicMock(
        spec=bittensor.wallet,
        coldkey=MagicMock(),
        coldkeypub=MagicMock(
            # mock ss58 address
            ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        ),
        hotkey=MagicMock(
            ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        ),
    )

    mock_hotkey_ss58s = ["5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"]

    mock_amounts = [mock_amount]  # more than 1000 RAO

    mock_neuron = MagicMock(
        is_null=False,
    )

    mock_do_stake = MagicMock(side_effect=ExitEarly)

    mock_subtensor = MagicMock(
        spec=bittensor.subtensor,
        network="mock_net",
        get_balance=MagicMock(
            return_value=bittensor.Balance.from_tao(mock_amount.tao + 20.0)
        ),  # enough balance to stake
        get_neuron_for_pubkey_and_subnet=MagicMock(return_value=mock_neuron),
        _do_stake=mock_do_stake,
    )

    with pytest.raises(ExitEarly):
        bittensor.subtensor.add_stake_multiple(
            mock_subtensor,
            wallet=mock_wallet,
            hotkey_ss58s=mock_hotkey_ss58s,
            amounts=mock_amounts,
        )

        mock_do_stake.assert_called_once()
        # args, kwargs
        _, kwargs = mock_do_stake.call_args

        assert kwargs["amount"] == pytest.approx(
            mock_amount.rao, rel=1e9
        )  # delta of 1.0 TAO


@pytest.mark.parametrize(
    "test_id, expected_output",
    [
        # Happy path test
        (
            "happy_path_default",
            "Create and return a new object.  See help(type) for accurate signature.",
        ),
    ],
)
def test_help(test_id, expected_output, capsys):
    # Act
    Subtensor.help()

    # Assert
    captured = capsys.readouterr()
    assert expected_output in captured.out, f"Test case {test_id} failed"


@pytest.fixture
def parser():
    return argparse.ArgumentParser()


# Mocking argparse.ArgumentParser.add_argument method to simulate ArgumentError
def test_argument_error_handling(monkeypatch, parser):
    def mock_add_argument(*args, **kwargs):
        raise argparse.ArgumentError(None, "message")

    monkeypatch.setattr(argparse.ArgumentParser, "add_argument", mock_add_argument)
    # No exception should be raised
    Subtensor.add_args(parser)


@pytest.mark.parametrize(
    "network, expected_network, expected_endpoint",
    [
        # Happy path tests
        ("finney", "finney", bittensor.__finney_entrypoint__),
        ("local", "local", bittensor.__local_entrypoint__),
        ("test", "test", bittensor.__finney_test_entrypoint__),
        ("archive", "archive", bittensor.__archive_entrypoint__),
        # Endpoint override tests
        (
            bittensor.__finney_entrypoint__,
            "finney",
            bittensor.__finney_entrypoint__,
        ),
        (
            "entrypoint-finney.opentensor.ai",
            "finney",
            bittensor.__finney_entrypoint__,
        ),
        (
            bittensor.__finney_test_entrypoint__,
            "test",
            bittensor.__finney_test_entrypoint__,
        ),
        (
            "test.finney.opentensor.ai",
            "test",
            bittensor.__finney_test_entrypoint__,
        ),
        (
            bittensor.__archive_entrypoint__,
            "archive",
            bittensor.__archive_entrypoint__,
        ),
        (
            "archive.chain.opentensor.ai",
            "archive",
            bittensor.__archive_entrypoint__,
        ),
        ("127.0.0.1", "local", "127.0.0.1"),
        ("localhost", "local", "localhost"),
        # Edge cases
        (None, None, None),
        ("unknown", "unknown network", "unknown"),
    ],
)
def test_determine_chain_endpoint_and_network(
    network, expected_network, expected_endpoint
):
    # Act
    result_network, result_endpoint = Subtensor.determine_chain_endpoint_and_network(
        network
    )

    # Assert
    assert result_network == expected_network
    assert result_endpoint == expected_endpoint


# Subtensor().get_error_info_by_index tests
@pytest.fixture
def substrate():
    class MockSubstrate:
        pass

    return MockSubstrate()


@pytest.fixture
def subtensor(substrate):
    mock.patch.object(
        subtensor_module,
        "get_subtensor_errors",
        return_value={
            "1": ("ErrorOne", "Description one"),
            "2": ("ErrorTwo", "Description two"),
        },
    ).start()
    return Subtensor()


def test_get_error_info_by_index_known_error(subtensor):
    name, description = subtensor.get_error_info_by_index(1)
    assert name == "ErrorOne"
    assert description == "Description one"


@pytest.fixture
def mock_logger():
    with mock.patch.object(_logger, "warning") as mock_warning:
        yield mock_warning


def test_get_error_info_by_index_unknown_error(subtensor, mock_logger):
    fake_index = 999
    name, description = subtensor.get_error_info_by_index(fake_index)
    assert name == "Unknown Error"
    assert description == ""
    mock_logger.assert_called_once_with(
        f"Subtensor returned an error with an unknown index: {fake_index}"
    )


# Subtensor()._get_hyperparameter tests
def test_hyperparameter_subnet_does_not_exist(subtensor, mocker):
    """Tests when the subnet does not exist."""
    subtensor.subnet_exists = mocker.MagicMock(return_value=False)
    assert subtensor._get_hyperparameter("Difficulty", 1, None) is None
    subtensor.subnet_exists.assert_called_once_with(1, None)


def test_hyperparameter_result_is_none(subtensor, mocker):
    """Tests when query_subtensor returns None."""
    subtensor.subnet_exists = mocker.MagicMock(return_value=True)
    subtensor.query_subtensor = mocker.MagicMock(return_value=None)
    assert subtensor._get_hyperparameter("Difficulty", 1, None) is None
    subtensor.subnet_exists.assert_called_once_with(1, None)
    subtensor.query_subtensor.assert_called_once_with("Difficulty", None, [1])


def test_hyperparameter_result_has_no_value(subtensor, mocker):
    """Test when the result has no 'value' attribute."""

    subtensor.subnet_exists = mocker.MagicMock(return_value=True)
    subtensor.query_subtensor = mocker.MagicMock(return_value=None)
    assert subtensor._get_hyperparameter("Difficulty", 1, None) is None
    subtensor.subnet_exists.assert_called_once_with(1, None)
    subtensor.query_subtensor.assert_called_once_with("Difficulty", None, [1])


def test_hyperparameter_success_int(subtensor, mocker):
    """Test when query_subtensor returns an integer value."""
    subtensor.subnet_exists = mocker.MagicMock(return_value=True)
    subtensor.query_subtensor = mocker.MagicMock(
        return_value=mocker.MagicMock(value=100)
    )
    assert subtensor._get_hyperparameter("Difficulty", 1, None) == 100
    subtensor.subnet_exists.assert_called_once_with(1, None)
    subtensor.query_subtensor.assert_called_once_with("Difficulty", None, [1])


def test_hyperparameter_success_float(subtensor, mocker):
    """Test when query_subtensor returns a float value."""
    subtensor.subnet_exists = mocker.MagicMock(return_value=True)
    subtensor.query_subtensor = mocker.MagicMock(
        return_value=mocker.MagicMock(value=0.5)
    )
    assert subtensor._get_hyperparameter("Difficulty", 1, None) == 0.5
    subtensor.subnet_exists.assert_called_once_with(1, None)
    subtensor.query_subtensor.assert_called_once_with("Difficulty", None, [1])


# Tests Hyper parameter calls
@pytest.mark.parametrize(
    "method, param_name, value, expected_result_type",
    [
        ("rho", "Rho", 1, int),
        ("kappa", "Kappa", 1.0, float),
        ("difficulty", "Difficulty", 1, int),
        ("recycle", "Burn", 1, Balance),
        ("immunity_period", "ImmunityPeriod", 1, int),
        ("validator_batch_size", "ValidatorBatchSize", 1, int),
        ("validator_prune_len", "ValidatorPruneLen", 1, int),
        ("validator_logits_divergence", "ValidatorLogitsDivergence", 1.0, float),
        ("validator_sequence_length", "ValidatorSequenceLength", 1, int),
        ("validator_epochs_per_reset", "ValidatorEpochsPerReset", 1, int),
        ("validator_epoch_length", "ValidatorEpochLen", 1, int),
        ("validator_exclude_quantile", "ValidatorExcludeQuantile", 1.0, float),
        ("max_allowed_validators", "MaxAllowedValidators", 1, int),
        ("min_allowed_weights", "MinAllowedWeights", 1, int),
        ("max_weight_limit", "MaxWeightsLimit", 1, float),
        ("adjustment_alpha", "AdjustmentAlpha", 1, float),
        ("bonds_moving_avg", "BondsMovingAverage", 1, float),
        ("scaling_law_power", "ScalingLawPower", 1, float),
        ("synergy_scaling_law_power", "SynergyScalingLawPower", 1, float),
        ("subnetwork_n", "SubnetworkN", 1, int),
        ("max_n", "MaxAllowedUids", 1, int),
        ("blocks_since_epoch", "BlocksSinceEpoch", 1, int),
        ("tempo", "Tempo", 1, int),
    ],
)
def test_hyper_parameter_success_calls(
    subtensor, mocker, method, param_name, value, expected_result_type
):
    """
    Tests various hyperparameter methods to ensure they correctly fetch their respective hyperparameters and return the
    expected values.
    """
    # Prep
    subtensor._get_hyperparameter = mocker.MagicMock(return_value=value)

    spy_u16_normalized_float = mocker.spy(subtensor_module, "U16_NORMALIZED_FLOAT")
    spy_u64_normalized_float = mocker.spy(subtensor_module, "U64_NORMALIZED_FLOAT")
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    subtensor_method = getattr(subtensor, method)
    result = subtensor_method(netuid=7, block=707)

    # Assertions
    subtensor._get_hyperparameter.assert_called_once_with(
        block=707, netuid=7, param_name=param_name
    )
    # if we change the methods logic in the future we have to be make sure the returned type is correct
    assert isinstance(result, expected_result_type)

    # Special cases
    if method in [
        "kappa",
        "validator_logits_divergence",
        "validator_exclude_quantile",
        "max_weight_limit",
    ]:
        spy_u16_normalized_float.assert_called_once()

    if method in ["adjustment_alpha", "bonds_moving_avg"]:
        spy_u64_normalized_float.assert_called_once()

    if method in ["recycle"]:
        spy_balance_from_rao.assert_called_once()


def test_blocks_since_last_update_success_calls(subtensor, mocker):
    """Tests the weights_rate_limit method to ensure it correctly fetches the LastUpdate hyperparameter."""
    # Prep
    uid = 7
    mocked_current_block = 2
    mocked_result = {uid: 1}
    subtensor._get_hyperparameter = mocker.MagicMock(return_value=mocked_result)
    subtensor.get_current_block = mocker.MagicMock(return_value=mocked_current_block)

    # Call
    result = subtensor.blocks_since_last_update(netuid=7, uid=uid)

    # Assertions
    subtensor.get_current_block.assert_called_once()
    subtensor._get_hyperparameter.assert_called_once_with(
        param_name="LastUpdate", netuid=7
    )
    assert result == 1
    # if we change the methods logic in the future we have to be make sure the returned type is correct
    assert isinstance(result, int)


def test_weights_rate_limit_success_calls(subtensor, mocker):
    """Tests the weights_rate_limit method to ensure it correctly fetches the WeightsSetRateLimit hyperparameter."""
    # Prep
    subtensor._get_hyperparameter = mocker.MagicMock(return_value=5)

    # Call
    result = subtensor.weights_rate_limit(netuid=7)

    # Assertions
    subtensor._get_hyperparameter.assert_called_once_with(
        param_name="WeightsSetRateLimit", netuid=7
    )
    # if we change the methods logic in the future we have to be make sure the returned type is correct
    assert isinstance(result, int)


@pytest.fixture
def sample_hyperparameters():
    return MagicMock(spec=SubnetHyperparameters)


def get_normalized_value(normalized_data, param_name):
    return next(
        (
            norm_value
            for p_name, _, norm_value in normalized_data
            if p_name == param_name
        ),
        None,
    )


@pytest.mark.parametrize(
    "param_name, max_value, mid_value, zero_value, is_balance",
    [
        ("adjustment_alpha", U64_MAX, U64_MAX / 2, 0, False),
        ("max_weight_limit", U16_MAX, U16_MAX / 2, 0, False),
        ("difficulty", U64_MAX, U64_MAX / 2, 0, False),
        ("min_difficulty", U64_MAX, U64_MAX / 2, 0, False),
        ("max_difficulty", U64_MAX, U64_MAX / 2, 0, False),
        ("bonds_moving_avg", U64_MAX, U64_MAX / 2, 0, False),
        ("min_burn", 10000000000, 5000000000, 0, True),  # These are in rao
        ("max_burn", 20000000000, 10000000000, 0, True),
    ],
    ids=[
        "adjustment-alpha",
        "max_weight_limit",
        "difficulty",
        "min_difficulty",
        "max_difficulty",
        "bonds_moving_avg",
        "min_burn",
        "max_burn",
    ],
)
def test_hyperparameter_normalization(
    sample_hyperparameters, param_name, max_value, mid_value, zero_value, is_balance
):
    setattr(sample_hyperparameters, param_name, mid_value)
    normalized = normalize_hyperparameters(sample_hyperparameters)
    norm_value = get_normalized_value(normalized, param_name)

    # Mid-value test
    if is_balance:
        numeric_value = float(str(norm_value).lstrip(bittensor.__tao_symbol__))
        expected_tao = mid_value / 1e9
        assert (
            numeric_value == expected_tao
        ), f"Mismatch in tao value for {param_name} at mid value"
    else:
        assert float(norm_value) == 0.5, f"Failed mid-point test for {param_name}"

    # Max-value test
    setattr(sample_hyperparameters, param_name, max_value)
    normalized = normalize_hyperparameters(sample_hyperparameters)
    norm_value = get_normalized_value(normalized, param_name)

    if is_balance:
        numeric_value = float(str(norm_value).lstrip(bittensor.__tao_symbol__))
        expected_tao = max_value / 1e9
        assert (
            numeric_value == expected_tao
        ), f"Mismatch in tao value for {param_name} at max value"
    else:
        assert float(norm_value) == 1.0, f"Failed max value test for {param_name}"

    # Zero-value test
    setattr(sample_hyperparameters, param_name, zero_value)
    normalized = normalize_hyperparameters(sample_hyperparameters)
    norm_value = get_normalized_value(normalized, param_name)

    if is_balance:
        numeric_value = float(str(norm_value).lstrip(bittensor.__tao_symbol__))
        expected_tao = zero_value / 1e9
        assert (
            numeric_value == expected_tao
        ), f"Mismatch in tao value for {param_name} at zero value"
    else:
        assert float(norm_value) == 0.0, f"Failed zero value test for {param_name}"


###########################
# Account functions tests #
###########################


# `get_total_stake_for_hotkey` tests
def test_get_total_stake_for_hotkey_success(subtensor, mocker):
    """Tests successful retrieval of total stake for hotkey."""
    # Prep
    subtensor.query_subtensor = mocker.MagicMock(return_value=mocker.MagicMock(value=1))
    fake_ss58_address = "12bzRJfh7arnnfPPUZHeJUaE62QLEwhK48QnH9LXeK2m1iZU"
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.get_total_stake_for_hotkey(ss58_address=fake_ss58_address)

    # Assertions
    subtensor.query_subtensor.assert_called_once_with(
        "TotalHotkeyStake", None, [fake_ss58_address]
    )
    spy_balance_from_rao.assert_called_once()
    # if we change the methods logic in the future we have to be make sure the returned type is correct
    assert isinstance(result, Balance)


def test_get_total_stake_for_hotkey_not_result(subtensor, mocker):
    """Tests retrieval of total stake for hotkey when no result is returned."""
    # Prep
    subtensor.query_subtensor = mocker.MagicMock(return_value=None)
    fake_ss58_address = "12bzRJfh7arnnfPPUZHeJUaE62QLEwhK48QnH9LXeK2m1iZU"
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.get_total_stake_for_hotkey(ss58_address=fake_ss58_address)

    # Assertions
    subtensor.query_subtensor.assert_called_once_with(
        "TotalHotkeyStake", None, [fake_ss58_address]
    )
    spy_balance_from_rao.assert_not_called()
    # if we change the methods logic in the future we have to be make sure the returned type is correct
    assert isinstance(result, type(None))


def test_get_total_stake_for_hotkey_not_value(subtensor, mocker):
    """Tests retrieval of total stake for hotkey when no value attribute is present."""
    # Prep
    subtensor.query_subtensor = mocker.MagicMock(return_value=object)
    fake_ss58_address = "12bzRJfh7arnnfPPUZHeJUaE62QLEwhK48QnH9LXeK2m1iZU"
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.get_total_stake_for_hotkey(ss58_address=fake_ss58_address)

    # Assertions
    subtensor.query_subtensor.assert_called_once_with(
        "TotalHotkeyStake", None, [fake_ss58_address]
    )
    spy_balance_from_rao.assert_not_called()
    # if we change the methods logic in the future we have to be make sure the returned type is correct
    assert isinstance(subtensor.query_subtensor.return_value, object)
    assert not hasattr(result, "value")


# `get_total_stake_for_coldkey` tests
def test_get_total_stake_for_coldkey_success(subtensor, mocker):
    """Tests successful retrieval of total stake for coldkey."""
    # Prep
    subtensor.query_subtensor = mocker.MagicMock(return_value=mocker.MagicMock(value=1))
    fake_ss58_address = "12bzRJfh7arnnfPPUZHeJUaE62QLEwhK48QnH9LXeK2m1iZU"
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.get_total_stake_for_coldkey(ss58_address=fake_ss58_address)

    # Assertions
    subtensor.query_subtensor.assert_called_once_with(
        "TotalColdkeyStake", None, [fake_ss58_address]
    )
    spy_balance_from_rao.assert_called_once()
    # if we change the methods logic in the future we have to be make sure the returned type is correct
    assert isinstance(result, Balance)


def test_get_total_stake_for_coldkey_not_result(subtensor, mocker):
    """Tests retrieval of total stake for coldkey when no result is returned."""
    # Prep
    subtensor.query_subtensor = mocker.MagicMock(return_value=None)
    fake_ss58_address = "12bzRJfh7arnnfPPUZHeJUaE62QLEwhK48QnH9LXeK2m1iZU"
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.get_total_stake_for_coldkey(ss58_address=fake_ss58_address)

    # Assertions
    subtensor.query_subtensor.assert_called_once_with(
        "TotalColdkeyStake", None, [fake_ss58_address]
    )
    spy_balance_from_rao.assert_not_called()
    # if we change the methods logic in the future we have to be make sure the returned type is correct
    assert isinstance(result, type(None))


def test_get_total_stake_for_coldkey_not_value(subtensor, mocker):
    """Tests retrieval of total stake for coldkey when no value attribute is present."""
    # Prep
    subtensor.query_subtensor = mocker.MagicMock(return_value=object)
    fake_ss58_address = "12bzRJfh7arnnfPPUZHeJUaE62QLEwhK48QnH9LXeK2m1iZU"
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.get_total_stake_for_coldkey(ss58_address=fake_ss58_address)

    # Assertions
    subtensor.query_subtensor.assert_called_once_with(
        "TotalColdkeyStake", None, [fake_ss58_address]
    )
    spy_balance_from_rao.assert_not_called()
    # if we change the methods logic in the future we have to be make sure the returned type is correct
    assert isinstance(subtensor.query_subtensor.return_value, object)
    assert not hasattr(result, "value")


# `get_stake` tests
def test_get_stake_returns_correct_data(mocker, subtensor):
    """Tests that get_stake returns correct data."""
    # Prep
    hotkey_ss58 = "test_hotkey"
    block = 123
    expected_query_result = [
        (mocker.MagicMock(value="coldkey1"), mocker.MagicMock(value=100)),
        (mocker.MagicMock(value="coldkey2"), mocker.MagicMock(value=200)),
    ]
    mocker.patch.object(
        subtensor, "query_map_subtensor", return_value=expected_query_result
    )

    # Call
    result = subtensor.get_stake(hotkey_ss58, block)

    # Assertion
    assert result == [
        ("coldkey1", Balance.from_rao(100)),
        ("coldkey2", Balance.from_rao(200)),
    ]
    subtensor.query_map_subtensor.assert_called_once_with("Stake", block, [hotkey_ss58])


def test_get_stake_no_block(mocker, subtensor):
    """Tests get_stake with no block specified."""
    # Prep
    hotkey_ss58 = "test_hotkey"
    expected_query_result = [
        (MagicMock(value="coldkey1"), MagicMock(value=100)),
    ]
    mocker.patch.object(
        subtensor, "query_map_subtensor", return_value=expected_query_result
    )

    # Call
    result = subtensor.get_stake(hotkey_ss58)

    # Assertion
    assert result == [("coldkey1", Balance.from_rao(100))]
    subtensor.query_map_subtensor.assert_called_once_with("Stake", None, [hotkey_ss58])


def test_get_stake_empty_result(mocker, subtensor):
    """Tests get_stake with an empty result."""
    # Prep
    hotkey_ss58 = "test_hotkey"
    block = 123
    expected_query_result = []
    mocker.patch.object(
        subtensor, "query_map_subtensor", return_value=expected_query_result
    )

    # Call
    result = subtensor.get_stake(hotkey_ss58, block)

    # Assertion
    assert result == []
    subtensor.query_map_subtensor.assert_called_once_with("Stake", block, [hotkey_ss58])


# `does_hotkey_exist` tests
def test_does_hotkey_exist_true(mocker, subtensor):
    """Test does_hotkey_exist returns True when hotkey exists and is valid."""
    # Prep
    hotkey_ss58 = "test_hotkey"
    block = 123
    mock_result = mocker.MagicMock(value="valid_coldkey")
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.does_hotkey_exist(hotkey_ss58, block)

    # Assertions
    assert result is True
    subtensor.query_subtensor.assert_called_once_with("Owner", block, [hotkey_ss58])


def test_does_hotkey_exist_false_special_value(mocker, subtensor):
    """Test does_hotkey_exist returns False when result value is the special value."""
    # Prep
    hotkey_ss58 = "test_hotkey"
    block = 123
    special_value = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
    mock_result = MagicMock(value=special_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.does_hotkey_exist(hotkey_ss58, block)

    # Assertions
    assert result is False
    subtensor.query_subtensor.assert_called_once_with("Owner", block, [hotkey_ss58])


def test_does_hotkey_exist_false_no_value(mocker, subtensor):
    """Test does_hotkey_exist returns False when result has no value attribute."""
    # Prep
    hotkey_ss58 = "test_hotkey"
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.does_hotkey_exist(hotkey_ss58, block)

    # Assertions
    assert result is False
    subtensor.query_subtensor.assert_called_once_with("Owner", block, [hotkey_ss58])


def test_does_hotkey_exist_false_no_result(mocker, subtensor):
    """Test does_hotkey_exist returns False when query_subtensor returns None."""
    # Prep
    hotkey_ss58 = "test_hotkey"
    block = 123
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)

    # Call
    result = subtensor.does_hotkey_exist(hotkey_ss58, block)

    # Assertions
    assert result is False
    subtensor.query_subtensor.assert_called_once_with("Owner", block, [hotkey_ss58])


def test_does_hotkey_exist_no_block(mocker, subtensor):
    """Test does_hotkey_exist with no block specified."""
    # Prep
    hotkey_ss58 = "test_hotkey"
    mock_result = mocker.MagicMock(value="valid_coldkey")
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.does_hotkey_exist(hotkey_ss58)

    # Assertions
    assert result is True
    subtensor.query_subtensor.assert_called_once_with("Owner", None, [hotkey_ss58])


# `get_hotkey_owner` tests
def test_get_hotkey_owner_exists(mocker, subtensor):
    """Test get_hotkey_owner when the hotkey exists."""
    # Prep
    hotkey_ss58 = "test_hotkey"
    block = 123
    expected_owner = "coldkey_owner"
    mock_result = mocker.MagicMock(value=expected_owner)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)
    mocker.patch.object(subtensor, "does_hotkey_exist", return_value=True)

    # Call
    result = subtensor.get_hotkey_owner(hotkey_ss58, block)

    # Assertions
    assert result == expected_owner
    subtensor.query_subtensor.assert_called_once_with("Owner", block, [hotkey_ss58])
    subtensor.does_hotkey_exist.assert_called_once_with(hotkey_ss58, block)


def test_get_hotkey_owner_does_not_exist(mocker, subtensor):
    """Test get_hotkey_owner when the hotkey does not exist."""
    # Prep
    hotkey_ss58 = "test_hotkey"
    block = 123
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)
    mocker.patch.object(subtensor, "does_hotkey_exist", return_value=False)

    # Call
    result = subtensor.get_hotkey_owner(hotkey_ss58, block)

    # Assertions
    assert result is None
    subtensor.query_subtensor.assert_called_once_with("Owner", block, [hotkey_ss58])
    subtensor.does_hotkey_exist.assert_not_called()


def test_get_hotkey_owner_no_block(mocker, subtensor):
    """Test get_hotkey_owner with no block specified."""
    # Prep
    hotkey_ss58 = "test_hotkey"
    expected_owner = "coldkey_owner"
    mock_result = mocker.MagicMock(value=expected_owner)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)
    mocker.patch.object(subtensor, "does_hotkey_exist", return_value=True)

    # Call
    result = subtensor.get_hotkey_owner(hotkey_ss58)

    # Assertions
    assert result == expected_owner
    subtensor.query_subtensor.assert_called_once_with("Owner", None, [hotkey_ss58])
    subtensor.does_hotkey_exist.assert_called_once_with(hotkey_ss58, None)


def test_get_hotkey_owner_no_value_attribute(mocker, subtensor):
    """Test get_hotkey_owner when the result has no value attribute."""
    # Prep
    hotkey_ss58 = "test_hotkey"
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)
    mocker.patch.object(subtensor, "does_hotkey_exist", return_value=True)

    # Call
    result = subtensor.get_hotkey_owner(hotkey_ss58, block)

    # Assertions
    assert result is None
    subtensor.query_subtensor.assert_called_once_with("Owner", block, [hotkey_ss58])
    subtensor.does_hotkey_exist.assert_not_called()


# `get_axon_info` tests
def test_get_axon_info_success(mocker, subtensor):
    """Test get_axon_info returns correct data when axon information is found."""
    # Prep
    netuid = 1
    hotkey_ss58 = "test_hotkey"
    block = 123
    mock_result = mocker.MagicMock(
        value={
            "ip": "192.168.1.1",
            "ip_type": 4,
            "port": 8080,
            "protocol": "tcp",
            "version": "1.0",
            "placeholder1": "data1",
            "placeholder2": "data2",
        }
    )
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_axon_info(netuid, hotkey_ss58, block)

    # Asserts
    assert result is not None
    assert result.ip == "192.168.1.1"
    assert result.ip_type == 4
    assert result.port == 8080
    assert result.protocol == "tcp"
    assert result.version == "1.0"
    assert result.placeholder1 == "data1"
    assert result.placeholder2 == "data2"
    assert result.hotkey == hotkey_ss58
    assert result.coldkey == ""
    subtensor.query_subtensor.assert_called_once_with(
        "Axons", block, [netuid, hotkey_ss58]
    )


def test_get_axon_info_no_data(mocker, subtensor):
    """Test get_axon_info returns None when no axon information is found."""
    # Prep
    netuid = 1
    hotkey_ss58 = "test_hotkey"
    block = 123
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)

    # Call
    result = subtensor.get_axon_info(netuid, hotkey_ss58, block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with(
        "Axons", block, [netuid, hotkey_ss58]
    )


def test_get_axon_info_no_value_attribute(mocker, subtensor):
    """Test get_axon_info returns None when result has no value attribute."""
    # Prep
    netuid = 1
    hotkey_ss58 = "test_hotkey"
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_axon_info(netuid, hotkey_ss58, block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with(
        "Axons", block, [netuid, hotkey_ss58]
    )


def test_get_axon_info_no_block(mocker, subtensor):
    """Test get_axon_info with no block specified."""
    # Prep
    netuid = 1
    hotkey_ss58 = "test_hotkey"
    mock_result = mocker.MagicMock(
        value={
            "ip": 3232235777,  # 192.168.1.1
            "ip_type": 4,
            "port": 8080,
            "protocol": "tcp",
            "version": "1.0",
            "placeholder1": "data1",
            "placeholder2": "data2",
        }
    )
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_axon_info(netuid, hotkey_ss58)

    # Asserts
    assert result is not None
    assert result.ip == "192.168.1.1"
    assert result.ip_type == 4
    assert result.port == 8080
    assert result.protocol == "tcp"
    assert result.version == "1.0"
    assert result.placeholder1 == "data1"
    assert result.placeholder2 == "data2"
    assert result.hotkey == hotkey_ss58
    assert result.coldkey == ""
    subtensor.query_subtensor.assert_called_once_with(
        "Axons", None, [netuid, hotkey_ss58]
    )


# get_prometheus_info tests
def test_get_prometheus_info_success(mocker, subtensor):
    """Test get_prometheus_info returns correct data when information is found."""
    # Prep
    netuid = 1
    hotkey_ss58 = "test_hotkey"
    block = 123
    mock_result = mocker.MagicMock(
        value={
            "ip": 3232235777,  # 192.168.1.1
            "ip_type": 4,
            "port": 9090,
            "version": "1.0",
            "block": 1000,
        }
    )
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_prometheus_info(netuid, hotkey_ss58, block)

    # Asserts
    assert result is not None
    assert result.ip == "192.168.1.1"
    assert result.ip_type == 4
    assert result.port == 9090
    assert result.version == "1.0"
    assert result.block == 1000
    subtensor.query_subtensor.assert_called_once_with(
        "Prometheus", block, [netuid, hotkey_ss58]
    )


def test_get_prometheus_info_no_data(mocker, subtensor):
    """Test get_prometheus_info returns None when no information is found."""
    # Prep
    netuid = 1
    hotkey_ss58 = "test_hotkey"
    block = 123
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)

    # Call
    result = subtensor.get_prometheus_info(netuid, hotkey_ss58, block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with(
        "Prometheus", block, [netuid, hotkey_ss58]
    )


def test_get_prometheus_info_no_value_attribute(mocker, subtensor):
    """Test get_prometheus_info returns None when result has no value attribute."""
    # Prep
    netuid = 1
    hotkey_ss58 = "test_hotkey"
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_prometheus_info(netuid, hotkey_ss58, block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with(
        "Prometheus", block, [netuid, hotkey_ss58]
    )


def test_get_prometheus_info_no_block(mocker, subtensor):
    """Test get_prometheus_info with no block specified."""
    # Prep
    netuid = 1
    hotkey_ss58 = "test_hotkey"
    mock_result = MagicMock(
        value={
            "ip": "192.168.1.1",
            "ip_type": 4,
            "port": 9090,
            "version": "1.0",
            "block": 1000,
        }
    )
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_prometheus_info(netuid, hotkey_ss58)

    # Asserts
    assert result is not None
    assert result.ip == "192.168.1.1"
    assert result.ip_type == 4
    assert result.port == 9090
    assert result.version == "1.0"
    assert result.block == 1000
    subtensor.query_subtensor.assert_called_once_with(
        "Prometheus", None, [netuid, hotkey_ss58]
    )


###########################
# Global Parameters tests #
###########################


# `block` property test
def test_block_property(mocker, subtensor):
    """Test block property returns the correct block number."""
    expected_block = 123
    mocker.patch.object(subtensor, "get_current_block", return_value=expected_block)

    result = subtensor.block

    assert result == expected_block
    subtensor.get_current_block.assert_called_once()


# `total_issuance` tests
def test_total_issuance_success(mocker, subtensor):
    """Test total_issuance returns correct data when issuance information is found."""
    # Prep
    block = 123
    issuance_value = 1000
    mock_result = mocker.MagicMock(value=issuance_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.total_issuance(block)

    # Asserts
    assert result is not None
    subtensor.query_subtensor.assert_called_once_with("TotalIssuance", block)
    spy_balance_from_rao.assert_called_once_with(
        subtensor.query_subtensor.return_value.value
    )


def test_total_issuance_no_data(mocker, subtensor):
    """Test total_issuance returns None when no issuance information is found."""
    # Prep
    block = 123
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.total_issuance(block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with("TotalIssuance", block)
    spy_balance_from_rao.assert_not_called()


def test_total_issuance_no_value_attribute(mocker, subtensor):
    """Test total_issuance returns None when result has no value attribute."""
    # Prep
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.total_issuance(block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with("TotalIssuance", block)
    spy_balance_from_rao.assert_not_called()


def test_total_issuance_no_block(mocker, subtensor):
    """Test total_issuance with no block specified."""
    # Prep
    issuance_value = 1000
    mock_result = mocker.MagicMock(value=issuance_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.total_issuance()

    # Asserts
    assert result is not None
    subtensor.query_subtensor.assert_called_once_with("TotalIssuance", None)
    spy_balance_from_rao.assert_called_once_with(
        subtensor.query_subtensor.return_value.value
    )


# `total_stake` method tests
def test_total_stake_success(mocker, subtensor):
    """Test total_stake returns correct data when stake information is found."""
    # Prep
    block = 123
    stake_value = 5000
    mock_result = mocker.MagicMock(value=stake_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.total_stake(block)

    # Asserts
    assert result is not None
    subtensor.query_subtensor.assert_called_once_with("TotalStake", block)
    spy_balance_from_rao.assert_called_once_with(
        subtensor.query_subtensor.return_value.value
    )


def test_total_stake_no_data(mocker, subtensor):
    """Test total_stake returns None when no stake information is found."""
    # Prep
    block = 123
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.total_stake(block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with("TotalStake", block)
    spy_balance_from_rao.assert_not_called()


def test_total_stake_no_value_attribute(mocker, subtensor):
    """Test total_stake returns None when result has no value attribute."""
    # Prep
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.total_stake(block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with("TotalStake", block)
    spy_balance_from_rao.assert_not_called()


def test_total_stake_no_block(mocker, subtensor):
    """Test total_stake with no block specified."""
    # Prep
    stake_value = 5000
    mock_result = mocker.MagicMock(value=stake_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.total_stake()

    # Asserts
    assert result is not None
    subtensor.query_subtensor.assert_called_once_with("TotalStake", None)
    (
        spy_balance_from_rao.assert_called_once_with(
            subtensor.query_subtensor.return_value.value
        ),
    )


# `serving_rate_limit` method tests
def test_serving_rate_limit_success(mocker, subtensor):
    """Test serving_rate_limit returns correct data when rate limit information is found."""
    # Prep
    netuid = 1
    block = 123
    rate_limit_value = "10"
    mocker.patch.object(subtensor, "_get_hyperparameter", return_value=rate_limit_value)

    # Call
    result = subtensor.serving_rate_limit(netuid, block)

    # Asserts
    assert result is not None
    assert result == int(rate_limit_value)
    subtensor._get_hyperparameter.assert_called_once_with(
        param_name="ServingRateLimit", netuid=netuid, block=block
    )


def test_serving_rate_limit_no_data(mocker, subtensor):
    """Test serving_rate_limit returns None when no rate limit information is found."""
    # Prep
    netuid = 1
    block = 123
    mocker.patch.object(subtensor, "_get_hyperparameter", return_value=None)

    # Call
    result = subtensor.serving_rate_limit(netuid, block)

    # Asserts
    assert result is None
    subtensor._get_hyperparameter.assert_called_once_with(
        param_name="ServingRateLimit", netuid=netuid, block=block
    )


def test_serving_rate_limit_no_block(mocker, subtensor):
    """Test serving_rate_limit with no block specified."""
    # Prep
    netuid = 1
    rate_limit_value = "10"
    mocker.patch.object(subtensor, "_get_hyperparameter", return_value=rate_limit_value)

    # Call
    result = subtensor.serving_rate_limit(netuid)

    # Asserts
    assert result is not None
    assert result == int(rate_limit_value)
    subtensor._get_hyperparameter.assert_called_once_with(
        param_name="ServingRateLimit", netuid=netuid, block=None
    )


# `tx_rate_limit` tests
def test_tx_rate_limit_success(mocker, subtensor):
    """Test tx_rate_limit returns correct data when rate limit information is found."""
    # Prep
    block = 123
    rate_limit_value = 100
    mock_result = mocker.MagicMock(value=rate_limit_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.tx_rate_limit(block)

    # Asserts
    assert result is not None
    assert result == rate_limit_value
    subtensor.query_subtensor.assert_called_once_with("TxRateLimit", block)


def test_tx_rate_limit_no_data(mocker, subtensor):
    """Test tx_rate_limit returns None when no rate limit information is found."""
    # Prep
    block = 123
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)

    # Call
    result = subtensor.tx_rate_limit(block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with("TxRateLimit", block)


def test_tx_rate_limit_no_value_attribute(mocker, subtensor):
    """Test tx_rate_limit returns None when result has no value attribute."""
    # Prep
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.tx_rate_limit(block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with("TxRateLimit", block)


def test_tx_rate_limit_no_block(mocker, subtensor):
    """Test tx_rate_limit with no block specified."""
    # Prep
    rate_limit_value = 100
    mock_result = mocker.MagicMock(value=rate_limit_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.tx_rate_limit()

    # Asserts
    assert result is not None
    assert result == rate_limit_value
    subtensor.query_subtensor.assert_called_once_with("TxRateLimit", None)


############################
# Network Parameters tests #
############################


# `subnet_exists` tests
def test_subnet_exists_success(mocker, subtensor):
    """Test subnet_exists returns True when subnet exists."""
    # Prep
    netuid = 1
    block = 123
    mock_result = mocker.MagicMock(value=True)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.subnet_exists(netuid, block)

    # Asserts
    assert result is True
    subtensor.query_subtensor.assert_called_once_with("NetworksAdded", block, [netuid])


def test_subnet_exists_no_data(mocker, subtensor):
    """Test subnet_exists returns False when no subnet information is found."""
    # Prep
    netuid = 1
    block = 123
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)

    # Call
    result = subtensor.subnet_exists(netuid, block)

    # Asserts
    assert result is False
    subtensor.query_subtensor.assert_called_once_with("NetworksAdded", block, [netuid])


def test_subnet_exists_no_value_attribute(mocker, subtensor):
    """Test subnet_exists returns False when result has no value attribute."""
    # Prep
    netuid = 1
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.subnet_exists(netuid, block)

    # Asserts
    assert result is False
    subtensor.query_subtensor.assert_called_once_with("NetworksAdded", block, [netuid])


def test_subnet_exists_no_block(mocker, subtensor):
    """Test subnet_exists with no block specified."""
    # Prep
    netuid = 1
    mock_result = mocker.MagicMock(value=True)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.subnet_exists(netuid)

    # Asserts
    assert result is True
    subtensor.query_subtensor.assert_called_once_with("NetworksAdded", None, [netuid])


# `get_all_subnet_netuids` tests
def test_get_all_subnet_netuids_success(mocker, subtensor):
    """Test get_all_subnet_netuids returns correct list when netuid information is found."""
    # Prep
    block = 123
    mock_netuid1 = mocker.MagicMock(value=1)
    mock_netuid2 = mocker.MagicMock(value=2)
    mock_result = mocker.MagicMock()
    mock_result.records = True
    mock_result.__iter__.return_value = [(mock_netuid1, True), (mock_netuid2, True)]
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_all_subnet_netuids(block)

    # Asserts
    assert result == [1, 2]
    subtensor.query_map_subtensor.assert_called_once_with("NetworksAdded", block)


def test_get_all_subnet_netuids_no_data(mocker, subtensor):
    """Test get_all_subnet_netuids returns empty list when no netuid information is found."""
    # Prep
    block = 123
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=None)

    # Call
    result = subtensor.get_all_subnet_netuids(block)

    # Asserts
    assert result == []
    subtensor.query_map_subtensor.assert_called_once_with("NetworksAdded", block)


def test_get_all_subnet_netuids_no_records_attribute(mocker, subtensor):
    """Test get_all_subnet_netuids returns empty list when result has no records attribute."""
    # Prep
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.records
    mock_result.__iter__.return_value = []
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_all_subnet_netuids(block)

    # Asserts
    assert result == []
    subtensor.query_map_subtensor.assert_called_once_with("NetworksAdded", block)


def test_get_all_subnet_netuids_no_block(mocker, subtensor):
    """Test get_all_subnet_netuids with no block specified."""
    # Prep
    mock_netuid1 = mocker.MagicMock(value=1)
    mock_netuid2 = mocker.MagicMock(value=2)
    mock_result = mocker.MagicMock()
    mock_result.records = True
    mock_result.__iter__.return_value = [(mock_netuid1, True), (mock_netuid2, True)]
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_all_subnet_netuids()

    # Asserts
    assert result == [1, 2]
    subtensor.query_map_subtensor.assert_called_once_with("NetworksAdded", None)


# `get_total_subnets` tests
def test_get_total_subnets_success(mocker, subtensor):
    """Test get_total_subnets returns correct data when total subnet information is found."""
    # Prep
    block = 123
    total_subnets_value = 10
    mock_result = mocker.MagicMock(value=total_subnets_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_total_subnets(block)

    # Asserts
    assert result is not None
    assert result == total_subnets_value
    subtensor.query_subtensor.assert_called_once_with("TotalNetworks", block)


def test_get_total_subnets_no_data(mocker, subtensor):
    """Test get_total_subnets returns None when no total subnet information is found."""
    # Prep
    block = 123
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)

    # Call
    result = subtensor.get_total_subnets(block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with("TotalNetworks", block)


def test_get_total_subnets_no_value_attribute(mocker, subtensor):
    """Test get_total_subnets returns None when result has no value attribute."""
    # Prep
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value  # Simulating a missing value attribute
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_total_subnets(block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with("TotalNetworks", block)


def test_get_total_subnets_no_block(mocker, subtensor):
    """Test get_total_subnets with no block specified."""
    # Prep
    total_subnets_value = 10
    mock_result = mocker.MagicMock(value=total_subnets_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_total_subnets()

    # Asserts
    assert result is not None
    assert result == total_subnets_value
    subtensor.query_subtensor.assert_called_once_with("TotalNetworks", None)


# `get_subnet_modality` tests
def test_get_subnet_modality_success(mocker, subtensor):
    """Test get_subnet_modality returns correct data when modality information is found."""
    # Prep
    netuid = 1
    block = 123
    modality_value = 42
    mock_result = mocker.MagicMock(value=modality_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnet_modality(netuid, block)

    # Asserts
    assert result is not None
    assert result == modality_value
    subtensor.query_subtensor.assert_called_once_with(
        "NetworkModality", block, [netuid]
    )


def test_get_subnet_modality_no_data(mocker, subtensor):
    """Test get_subnet_modality returns None when no modality information is found."""
    # Prep
    netuid = 1
    block = 123
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)

    # Call
    result = subtensor.get_subnet_modality(netuid, block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with(
        "NetworkModality", block, [netuid]
    )


def test_get_subnet_modality_no_value_attribute(mocker, subtensor):
    """Test get_subnet_modality returns None when result has no value attribute."""
    # Prep
    netuid = 1
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value  # Simulating a missing value attribute
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnet_modality(netuid, block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with(
        "NetworkModality", block, [netuid]
    )


def test_get_subnet_modality_no_block_specified(mocker, subtensor):
    """Test get_subnet_modality with no block specified."""
    # Prep
    netuid = 1
    modality_value = 42
    mock_result = mocker.MagicMock(value=modality_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnet_modality(netuid)

    # Asserts
    assert result is not None
    assert result == modality_value
    subtensor.query_subtensor.assert_called_once_with("NetworkModality", None, [netuid])


# `get_emission_value_by_subnet` tests
def test_get_emission_value_by_subnet_success(mocker, subtensor):
    """Test get_emission_value_by_subnet returns correct data when emission value is found."""
    # Prep
    netuid = 1
    block = 123
    emission_value = 1000
    mock_result = mocker.MagicMock(value=emission_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.get_emission_value_by_subnet(netuid, block)

    # Asserts
    assert result is not None
    subtensor.query_subtensor.assert_called_once_with("EmissionValues", block, [netuid])
    spy_balance_from_rao.assert_called_once_with(emission_value)
    assert result == Balance.from_rao(emission_value)


def test_get_emission_value_by_subnet_no_data(mocker, subtensor):
    """Test get_emission_value_by_subnet returns None when no emission value is found."""
    # Prep
    netuid = 1
    block = 123
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.get_emission_value_by_subnet(netuid, block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with("EmissionValues", block, [netuid])
    spy_balance_from_rao.assert_not_called()


def test_get_emission_value_by_subnet_no_value_attribute(mocker, subtensor):
    """Test get_emission_value_by_subnet returns None when result has no value attribute."""
    # Prep
    netuid = 1
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value  # Simulating a missing value attribute
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.get_emission_value_by_subnet(netuid, block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with("EmissionValues", block, [netuid])
    spy_balance_from_rao.assert_not_called()


def test_get_emission_value_by_subnet_no_block_specified(mocker, subtensor):
    """Test get_emission_value_by_subnet with no block specified."""
    # Prep
    netuid = 1
    emission_value = 1000
    mock_result = mocker.MagicMock(value=emission_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.get_emission_value_by_subnet(netuid)

    # Asserts
    assert result is not None
    subtensor.query_subtensor.assert_called_once_with("EmissionValues", None, [netuid])
    spy_balance_from_rao.assert_called_once_with(emission_value)
    assert result == Balance.from_rao(emission_value)


# `get_subnet_connection_requirements` tests
def test_get_subnet_connection_requirements_success(mocker, subtensor):
    """Test get_subnet_connection_requirements returns correct data when requirements are found."""
    # Prep
    netuid = 1
    block = 123
    mock_tuple1 = (mocker.MagicMock(value="requirement1"), mocker.MagicMock(value=10))
    mock_tuple2 = (mocker.MagicMock(value="requirement2"), mocker.MagicMock(value=20))
    mock_result = mocker.MagicMock()
    mock_result.records = [mock_tuple1, mock_tuple2]
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnet_connection_requirements(netuid, block)

    # Asserts
    assert result == {"requirement1": 10, "requirement2": 20}
    subtensor.query_map_subtensor.assert_called_once_with(
        "NetworkConnect", block, [netuid]
    )


def test_get_subnet_connection_requirements_no_data(mocker, subtensor):
    """Test get_subnet_connection_requirements returns empty dict when no data is found."""
    # Prep
    netuid = 1
    block = 123
    mock_result = mocker.MagicMock()
    mock_result.records = []
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnet_connection_requirements(netuid, block)

    # Asserts
    assert result == {}
    subtensor.query_map_subtensor.assert_called_once_with(
        "NetworkConnect", block, [netuid]
    )


def test_get_subnet_connection_requirements_no_records_attribute(mocker, subtensor):
    """Test get_subnet_connection_requirements returns empty dict when result has no records attribute."""
    # Prep
    netuid = 1
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.records  # Simulating a missing records attribute

    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnet_connection_requirements(netuid, block)

    # Asserts
    assert result == {}
    subtensor.query_map_subtensor.assert_called_once_with(
        "NetworkConnect", block, [netuid]
    )


def test_get_subnet_connection_requirements_no_block_specified(mocker, subtensor):
    """Test get_subnet_connection_requirements with no block specified."""
    # Prep
    netuid = 1
    mock_tuple1 = (mocker.MagicMock(value="requirement1"), mocker.MagicMock(value=10))
    mock_tuple2 = (mocker.MagicMock(value="requirement2"), mocker.MagicMock(value=20))
    mock_result = mocker.MagicMock()
    mock_result.records = [mock_tuple1, mock_tuple2]
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnet_connection_requirements(netuid)

    # Asserts
    assert result == {"requirement1": 10, "requirement2": 20}
    subtensor.query_map_subtensor.assert_called_once_with(
        "NetworkConnect", None, [netuid]
    )


# `get_subnets` tests
def test_get_subnets_success(mocker, subtensor):
    """Test get_subnets returns correct list when subnet information is found."""
    # Prep
    block = 123
    mock_netuid1 = mocker.MagicMock(value=1)
    mock_netuid2 = mocker.MagicMock(value=2)
    mock_result = mocker.MagicMock()
    mock_result.records = [(mock_netuid1, True), (mock_netuid2, True)]
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnets(block)

    # Asserts
    assert result == [1, 2]
    subtensor.query_map_subtensor.assert_called_once_with("NetworksAdded", block)


def test_get_subnets_no_data(mocker, subtensor):
    """Test get_subnets returns empty list when no subnet information is found."""
    # Prep
    block = 123
    mock_result = mocker.MagicMock()
    mock_result.records = []
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnets(block)

    # Asserts
    assert result == []
    subtensor.query_map_subtensor.assert_called_once_with("NetworksAdded", block)


def test_get_subnets_no_records_attribute(mocker, subtensor):
    """Test get_subnets returns empty list when result has no records attribute."""
    # Prep
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.records  # Simulating a missing records attribute
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnets(block)

    # Asserts
    assert result == []
    subtensor.query_map_subtensor.assert_called_once_with("NetworksAdded", block)


def test_get_subnets_no_block_specified(mocker, subtensor):
    """Test get_subnets with no block specified."""
    # Prep
    mock_netuid1 = mocker.MagicMock(value=1)
    mock_netuid2 = mocker.MagicMock(value=2)
    mock_result = mocker.MagicMock()
    mock_result.records = [(mock_netuid1, True), (mock_netuid2, True)]
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnets()

    # Asserts
    assert result == [1, 2]
    subtensor.query_map_subtensor.assert_called_once_with("NetworksAdded", None)


def test_get_subnets_correct_length(mocker, subtensor):
    """Test get_subnets returns a list of the correct length."""
    # Prep
    block = 123
    num_records = 500
    mock_records = [(mocker.MagicMock(value=i), True) for i in range(num_records)]
    mock_result = mocker.MagicMock()
    mock_result.records = mock_records
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnets(block)

    # Asserts
    assert len(result) == num_records
    subtensor.query_map_subtensor.assert_called_once_with("NetworksAdded", block)


# `get_all_subnets_info` tests
def test_get_all_subnets_info_success(mocker, subtensor):
    """Test get_all_subnets_info returns correct data when subnet information is found."""
    # Prep
    block = 123
    subnet_data = [1, 2, 3]  # Mocked response data
    mocker.patch.object(
        subtensor.substrate, "get_block_hash", return_value="mock_block_hash"
    )
    mock_response = {"result": subnet_data}
    mocker.patch.object(subtensor.substrate, "rpc_request", return_value=mock_response)
    mocker.patch.object(
        subtensor_module.SubnetInfo,
        "list_from_vec_u8",
        return_value="list_from_vec_u80",
    )

    # Call
    result = subtensor.get_all_subnets_info(block)

    # Asserts
    subtensor.substrate.get_block_hash.assert_called_once_with(block)
    subtensor.substrate.rpc_request.assert_called_once_with(
        method="subnetInfo_getSubnetsInfo", params=["mock_block_hash"]
    )
    subtensor_module.SubnetInfo.list_from_vec_u8.assert_called_once_with(subnet_data)


@pytest.mark.parametrize("result_", [[], None])
def test_get_all_subnets_info_no_data(mocker, subtensor, result_):
    """Test get_all_subnets_info returns empty list when no subnet information is found."""
    # Prep
    block = 123
    mocker.patch.object(
        subtensor.substrate, "get_block_hash", return_value="mock_block_hash"
    )
    mock_response = {"result": result_}
    mocker.patch.object(subtensor.substrate, "rpc_request", return_value=mock_response)
    mocker.patch.object(subtensor_module.SubnetInfo, "list_from_vec_u8")

    # Call
    result = subtensor.get_all_subnets_info(block)

    # Asserts
    assert result == []
    subtensor.substrate.get_block_hash.assert_called_once_with(block)
    subtensor.substrate.rpc_request.assert_called_once_with(
        method="subnetInfo_getSubnetsInfo", params=["mock_block_hash"]
    )
    subtensor_module.SubnetInfo.list_from_vec_u8.assert_not_called()


def test_get_all_subnets_info_retry(mocker, subtensor):
    """Test get_all_subnets_info retries on failure."""
    # Prep
    block = 123
    subnet_data = [1, 2, 3]
    mocker.patch.object(
        subtensor.substrate, "get_block_hash", return_value="mock_block_hash"
    )
    mock_response = {"result": subnet_data}
    mock_rpc_request = mocker.patch.object(
        subtensor.substrate,
        "rpc_request",
        side_effect=[Exception, Exception, mock_response],
    )
    mocker.patch.object(
        subtensor_module.SubnetInfo, "list_from_vec_u8", return_value=["some_data"]
    )

    # Call
    result = subtensor.get_all_subnets_info(block)

    # Asserts
    subtensor.substrate.get_block_hash.assert_called_with(block)
    assert mock_rpc_request.call_count == 3
    subtensor_module.SubnetInfo.list_from_vec_u8.assert_called_once_with(subnet_data)
    assert result == ["some_data"]


# `get_subnet_info` tests
def test_get_subnet_info_success(mocker, subtensor):
    """Test get_subnet_info returns correct data when subnet information is found."""
    # Prep
    netuid = 1
    block = 123
    subnet_data = [1, 2, 3]
    mocker.patch.object(
        subtensor.substrate, "get_block_hash", return_value="mock_block_hash"
    )
    mock_response = {"result": subnet_data}
    mocker.patch.object(subtensor.substrate, "rpc_request", return_value=mock_response)
    mocker.patch.object(
        subtensor_module.SubnetInfo, "from_vec_u8", return_value=["from_vec_u8"]
    )

    # Call
    result = subtensor.get_subnet_info(netuid, block)

    # Asserts
    subtensor.substrate.get_block_hash.assert_called_once_with(block)
    subtensor.substrate.rpc_request.assert_called_once_with(
        method="subnetInfo_getSubnetInfo", params=[netuid, "mock_block_hash"]
    )
    subtensor_module.SubnetInfo.from_vec_u8.assert_called_once_with(subnet_data)


@pytest.mark.parametrize("result_", [None, {}])
def test_get_subnet_info_no_data(mocker, subtensor, result_):
    """Test get_subnet_info returns None when no subnet information is found."""
    # Prep
    netuid = 1
    block = 123
    mocker.patch.object(
        subtensor.substrate, "get_block_hash", return_value="mock_block_hash"
    )
    mock_response = {"result": result_}
    mocker.patch.object(subtensor.substrate, "rpc_request", return_value=mock_response)
    mocker.patch.object(subtensor_module.SubnetInfo, "from_vec_u8")

    # Call
    result = subtensor.get_subnet_info(netuid, block)

    # Asserts
    assert result is None
    subtensor.substrate.get_block_hash.assert_called_once_with(block)
    subtensor.substrate.rpc_request.assert_called_once_with(
        method="subnetInfo_getSubnetInfo", params=[netuid, "mock_block_hash"]
    )
    subtensor_module.SubnetInfo.from_vec_u8.assert_not_called()


def test_get_subnet_info_retry(mocker, subtensor):
    """Test get_subnet_info retries on failure."""
    # Prep
    netuid = 1
    block = 123
    subnet_data = [1, 2, 3]
    mocker.patch.object(
        subtensor.substrate, "get_block_hash", return_value="mock_block_hash"
    )
    mock_response = {"result": subnet_data}
    mock_rpc_request = mocker.patch.object(
        subtensor.substrate,
        "rpc_request",
        side_effect=[Exception, Exception, mock_response],
    )
    mocker.patch.object(
        subtensor_module.SubnetInfo, "from_vec_u8", return_value=["from_vec_u8"]
    )

    # Call
    result = subtensor.get_subnet_info(netuid, block)

    # Asserts
    subtensor.substrate.get_block_hash.assert_called_with(block)
    assert mock_rpc_request.call_count == 3
    subtensor_module.SubnetInfo.from_vec_u8.assert_called_once_with(subnet_data)


# `get_subnet_hyperparameters` tests
def test_get_subnet_hyperparameters_success(mocker, subtensor):
    """Test get_subnet_hyperparameters returns correct data when hyperparameters are found."""
    # Prep
    netuid = 1
    block = 123
    hex_bytes_result = "0x010203"
    bytes_result = bytes.fromhex(hex_bytes_result[2:])
    mocker.patch.object(subtensor, "query_runtime_api", return_value=hex_bytes_result)
    mocker.patch.object(
        subtensor_module.SubnetHyperparameters,
        "from_vec_u8",
        return_value=["from_vec_u8"],
    )

    # Call
    result = subtensor.get_subnet_hyperparameters(netuid, block)

    # Asserts
    subtensor.query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_hyperparams",
        params=[netuid],
        block=block,
    )
    subtensor_module.SubnetHyperparameters.from_vec_u8.assert_called_once_with(
        bytes_result
    )


def test_get_subnet_hyperparameters_no_data(mocker, subtensor):
    """Test get_subnet_hyperparameters returns empty list when no data is found."""
    # Prep
    netuid = 1
    block = 123
    mocker.patch.object(subtensor, "query_runtime_api", return_value=None)
    mocker.patch.object(subtensor_module.SubnetHyperparameters, "from_vec_u8")

    # Call
    result = subtensor.get_subnet_hyperparameters(netuid, block)

    # Asserts
    assert result == []
    subtensor.query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_hyperparams",
        params=[netuid],
        block=block,
    )
    subtensor_module.SubnetHyperparameters.from_vec_u8.assert_not_called()


def test_get_subnet_hyperparameters_hex_without_prefix(mocker, subtensor):
    """Test get_subnet_hyperparameters correctly processes hex string without '0x' prefix."""
    # Prep
    netuid = 1
    block = 123
    hex_bytes_result = "010203"
    bytes_result = bytes.fromhex(hex_bytes_result)
    mocker.patch.object(subtensor, "query_runtime_api", return_value=hex_bytes_result)
    mocker.patch.object(subtensor_module.SubnetHyperparameters, "from_vec_u8")

    # Call
    result = subtensor.get_subnet_hyperparameters(netuid, block)

    # Asserts
    subtensor.query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_hyperparams",
        params=[netuid],
        block=block,
    )
    subtensor_module.SubnetHyperparameters.from_vec_u8.assert_called_once_with(
        bytes_result
    )


# `get_subnet_owner` tests
def test_get_subnet_owner_success(mocker, subtensor):
    """Test get_subnet_owner returns correct data when owner information is found."""
    # Prep
    netuid = 1
    block = 123
    owner_address = "5F3sa2TJAWMqDhXG6jhV4N8ko9rXPM6twz9mG9m3rrgq3xiJ"
    mock_result = mocker.MagicMock(value=owner_address)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnet_owner(netuid, block)

    # Asserts
    subtensor.query_subtensor.assert_called_once_with("SubnetOwner", block, [netuid])
    assert result == owner_address


def test_get_subnet_owner_no_data(mocker, subtensor):
    """Test get_subnet_owner returns None when no owner information is found."""
    # Prep
    netuid = 1
    block = 123
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)

    # Call
    result = subtensor.get_subnet_owner(netuid, block)

    # Asserts
    subtensor.query_subtensor.assert_called_once_with("SubnetOwner", block, [netuid])
    assert result is None


def test_get_subnet_owner_no_value_attribute(mocker, subtensor):
    """Test get_subnet_owner returns None when result has no value attribute."""
    # Prep
    netuid = 1
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value  # Simulating a missing value attribute
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnet_owner(netuid, block)

    # Asserts
    subtensor.query_subtensor.assert_called_once_with("SubnetOwner", block, [netuid])
    assert result is None


####################
# Nomination tests #
####################


# `is_hotkey_delegate` tests
def test_is_hotkey_delegate_success(mocker, subtensor):
    """Test is_hotkey_delegate returns True when hotkey is a delegate."""
    # Prep
    hotkey_ss58 = "hotkey_ss58"
    block = 123
    mock_delegates = [
        mocker.MagicMock(hotkey_ss58=hotkey_ss58),
        mocker.MagicMock(hotkey_ss58="hotkey_ss583"),
    ]
    mocker.patch.object(subtensor, "get_delegates", return_value=mock_delegates)

    # Call
    result = subtensor.is_hotkey_delegate(hotkey_ss58, block)

    # Asserts
    subtensor.get_delegates.assert_called_once_with(block=block)
    assert result is True


def test_is_hotkey_delegate_not_found(mocker, subtensor):
    """Test is_hotkey_delegate returns False when hotkey is not a delegate."""
    # Prep
    hotkey_ss58 = "hotkey_ss58"
    block = 123
    mock_delegates = [mocker.MagicMock(hotkey_ss58="hotkey_ss583")]
    mocker.patch.object(subtensor, "get_delegates", return_value=mock_delegates)

    # Call
    result = subtensor.is_hotkey_delegate(hotkey_ss58, block)

    # Asserts
    subtensor.get_delegates.assert_called_once_with(block=block)
    assert result is False


# `get_delegate_take` tests
def test_get_delegate_take_success(mocker, subtensor):
    """Test get_delegate_take returns correct data when delegate take is found."""
    # Prep
    hotkey_ss58 = "hotkey_ss58"
    block = 123
    delegate_take_value = 32768
    mock_result = mocker.MagicMock(value=delegate_take_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)
    spy_u16_normalized_float = mocker.spy(subtensor_module, "U16_NORMALIZED_FLOAT")

    # Call
    subtensor.get_delegate_take(hotkey_ss58, block)

    # Asserts
    subtensor.query_subtensor.assert_called_once_with("Delegates", block, [hotkey_ss58])
    spy_u16_normalized_float.assert_called_once_with(delegate_take_value)


def test_get_delegate_take_no_data(mocker, subtensor):
    """Test get_delegate_take returns None when no delegate take is found."""
    # Prep
    hotkey_ss58 = "hotkey_ss58"
    block = 123
    delegate_take_value = 32768
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)
    spy_u16_normalized_float = mocker.spy(subtensor_module, "U16_NORMALIZED_FLOAT")

    # Call
    result = subtensor.get_delegate_take(hotkey_ss58, block)

    # Asserts
    subtensor.query_subtensor.assert_called_once_with("Delegates", block, [hotkey_ss58])
    spy_u16_normalized_float.assert_not_called()
    assert result is None


def test_get_remaining_arbitration_period(subtensor, mocker):
    """Tests successful retrieval of total stake for hotkey."""
    # Prep
    subtensor.query_subtensor = mocker.MagicMock(return_value=mocker.MagicMock(value=0))
    fake_ss58_address = "12bzRJfh7arnnfPPUZHeJUaE62QLEwhK48QnH9LXeK2m1iZU"

    # Call
    result = subtensor.get_remaining_arbitration_period(coldkey_ss58=fake_ss58_address)

    # Assertions
    subtensor.query_subtensor.assert_called_once_with(
        name="ColdkeyArbitrationBlock", block=None, params=[fake_ss58_address]
    )
    # if we change the methods logic in the future we have to be make sure the returned type is correct
    assert result == 0


def test_get_remaining_arbitration_period_happy(subtensor, mocker):
    """Tests successful retrieval of total stake for hotkey."""
    # Prep
    subtensor.query_subtensor = mocker.MagicMock(
        return_value=mocker.MagicMock(value=2000)
    )
    fake_ss58_address = "12bzRJfh7arnnfPPUZHeJUaE62QLEwhK48QnH9LXeK2m1iZU"

    # Call
    result = subtensor.get_remaining_arbitration_period(
        coldkey_ss58=fake_ss58_address, block=200
    )

    # Assertions
    subtensor.query_subtensor.assert_called_once_with(
        name="ColdkeyArbitrationBlock", block=200, params=[fake_ss58_address]
    )
    # if we change the methods logic in the future we have to be make sure the returned type is correct
    assert result == 1800  # 2000 - 200
