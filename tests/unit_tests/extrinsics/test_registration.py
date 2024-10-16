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
from bittensor_wallet import Wallet

from bittensor.core.extrinsics import registration
from bittensor.core.subtensor import Subtensor
from bittensor.utils.registration import POWSolution


# Mocking external dependencies
@pytest.fixture
def mock_subtensor(mocker):
    mock = mocker.MagicMock(spec=Subtensor)
    mock.network = "mock_network"
    mock.substrate = mocker.MagicMock()
    return mock


@pytest.fixture
def mock_wallet(mocker):
    mock = mocker.MagicMock(spec=Wallet)
    mock.coldkeypub.ss58_address = "mock_address"
    mock.coldkey = mocker.MagicMock()
    mock.hotkey = mocker.MagicMock()
    mock.hotkey.ss58_address = "fake_ss58_address"
    return mock


@pytest.fixture
def mock_pow_solution(mocker):
    mock = mocker.MagicMock(spec=POWSolution)
    mock.block_number = 123
    mock.nonce = 456
    mock.seal = [0, 1, 2, 3]
    mock.is_stale.return_value = False
    return mock


@pytest.fixture
def mock_new_wallet(mocker):
    mock = mocker.MagicMock(spec=Wallet)
    mock.coldkeypub.ss58_address = "mock_address"
    mock.coldkey = mocker.MagicMock()
    mock.hotkey = mocker.MagicMock()
    return mock


@pytest.mark.parametrize(
    "subnet_exists, neuron_is_null, prompt, prompt_response, cuda_available, expected_result, test_id",
    [
        (False, True, True, True, True, False, "subnet-does-not-exist"),
        (True, False, True, True, True, True, "neuron-already-registered"),
        (True, True, True, False, True, False, "user-declines-prompt"),
        (True, True, False, None, False, False, "cuda-unavailable"),
    ],
)
def test_register_extrinsic_without_pow(
    mock_subtensor,
    mock_wallet,
    subnet_exists,
    neuron_is_null,
    prompt,
    prompt_response,
    cuda_available,
    expected_result,
    test_id,
    mocker,
):
    # Arrange
    with mocker.patch.object(
        mock_subtensor, "subnet_exists", return_value=subnet_exists
    ), mocker.patch.object(
        mock_subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=mocker.MagicMock(is_null=neuron_is_null),
    ), mocker.patch(
        "rich.prompt.Confirm.ask", return_value=prompt_response
    ), mocker.patch("torch.cuda.is_available", return_value=cuda_available):
        # Act
        result = registration.register_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            netuid=123,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            prompt=prompt,
            max_allowed_attempts=3,
            output_in_place=True,
            cuda=True,
            dev_id=0,
            tpb=256,
            num_processes=None,
            update_interval=None,
            log_verbose=False,
        )

        # Assert
        assert result == expected_result, f"Test failed for test_id: {test_id}"


@pytest.mark.parametrize(
    "pow_success, pow_stale, registration_success, cuda, hotkey_registered, expected_result, test_id",
    [
        (True, False, True, False, False, True, "successful-with-valid-pow"),
        (True, False, True, True, False, True, "successful-with-valid-cuda-pow"),
        # Pow failed but key was registered already
        (False, False, False, False, True, True, "hotkey-registered"),
        # Pow was a success but registration failed with error 'key already registered'
        (True, False, False, False, False, True, "registration-fail-key-registered"),
    ],
)
def test_register_extrinsic_with_pow(
    mock_subtensor,
    mock_wallet,
    mock_pow_solution,
    pow_success,
    pow_stale,
    registration_success,
    cuda,
    hotkey_registered,
    expected_result,
    test_id,
    mocker,
):
    # Arrange
    with mocker.patch(
        "bittensor.utils.registration._solve_for_difficulty_fast",
        return_value=mock_pow_solution if pow_success else None,
    ), mocker.patch(
        "bittensor.utils.registration._solve_for_difficulty_fast_cuda",
        return_value=mock_pow_solution if pow_success else None,
    ), mocker.patch(
        "bittensor.core.extrinsics.registration._do_pow_register",
        return_value=(registration_success, "HotKeyAlreadyRegisteredInSubNet"),
    ), mocker.patch("torch.cuda.is_available", return_value=cuda):
        # Act
        if pow_success:
            mock_pow_solution.is_stale.return_value = pow_stale

        if not pow_success and hotkey_registered:
            mock_subtensor.is_hotkey_registered = mocker.MagicMock(
                return_value=hotkey_registered
            )

        result = registration.register_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            netuid=123,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            prompt=False,
            max_allowed_attempts=3,
            output_in_place=True,
            cuda=cuda,
            dev_id=0,
            tpb=256,
            num_processes=None,
            update_interval=None,
            log_verbose=False,
        )

        # Assert
        assert result == expected_result, f"Test failed for test_id: {test_id}."


@pytest.mark.parametrize(
    "subnet_exists, neuron_is_null, recycle_success, prompt, prompt_response, is_registered, expected_result, test_id",
    [
        # Happy paths
        (True, False, None, False, None, None, True, "neuron-not-null"),
        (True, True, True, True, True, True, True, "happy-path-wallet-registered"),
        # Error paths
        (False, True, False, False, None, None, False, "subnet-non-existence"),
        (True, True, True, True, False, None, False, "prompt-declined"),
        (True, True, False, True, True, False, False, "error-path-recycling-failed"),
        (True, True, True, True, True, False, False, "error-path-not-registered"),
    ],
)
def test_burned_register_extrinsic(
    mock_subtensor,
    mock_wallet,
    subnet_exists,
    neuron_is_null,
    recycle_success,
    prompt,
    prompt_response,
    is_registered,
    expected_result,
    test_id,
    mocker,
):
    # Arrange
    with mocker.patch.object(
        mock_subtensor, "subnet_exists", return_value=subnet_exists
    ), mocker.patch.object(
        mock_subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=mocker.MagicMock(is_null=neuron_is_null),
    ), mocker.patch(
        "bittensor.core.extrinsics.registration._do_burned_register",
        return_value=(recycle_success, "Mock error message"),
    ), mocker.patch.object(
        mock_subtensor, "is_hotkey_registered", return_value=is_registered
    ):
        mock_confirm = mocker.MagicMock(return_value=prompt_response)
        registration.Confirm.ask = mock_confirm
        # Act
        result = registration.burned_register_extrinsic(
            subtensor=mock_subtensor, wallet=mock_wallet, netuid=123, prompt=prompt
        )
        # Assert
        assert result == expected_result, f"Test failed for test_id: {test_id}"

        if prompt:
            mock_confirm.assert_called_once()
        else:
            mock_confirm.assert_not_called()
