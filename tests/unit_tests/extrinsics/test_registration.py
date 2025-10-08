import pytest
from bittensor_wallet import Wallet
from bittensor.core.types import ExtrinsicResponse
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
    "subnet_exists, neuron_is_null, cuda_available, expected_result, expected_message",
    [
        (
            False,
            True,
            True,
            False,
            "Subnet 123 does not exist.",
        ),
        (True, False, True, True, "Already registered."),
        (True, True, False, False, "CUDA not available."),
    ],
    ids=["subnet-does-not-exist", "neuron-already-registered", "cuda-unavailable"],
)
def test_register_extrinsic_without_pow(
    mock_subtensor,
    mock_wallet,
    mocker,
    subnet_exists,
    neuron_is_null,
    cuda_available,
    expected_result,
    expected_message,
):
    # Arrange
    mocker.patch.object(mock_subtensor, "subnet_exists", return_value=subnet_exists)
    fake_neuron = mocker.patch.object(
        mock_subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=mocker.MagicMock(is_null=neuron_is_null),
    )
    mocker.patch("torch.cuda.is_available", return_value=cuda_available)
    mocker.patch(
        "bittensor.utils.registration.pow._get_block_with_retry",
        return_value=(0, 0, "00ff11ee"),
    )

    # Act
    result = registration.register_extrinsic(
        subtensor=mock_subtensor,
        wallet=mock_wallet,
        netuid=123,
        wait_for_inclusion=True,
        wait_for_finalization=True,
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
    data = (
        {"neuron": fake_neuron.return_value}
        if fake_neuron.call_count > 0 and cuda_available
        else None
    )
    expected_result = ExtrinsicResponse(
        expected_result,
        expected_message,
        extrinsic_function="register_extrinsic",
        data=data,
    )
    assert result == expected_result


@pytest.mark.parametrize(
    "pow_success, pow_stale, registration_success, cuda, hotkey_registered, expected_result",
    [
        (True, False, True, False, False, True),
        (True, False, True, True, False, True),
        # Pow failed but key was registered already
        (False, False, False, False, True, True),
        # Pow was a success but registration failed with error 'key already registered'
        (True, False, False, False, False, False),
    ],
    ids=[
        "successful-with-valid-pow",
        "successful-with-valid-cuda-pow",
        "hotkey-registered",
        "registration-fail-key-registered",
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
    mocker,
):
    # Arrange
    mocker.patch(
        "bittensor.utils.registration.pow._solve_for_difficulty_fast",
        return_value=mock_pow_solution if pow_success else None,
    )
    mocker.patch(
        "bittensor.utils.registration.pow._solve_for_difficulty_fast_cuda",
        return_value=mock_pow_solution if pow_success else None,
    )
    mocker.patch.object(
        mock_subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(
            registration_success, "HotKeyAlreadyRegisteredInSubNet"
        ),
    )
    mocker.patch("torch.cuda.is_available", return_value=cuda)

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
    assert result[0] is expected_result


@pytest.mark.parametrize(
    "subnet_exists, neuron_is_null, recycle_success, is_registered, expected_result, test_id",
    [
        # Happy paths
        (True, False, None, None, True, "neuron-not-null"),
        (True, True, True, True, True, "happy-path-wallet-registered"),
        # Error paths
        (False, True, False, None, False, "subnet-non-existence"),
        (True, True, False, False, False, "error-path-recycling-failed"),
        (True, True, True, False, False, "error-path-not-registered"),
    ],
)
def test_burned_register_extrinsic(
    mock_subtensor,
    mock_wallet,
    subnet_exists,
    neuron_is_null,
    recycle_success,
    is_registered,
    expected_result,
    test_id,
    mocker,
):
    # Arrange
    mock_substrate_ = mocker.MagicMock(
        **{"get_payment_info.return_value": {"partial_fee": 10}}
    )
    mocker.patch.object(mock_subtensor, "substrate", mock_substrate_)
    mocker.patch.object(mock_subtensor, "subnet_exists", return_value=subnet_exists)
    mocker.patch.object(
        mock_subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=mocker.MagicMock(is_null=neuron_is_null),
    )
    mocker.patch.object(
        mock_subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(recycle_success, "Mock error message"),
    )
    mocker.patch.object(
        mock_subtensor, "is_hotkey_registered", return_value=is_registered
    )

    # Act
    result = registration.burned_register_extrinsic(
        subtensor=mock_subtensor, wallet=mock_wallet, netuid=123
    )
    # Assert
    assert result.success == expected_result, f"Test failed for test_id: {test_id}"


def test_set_subnet_identity_extrinsic_is_success(mock_subtensor, mock_wallet, mocker):
    """Verify that set_subnet_identity_extrinsic calls the correct functions and returns the correct result."""
    # Preps
    netuid = 123
    subnet_name = "mock_subnet_name"
    github_repo = "mock_github_repo"
    subnet_contact = "mock_subnet_contact"
    subnet_url = "mock_subnet_url"
    logo_url = "mock_logo_url"
    discord = "mock_discord"
    description = "mock_description"
    additional = "mock_additional"

    mocked_compose_call = mocker.patch.object(mock_subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        mock_subtensor, "sign_and_send_extrinsic"
    )

    # Call
    result = registration.set_subnet_identity_extrinsic(
        subtensor=mock_subtensor,
        wallet=mock_wallet,
        netuid=netuid,
        subnet_name=subnet_name,
        github_repo=github_repo,
        subnet_contact=subnet_contact,
        subnet_url=subnet_url,
        logo_url=logo_url,
        discord=discord,
        description=description,
        additional=additional,
    )

    # Asserts
    mocked_compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="set_subnet_identity",
        call_params={
            "hotkey": mock_wallet.hotkey.ss58_address,
            "netuid": netuid,
            "subnet_name": subnet_name,
            "github_repo": github_repo,
            "subnet_contact": subnet_contact,
            "subnet_url": subnet_url,
            "logo_url": logo_url,
            "discord": discord,
            "description": description,
            "additional": additional,
        },
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=mock_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )

    assert result == mocked_sign_and_send_extrinsic.return_value


def test_set_subnet_identity_extrinsic_is_failed(mock_subtensor, mock_wallet, mocker):
    """Verify that set_subnet_identity_extrinsic calls the correct functions and returns False with bad result."""
    # Preps
    netuid = 123
    subnet_name = "mock_subnet_name"
    github_repo = "mock_github_repo"
    subnet_contact = "mock_subnet_contact"
    subnet_url = "mock_subnet_url"
    logo_url = "mock_logo_url"
    discord = "mock_discord"
    description = "mock_description"
    additional = "mock_additional"

    fake_error_message = "error message"

    mocked_compose_call = mocker.patch.object(mock_subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        mock_subtensor,
        "sign_and_send_extrinsic",
    )

    # Call
    result = registration.set_subnet_identity_extrinsic(
        subtensor=mock_subtensor,
        wallet=mock_wallet,
        netuid=netuid,
        subnet_name=subnet_name,
        github_repo=github_repo,
        subnet_contact=subnet_contact,
        subnet_url=subnet_url,
        logo_url=logo_url,
        discord=discord,
        description=description,
        additional=additional,
    )

    # Asserts
    mocked_compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="set_subnet_identity",
        call_params={
            "hotkey": mock_wallet.hotkey.ss58_address,
            "netuid": netuid,
            "subnet_name": subnet_name,
            "github_repo": github_repo,
            "subnet_contact": subnet_contact,
            "subnet_url": subnet_url,
            "logo_url": logo_url,
            "discord": discord,
            "description": description,
            "additional": additional,
        },
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=mock_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )

    assert result == mocked_sign_and_send_extrinsic.return_value
