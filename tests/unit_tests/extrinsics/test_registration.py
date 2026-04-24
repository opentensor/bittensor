import pytest
from bittensor_wallet import Wallet
from bittensor.core.types import ExtrinsicResponse
from bittensor.core.extrinsics import registration
from bittensor.core.subtensor import Subtensor
from bittensor.utils.balance import Balance


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


@pytest.mark.parametrize(
    "subnet_exists, neuron_is_null, recycle_success, is_registered, expected_result, test_id",
    [
        # Happy paths
        (True, False, None, None, True, "neuron-not-null"),
        (True, True, True, True, True, "happy-path-wallet-registered"),
        # Error paths
        (True, True, True, False, False, "error-path-not-registered"),
        (False, True, False, None, False, "subnet-non-existence"),
        (True, True, False, False, False, "error-path-recycling-failed"),
    ],
)
def test_register_limit_extrinsic(
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
    result = registration.register_limit_extrinsic(
        subtensor=mock_subtensor,
        wallet=mock_wallet,
        netuid=123,
        limit_price=Balance.from_rao(1000000000),
    )
    # Assert
    assert result.success == expected_result, f"Test failed for test_id: {test_id}"
