import pytest
from bittensor.core.extrinsics import root
from bittensor.core.subtensor import Subtensor
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import Balance


@pytest.fixture
def mock_subtensor(mocker):
    mock = mocker.MagicMock(spec=Subtensor)
    mock.network = "magic_mock"
    mock.substrate = mocker.Mock()
    return mock


@pytest.fixture
def mock_wallet(mocker):
    mock = mocker.MagicMock()
    mock.hotkey.ss58_address = "fake_hotkey_address"
    return mock


@pytest.mark.parametrize(
    "wait_for_inclusion, wait_for_finalization, hotkey_registered, get_uid_for_hotkey_on_subnet, registration_success, expected_result",
    [
        (
            False,
            True,
            [True, None],
            0,
            True,
            True,
        ),  # Already registered after attempt
        (
            False,
            True,
            [False, 1],
            0,
            True,
            True,
        ),  # Registration succeeds with user confirmation
        (False, True, [False, None], 0, False, False),  # Registration fails
        (
            False,
            True,
            [False, None],
            None,
            True,
            False,
        ),  # Registration succeeds but neuron not found
    ],
    ids=[
        "success-already-registered",
        "success-registration-succeeds",
        "failure-registration-failed",
        "failure-neuron-not-found",
    ],
)
def test_root_register_extrinsic(
    mock_subtensor,
    mock_wallet,
    wait_for_inclusion,
    wait_for_finalization,
    hotkey_registered,
    get_uid_for_hotkey_on_subnet,
    registration_success,
    expected_result,
    mocker,
):
    # Preps
    mock_subtensor.is_hotkey_registered.return_value = hotkey_registered[0]
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        mock_subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(registration_success, "Error registering"),
    )
    mocker.patch.object(
        mock_subtensor.substrate,
        "query",
        return_value=hotkey_registered[1],
    )
    mocker.patch.object(
        mock_subtensor,
        "get_balance",
        return_value=Balance(1),
    )
    mocked_get_uid_for_hotkey_on_subnet = mocker.patch.object(
        mock_subtensor,
        "get_uid_for_hotkey_on_subnet",
        return_value=get_uid_for_hotkey_on_subnet,
    )

    # Act
    result = root.root_register_extrinsic(
        subtensor=mock_subtensor,
        wallet=mock_wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    # Assert
    assert result.success == expected_result

    if not hotkey_registered[0]:
        mock_subtensor.compose_call.assert_called_once_with(
            call_module="SubtensorModule",
            call_function="root_register",
            call_params={"hotkey": "fake_hotkey_address"},
        )
        mocked_sign_and_send_extrinsic.assert_called_once_with(
            call=mock_subtensor.compose_call.return_value,
            wallet=mock_wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=None,
            raise_error=False,
        )


def test_root_register_extrinsic_insufficient_balance(
    mock_subtensor,
    mock_wallet,
    mocker,
):
    mocker.patch.object(
        mock_subtensor,
        "get_balance",
        return_value=Balance(0),
    )

    success, _ = root.root_register_extrinsic(
        subtensor=mock_subtensor,
        wallet=mock_wallet,
    )

    assert success is False

    mock_subtensor.get_balance.assert_called_once_with(
        address=mock_wallet.coldkeypub.ss58_address,
        block=mock_subtensor.get_current_block.return_value,
    )
    mock_subtensor.substrate.submit_extrinsic.assert_not_called()


def test_set_root_claim_type_extrinsic(subtensor, fake_wallet, mocker):
    """Tests `set_root_claim_type_extrinsic` extrinsic function."""
    # Preps
    new_root_claim_type = mocker.Mock(spec=int)
    mocked_pallet_compose_call = mocker.patch.object(
        root.SubtensorModule, "set_root_claim_type"
    )
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # call
    response = root.set_root_claim_type_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        new_root_claim_type=new_root_claim_type,
    )

    # asserts
    mocked_pallet_compose_call.assert_called_once_with(
        new_root_claim_type=new_root_claim_type
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_pallet_compose_call.return_value,
        wallet=fake_wallet,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value


def test_claim_root_extrinsic(subtensor, fake_wallet, mocker):
    """Tests `claim_root_extrinsic` extrinsic function."""
    # Preps
    netuids = mocker.Mock(spec=list)
    mocked_pallet_compose_call = mocker.patch.object(root.SubtensorModule, "claim_root")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # call
    response = root.claim_root_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuids=netuids,
    )

    # asserts
    mocked_pallet_compose_call.assert_called_once_with(subnets=netuids)
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_pallet_compose_call.return_value,
        wallet=fake_wallet,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value
