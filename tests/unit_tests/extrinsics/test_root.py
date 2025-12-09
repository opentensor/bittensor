import pytest
from bittensor.core.chain_data import RootClaimType
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
        wait_for_revealed_execution=True,
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


@pytest.mark.parametrize(
    "new_root_claim_type, expected_normalized",
    [
        ("Swap", "Swap"),
        ("Keep", "Keep"),
        (RootClaimType.Swap, "Swap"),
        (RootClaimType.Keep, "Keep"),
        (
            {"KeepSubnets": {"subnets": [1, 2, 3]}},
            {"KeepSubnets": {"subnets": [1, 2, 3]}},
        ),
        (RootClaimType.KeepSubnets([1, 2, 3]), {"KeepSubnets": {"subnets": [1, 2, 3]}}),
    ],
    ids=[
        "string-swap",
        "string-keep",
        "enum-swap",
        "enum-keep",
        "dict-keep-subnets",
        "callable-keep-subnets",
    ],
)
def test_set_root_claim_type_extrinsic(
    subtensor, fake_wallet, mocker, new_root_claim_type, expected_normalized
):
    """Tests `set_root_claim_type_extrinsic` extrinsic function with various input formats."""
    # Preps
    mocked_normalize = mocker.patch.object(
        RootClaimType, "normalize", return_value=expected_normalized
    )
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
    mocked_normalize.assert_called_once_with(new_root_claim_type)
    mocked_pallet_compose_call.assert_called_once_with(
        new_root_claim_type=expected_normalized
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


@pytest.mark.parametrize(
    "invalid_input, expected_error",
    [
        ("InvalidType", ValueError),
        ({"InvalidKey": {}}, ValueError),
        ({"KeepSubnets": {}}, ValueError),  # Empty subnets
        ({"KeepSubnets": {"subnets": []}}, ValueError),  # Empty subnets list
        (
            {"KeepSubnets": {"subnets": ["not", "integers"]}},
            ValueError,
        ),  # Non-integer subnets
        (123, TypeError),  # Wrong type
    ],
    ids=[
        "invalid-string",
        "invalid-dict-key",
        "empty-subnets-dict",
        "empty-subnets-list",
        "non-integer-subnets",
        "wrong-type",
    ],
)
def test_set_root_claim_type_extrinsic_validation_with_raise_error(
    subtensor, fake_wallet, mocker, invalid_input, expected_error
):
    """Tests `set_root_claim_type_extrinsic` validation for invalid inputs with raise_error=True."""
    # Preps
    test_error = expected_error("Test error")
    mocked_normalize = mocker.patch.object(
        RootClaimType, "normalize", side_effect=test_error
    )
    mocked_pallet_compose_call = mocker.patch.object(
        root.SubtensorModule, "set_root_claim_type"
    )

    # call and assert
    with pytest.raises(expected_error):
        root.set_root_claim_type_extrinsic(
            subtensor=subtensor,
            wallet=fake_wallet,
            new_root_claim_type=invalid_input,
            raise_error=True,
        )

    mocked_normalize.assert_called_once_with(invalid_input)
    mocked_pallet_compose_call.assert_not_called()


@pytest.mark.parametrize(
    "invalid_input, expected_error",
    [
        ("InvalidType", ValueError),
        ({"InvalidKey": {}}, ValueError),
        ({"KeepSubnets": {"subnets": []}}, ValueError),  # Empty subnets list
        (123, TypeError),  # Wrong type
    ],
    ids=[
        "invalid-string-no-raise",
        "invalid-dict-key-no-raise",
        "empty-subnets-list-no-raise",
        "wrong-type-no-raise",
    ],
)
def test_set_root_claim_type_extrinsic_validation_without_raise_error(
    subtensor, fake_wallet, mocker, invalid_input, expected_error
):
    """Tests `set_root_claim_type_extrinsic` validation for invalid inputs with raise_error=False."""
    # Preps
    test_error = expected_error("Test error")
    mocked_normalize = mocker.patch.object(
        RootClaimType, "normalize", side_effect=test_error
    )
    mocked_pallet_compose_call = mocker.patch.object(
        root.SubtensorModule, "set_root_claim_type"
    )
    mocked_from_exception = mocker.patch.object(ExtrinsicResponse, "from_exception")

    # call
    response = root.set_root_claim_type_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        new_root_claim_type=invalid_input,
        raise_error=False,
    )

    # assert
    mocked_normalize.assert_called_once_with(invalid_input)
    mocked_pallet_compose_call.assert_not_called()
    mocked_from_exception.assert_called_once_with(raise_error=False, error=test_error)
    assert response == mocked_from_exception.return_value


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


@pytest.mark.parametrize(
    "new_claim_type, expected_normalized, hotkey_ss58, netuid",
    [
        ("Swap", "Swap", "fake_hotkey_address", 1),
        ("Keep", "Keep", "fake_hotkey_address", 2),
        (RootClaimType.Swap, "Swap", "fake_hotkey_address", 1),
        (RootClaimType.Keep, "Keep", "fake_hotkey_address", 2),
        (
            {"KeepSubnets": {"subnets": [1, 2, 3]}},
            {"KeepSubnets": {"subnets": [1, 2, 3]}},
            "fake_hotkey_address",
            1,
        ),
        (
            RootClaimType.KeepSubnets([1, 2, 3]),
            {"KeepSubnets": {"subnets": [1, 2, 3]}},
            "fake_hotkey_address",
            2,
        ),
    ],
    ids=[
        "string-swap",
        "string-keep",
        "enum-swap",
        "enum-keep",
        "dict-keep-subnets",
        "callable-keep-subnets",
    ],
)
def test_set_validator_claim_type_extrinsic(
    subtensor,
    fake_wallet,
    mocker,
    new_claim_type,
    expected_normalized,
    hotkey_ss58,
    netuid,
):
    """Tests `set_validator_claim_type_extrinsic` extrinsic function with various input formats."""
    # Preps
    mocked_normalize = mocker.patch.object(
        RootClaimType, "normalize", return_value=expected_normalized
    )
    mocked_pallet_compose_call = mocker.patch.object(
        root.SubtensorModule, "set_validator_claim_type"
    )
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # call
    response = root.set_validator_claim_type_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
        new_claim_type=new_claim_type,
    )

    # asserts
    mocked_normalize.assert_called_once_with(new_claim_type)
    mocked_pallet_compose_call.assert_called_once_with(
        hotkey=hotkey_ss58,
        netuid=netuid,
        new_claim_type=expected_normalized,
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


def test_set_validator_claim_type_extrinsic_delegated_not_allowed(
    subtensor, fake_wallet, mocker
):
    """Tests that `set_validator_claim_type_extrinsic` raises ValueError for Delegated claim type."""
    # Preps
    mocked_normalize = mocker.patch.object(
        RootClaimType, "normalize", return_value="Delegated"
    )
    mocked_pallet_compose_call = mocker.patch.object(
        root.SubtensorModule, "set_validator_claim_type"
    )

    # call and assert
    with pytest.raises(
        ValueError, match="Delegated claim type cannot be set for validators"
    ):
        root.set_validator_claim_type_extrinsic(
            subtensor=subtensor,
            wallet=fake_wallet,
            hotkey_ss58="fake_hotkey_address",
            netuid=1,
            new_claim_type="Delegated",
            raise_error=True,
        )

    mocked_normalize.assert_called_once_with("Delegated")
    mocked_pallet_compose_call.assert_not_called()


def test_set_validator_claim_type_extrinsic_delegated_not_allowed_no_raise(
    subtensor, fake_wallet, mocker
):
    """Tests that `set_validator_claim_type_extrinsic` returns error response for Delegated claim type when raise_error=False."""
    # Preps
    mocked_normalize = mocker.patch.object(
        RootClaimType, "normalize", return_value="Delegated"
    )
    mocked_pallet_compose_call = mocker.patch.object(
        root.SubtensorModule, "set_validator_claim_type"
    )
    mocked_from_exception = mocker.patch.object(ExtrinsicResponse, "from_exception")

    # call
    response = root.set_validator_claim_type_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58="fake_hotkey_address",
        netuid=1,
        new_claim_type="Delegated",
        raise_error=False,
    )

    # assert
    mocked_normalize.assert_called_once_with("Delegated")
    mocked_pallet_compose_call.assert_not_called()
    mocked_from_exception.assert_called_once()
    assert response == mocked_from_exception.return_value


@pytest.mark.parametrize(
    "invalid_input, expected_error",
    [
        ("InvalidType", ValueError),
        ({"InvalidKey": {}}, ValueError),
        ({"KeepSubnets": {}}, ValueError),  # Empty subnets
        ({"KeepSubnets": {"subnets": []}}, ValueError),  # Empty subnets list
        (
            {"KeepSubnets": {"subnets": ["not", "integers"]}},
            ValueError,
        ),  # Non-integer subnets
        (123, TypeError),  # Wrong type
    ],
    ids=[
        "invalid-string",
        "invalid-dict-key",
        "empty-subnets-dict",
        "empty-subnets-list",
        "non-integer-subnets",
        "wrong-type",
    ],
)
def test_set_validator_claim_type_extrinsic_validation_with_raise_error(
    subtensor, fake_wallet, mocker, invalid_input, expected_error
):
    """Tests `set_validator_claim_type_extrinsic` validation for invalid inputs with raise_error=True."""
    # Preps
    test_error = expected_error("Test error")
    mocked_normalize = mocker.patch.object(
        RootClaimType, "normalize", side_effect=test_error
    )
    mocked_pallet_compose_call = mocker.patch.object(
        root.SubtensorModule, "set_validator_claim_type"
    )

    # call and assert
    with pytest.raises(expected_error):
        root.set_validator_claim_type_extrinsic(
            subtensor=subtensor,
            wallet=fake_wallet,
            hotkey_ss58="fake_hotkey_address",
            netuid=1,
            new_claim_type=invalid_input,
            raise_error=True,
        )

    mocked_normalize.assert_called_once_with(invalid_input)
    mocked_pallet_compose_call.assert_not_called()


@pytest.mark.parametrize(
    "invalid_input, expected_error",
    [
        ("InvalidType", ValueError),
        ({"InvalidKey": {}}, ValueError),
        ({"KeepSubnets": {"subnets": []}}, ValueError),  # Empty subnets list
        (123, TypeError),  # Wrong type
    ],
    ids=[
        "invalid-string-no-raise",
        "invalid-dict-key-no-raise",
        "empty-subnets-list-no-raise",
        "wrong-type-no-raise",
    ],
)
def test_set_validator_claim_type_extrinsic_validation_without_raise_error(
    subtensor, fake_wallet, mocker, invalid_input, expected_error
):
    """Tests `set_validator_claim_type_extrinsic` validation for invalid inputs with raise_error=False."""
    # Preps
    test_error = expected_error("Test error")
    mocked_normalize = mocker.patch.object(
        RootClaimType, "normalize", side_effect=test_error
    )
    mocked_pallet_compose_call = mocker.patch.object(
        root.SubtensorModule, "set_validator_claim_type"
    )
    mocked_from_exception = mocker.patch.object(ExtrinsicResponse, "from_exception")

    # call
    response = root.set_validator_claim_type_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58="fake_hotkey_address",
        netuid=1,
        new_claim_type=invalid_input,
        raise_error=False,
    )

    # assert
    mocked_normalize.assert_called_once_with(invalid_input)
    mocked_pallet_compose_call.assert_not_called()
    mocked_from_exception.assert_called_once_with(raise_error=False, error=test_error)
    assert response == mocked_from_exception.return_value
