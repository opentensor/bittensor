import pytest

from bittensor.core.chain_data import RootClaimType
from bittensor.core.extrinsics.asyncex import root as async_root
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import Balance


@pytest.mark.asyncio
async def test_get_limits_success(subtensor, mocker):
    """Tests successful retrieval of weight limits."""
    # Preps
    mocked_get_hyperparameter = mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        side_effect=[10, 100],
    )
    mocked_u16_normalized_float = mocker.patch.object(
        async_root,
        "u16_normalized_float",
        return_value=0.1,
    )

    # Call
    result = await async_root._get_limits(subtensor)

    # Asserts
    mocked_get_hyperparameter.assert_has_calls(
        [
            mocker.call("MinAllowedWeights", netuid=0),
            mocker.call("MaxWeightsLimit", netuid=0),
        ]
    )
    mocked_u16_normalized_float.assert_called_once_with(100)
    assert result == (10, 0.1)


@pytest.mark.asyncio
async def test_root_register_extrinsic_success(subtensor, fake_wallet, mocker):
    """Tests successful registration to root network."""
    # Preps
    fake_wallet.hotkey.ss58_address = "fake_hotkey_address"
    fake_wallet.hotkey_str = "fake_hotkey"
    fake_uid = 123

    mocked_unlock_key = mocker.patch.object(
        async_root.ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_is_hotkey_registered = mocker.patch.object(
        subtensor,
        "is_hotkey_registered",
        return_value=False,
    )
    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )
    mocked_query = mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=fake_uid,
    )
    mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=Balance(0),
    )
    mocker.patch.object(
        subtensor,
        "get_balance",
        return_value=Balance(1),
    )

    # Call
    result = await async_root.root_register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_unlock_key.assert_called_once_with(fake_wallet, False, unlock_type="both")
    mocked_is_hotkey_registered.assert_called_once_with(
        netuid=0, hotkey_ss58="fake_hotkey_address"
    )
    mocked_compose_call.assert_called_once()
    mocked_sign_and_send_extrinsic.assert_called_once()
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Uids",
        params=[0, "fake_hotkey_address"],
        block_hash=None,
        reuse_block_hash=False,
    )
    assert result.success is True
    assert result.message == "Success"


@pytest.mark.asyncio
async def test_root_register_extrinsic_insufficient_balance(
    subtensor,
    fake_wallet,
    mocker,
):
    mocked_unlock_key = mocker.patch.object(
        async_root.ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=Balance(1),
    )
    mocker.patch.object(
        subtensor,
        "get_balance",
        return_value=Balance(0),
    )

    result = await async_root.root_register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    mocked_unlock_key.assert_called_once_with(fake_wallet, False, unlock_type="both")
    assert result.success is False

    subtensor.get_balance.assert_called_once_with(
        fake_wallet.coldkeypub.ss58_address,
        block_hash=subtensor.substrate.get_chain_head.return_value,
    )
    subtensor.substrate.submit_extrinsic.assert_not_called()


@pytest.mark.asyncio
async def test_root_register_extrinsic_unlock_failed(subtensor, fake_wallet, mocker):
    """Tests registration fails due to unlock failure."""
    # Preps
    mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=Balance(0),
    )
    mocker.patch.object(
        subtensor,
        "get_balance",
        return_value=Balance(1),
    )
    mocked_unlock_key = mocker.patch.object(
        async_root.ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=False, message="Unlocked"),
    )

    # Call
    result = await async_root.root_register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_unlock_key.assert_called_once_with(fake_wallet, False, unlock_type="both")
    assert result.success is False


@pytest.mark.asyncio
async def test_root_register_extrinsic_already_registered(
    subtensor, fake_wallet, mocker
):
    """Tests registration when hotkey is already registered."""
    # Preps
    fake_wallet.hotkey.ss58_address = "fake_hotkey_address"

    mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=Balance(0),
    )
    mocker.patch.object(
        subtensor,
        "get_balance",
        return_value=Balance(1),
    )
    mocked_unlock_key = mocker.patch.object(
        async_root.ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_is_hotkey_registered = mocker.patch.object(
        subtensor,
        "is_hotkey_registered",
        return_value=True,
    )

    # Call
    result = await async_root.root_register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_unlock_key.assert_called_once_with(fake_wallet, False, unlock_type="both")
    mocked_is_hotkey_registered.assert_called_once_with(
        netuid=0, hotkey_ss58="fake_hotkey_address"
    )
    assert result.success is True


@pytest.mark.asyncio
async def test_root_register_extrinsic_transaction_failed(
    subtensor, fake_wallet, mocker
):
    """Tests registration fails due to transaction failure."""
    # Preps
    fake_wallet.hotkey.ss58_address = "fake_hotkey_address"

    mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=Balance(0),
    )
    mocker.patch.object(
        subtensor,
        "get_balance",
        return_value=Balance(1),
    )
    mocked_unlock_key = mocker.patch.object(
        async_root.ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_is_hotkey_registered = mocker.patch.object(
        subtensor,
        "is_hotkey_registered",
        return_value=False,
    )
    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(False, "Transaction failed"),
    )

    # Call
    result = await async_root.root_register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_unlock_key.assert_called_once_with(fake_wallet, False, unlock_type="both")
    mocked_is_hotkey_registered.assert_called_once_with(
        netuid=0, hotkey_ss58="fake_hotkey_address"
    )
    mocked_compose_call.assert_called_once()
    mocked_sign_and_send_extrinsic.assert_called_once()
    assert result.success is False


@pytest.mark.asyncio
async def test_root_register_extrinsic_uid_not_found(subtensor, fake_wallet, mocker):
    """Tests registration fails because UID is not found after successful transaction."""
    # Preps
    fake_wallet.hotkey.ss58_address = "fake_hotkey_address"

    mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=Balance(0),
    )
    mocker.patch.object(
        subtensor,
        "get_balance",
        return_value=Balance(1),
    )
    mocked_unlock_key = mocker.patch.object(
        async_root.ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_is_hotkey_registered = mocker.patch.object(
        subtensor,
        "is_hotkey_registered",
        return_value=False,
    )
    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, ""),
    )
    mocked_query = mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=None,
    )

    # Call
    result = await async_root.root_register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_unlock_key.assert_called_once_with(fake_wallet, False, unlock_type="both")
    mocked_is_hotkey_registered.assert_called_once_with(
        netuid=0, hotkey_ss58="fake_hotkey_address"
    )
    mocked_compose_call.assert_called_once()
    mocked_sign_and_send_extrinsic.assert_called_once()
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Uids",
        params=[0, "fake_hotkey_address"],
        block_hash=None,
        reuse_block_hash=False,
    )
    assert result.success is False


@pytest.mark.asyncio
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
async def test_set_root_claim_type_extrinsic(
    subtensor, fake_wallet, mocker, new_root_claim_type, expected_normalized
):
    """Tests `set_root_claim_type_extrinsic` extrinsic function with various input formats."""
    # Preps
    mocked_normalize = mocker.patch.object(
        RootClaimType, "normalize", return_value=expected_normalized
    )
    mocked_pallet_compose_call = mocker.patch.object(
        async_root.SubtensorModule, "set_root_claim_type", new=mocker.AsyncMock()
    )
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # call
    response = await async_root.set_root_claim_type_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        new_root_claim_type=new_root_claim_type,
    )

    # asserts
    mocked_normalize.assert_called_once_with(new_root_claim_type)
    mocked_pallet_compose_call.assert_awaited_once_with(
        new_root_claim_type=expected_normalized,
    )
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
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
@pytest.mark.asyncio
async def test_set_root_claim_type_extrinsic_validation_with_raise_error(
    subtensor, fake_wallet, mocker, invalid_input, expected_error
):
    """Tests `set_root_claim_type_extrinsic` validation for invalid inputs with raise_error=True."""
    # Preps
    test_error = expected_error("Test error")
    mocked_normalize = mocker.patch.object(
        RootClaimType, "normalize", side_effect=test_error
    )
    mocked_pallet_compose_call = mocker.patch.object(
        async_root.SubtensorModule, "set_root_claim_type", new=mocker.AsyncMock()
    )

    # call and assert
    with pytest.raises(expected_error):
        await async_root.set_root_claim_type_extrinsic(
            subtensor=subtensor,
            wallet=fake_wallet,
            new_root_claim_type=invalid_input,
            raise_error=True,
        )

    mocked_normalize.assert_called_once_with(invalid_input)
    mocked_pallet_compose_call.assert_not_awaited()


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
@pytest.mark.asyncio
async def test_set_root_claim_type_extrinsic_validation_without_raise_error(
    subtensor, fake_wallet, mocker, invalid_input, expected_error
):
    """Tests `set_root_claim_type_extrinsic` validation for invalid inputs with raise_error=False."""
    # Preps
    test_error = expected_error("Test error")
    mocked_normalize = mocker.patch.object(
        RootClaimType, "normalize", side_effect=test_error
    )
    mocked_pallet_compose_call = mocker.patch.object(
        async_root.SubtensorModule, "set_root_claim_type", new=mocker.AsyncMock()
    )
    mocked_from_exception = mocker.patch.object(ExtrinsicResponse, "from_exception")

    # call
    response = await async_root.set_root_claim_type_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        new_root_claim_type=invalid_input,
        raise_error=False,
    )

    # assert
    mocked_normalize.assert_called_once_with(invalid_input)
    mocked_pallet_compose_call.assert_not_awaited()
    mocked_from_exception.assert_called_once_with(raise_error=False, error=test_error)
    assert response == mocked_from_exception.return_value


@pytest.mark.asyncio
async def test_claim_root_extrinsic(subtensor, fake_wallet, mocker):
    """Tests `claim_root_extrinsic` extrinsic function."""
    # Preps
    netuids = mocker.Mock(spec=list)
    mocked_pallet_compose_call = mocker.patch.object(
        async_root.SubtensorModule, "claim_root", new=mocker.AsyncMock()
    )
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # call
    response = await async_root.claim_root_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuids=netuids,
    )

    # asserts
    mocked_pallet_compose_call.assert_called_once_with(subnets=netuids)
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_pallet_compose_call.return_value,
        wallet=fake_wallet,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value


@pytest.mark.asyncio
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
async def test_set_validator_claim_type_extrinsic(
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
        async_root.SubtensorModule, "set_validator_claim_type", new=mocker.AsyncMock()
    )
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # call
    response = await async_root.set_validator_claim_type_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
        new_claim_type=new_claim_type,
    )

    # asserts
    mocked_normalize.assert_called_once_with(new_claim_type)
    mocked_pallet_compose_call.assert_awaited_once_with(
        hotkey=hotkey_ss58,
        netuid=netuid,
        new_claim_type=expected_normalized,
    )
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_pallet_compose_call.return_value,
        wallet=fake_wallet,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value


@pytest.mark.asyncio
async def test_set_validator_claim_type_extrinsic_delegated_not_allowed(
    subtensor, fake_wallet, mocker
):
    """Tests that `set_validator_claim_type_extrinsic` raises ValueError for Delegated claim type."""
    # Preps
    mocked_normalize = mocker.patch.object(
        RootClaimType, "normalize", return_value="Delegated"
    )
    mocked_pallet_compose_call = mocker.patch.object(
        async_root.SubtensorModule, "set_validator_claim_type", new=mocker.AsyncMock()
    )

    # call and assert
    with pytest.raises(
        ValueError, match="Delegated claim type cannot be set for validators"
    ):
        await async_root.set_validator_claim_type_extrinsic(
            subtensor=subtensor,
            wallet=fake_wallet,
            hotkey_ss58="fake_hotkey_address",
            netuid=1,
            new_claim_type="Delegated",
            raise_error=True,
        )

    mocked_normalize.assert_called_once_with("Delegated")
    mocked_pallet_compose_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_validator_claim_type_extrinsic_delegated_not_allowed_no_raise(
    subtensor, fake_wallet, mocker
):
    """Tests that `set_validator_claim_type_extrinsic` returns error response for Delegated claim type when raise_error=False."""
    # Preps
    mocked_normalize = mocker.patch.object(
        RootClaimType, "normalize", return_value="Delegated"
    )
    mocked_pallet_compose_call = mocker.patch.object(
        async_root.SubtensorModule, "set_validator_claim_type", new=mocker.AsyncMock()
    )
    mocked_from_exception = mocker.patch.object(ExtrinsicResponse, "from_exception")

    # call
    response = await async_root.set_validator_claim_type_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58="fake_hotkey_address",
        netuid=1,
        new_claim_type="Delegated",
        raise_error=False,
    )

    # assert
    mocked_normalize.assert_called_once_with("Delegated")
    mocked_pallet_compose_call.assert_not_awaited()
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
@pytest.mark.asyncio
async def test_set_validator_claim_type_extrinsic_validation_with_raise_error(
    subtensor, fake_wallet, mocker, invalid_input, expected_error
):
    """Tests `set_validator_claim_type_extrinsic` validation for invalid inputs with raise_error=True."""
    # Preps
    test_error = expected_error("Test error")
    mocked_normalize = mocker.patch.object(
        RootClaimType, "normalize", side_effect=test_error
    )
    mocked_pallet_compose_call = mocker.patch.object(
        async_root.SubtensorModule, "set_validator_claim_type", new=mocker.AsyncMock()
    )

    # call and assert
    with pytest.raises(expected_error):
        await async_root.set_validator_claim_type_extrinsic(
            subtensor=subtensor,
            wallet=fake_wallet,
            hotkey_ss58="fake_hotkey_address",
            netuid=1,
            new_claim_type=invalid_input,
            raise_error=True,
        )

    mocked_normalize.assert_called_once_with(invalid_input)
    mocked_pallet_compose_call.assert_not_awaited()


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
@pytest.mark.asyncio
async def test_set_validator_claim_type_extrinsic_validation_without_raise_error(
    subtensor, fake_wallet, mocker, invalid_input, expected_error
):
    """Tests `set_validator_claim_type_extrinsic` validation for invalid inputs with raise_error=False."""
    # Preps
    test_error = expected_error("Test error")
    mocked_normalize = mocker.patch.object(
        RootClaimType, "normalize", side_effect=test_error
    )
    mocked_pallet_compose_call = mocker.patch.object(
        async_root.SubtensorModule, "set_validator_claim_type", new=mocker.AsyncMock()
    )
    mocked_from_exception = mocker.patch.object(ExtrinsicResponse, "from_exception")

    # call
    response = await async_root.set_validator_claim_type_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58="fake_hotkey_address",
        netuid=1,
        new_claim_type=invalid_input,
        raise_error=False,
    )

    # assert
    mocked_normalize.assert_called_once_with(invalid_input)
    mocked_pallet_compose_call.assert_not_awaited()
    mocked_from_exception.assert_called_once_with(raise_error=False, error=test_error)
    assert response == mocked_from_exception.return_value
