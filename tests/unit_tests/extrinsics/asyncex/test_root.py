import pytest

from bittensor.core.errors import SubstrateRequestException
from bittensor.core.extrinsics.asyncex import root as async_root

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
        async_root,
        "unlock_key",
        return_value=mocker.Mock(success=True, message="Unlocked"),
    )
    mocked_is_hotkey_registered = mocker.patch.object(
        subtensor,
        "is_hotkey_registered",
        return_value=False,
    )
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=(True, ""),
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
    mocked_unlock_key.assert_called_once_with(fake_wallet)
    mocked_is_hotkey_registered.assert_called_once_with(
        netuid=0, hotkey_ss58="fake_hotkey_address"
    )
    mocked_compose_call.assert_called_once()
    mocked_sign_and_send_extrinsic.assert_called_once()
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Uids",
        params=[0, "fake_hotkey_address"],
    )
    assert result is True


@pytest.mark.asyncio
async def test_root_register_extrinsic_insufficient_balance(
    subtensor,
    fake_wallet,
    mocker,
):
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

    assert result is False

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
        async_root,
        "unlock_key",
        return_value=mocker.Mock(success=False, message="Unlock failed"),
    )

    # Call
    result = await async_root.root_register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_unlock_key.assert_called_once_with(fake_wallet)
    assert result is False


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
        async_root,
        "unlock_key",
        return_value=mocker.Mock(success=True, message="Unlocked"),
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
    mocked_unlock_key.assert_called_once_with(fake_wallet)
    mocked_is_hotkey_registered.assert_called_once_with(
        netuid=0, hotkey_ss58="fake_hotkey_address"
    )
    assert result is True


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
        async_root,
        "unlock_key",
        return_value=mocker.Mock(success=True, message="Unlocked"),
    )
    mocked_is_hotkey_registered = mocker.patch.object(
        subtensor,
        "is_hotkey_registered",
        return_value=False,
    )
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=(False, "Transaction failed"),
    )

    # Call
    result = await async_root.root_register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_unlock_key.assert_called_once_with(fake_wallet)
    mocked_is_hotkey_registered.assert_called_once_with(
        netuid=0, hotkey_ss58="fake_hotkey_address"
    )
    mocked_compose_call.assert_called_once()
    mocked_sign_and_send_extrinsic.assert_called_once()
    assert result is False


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
        async_root,
        "unlock_key",
        return_value=mocker.Mock(success=True, message="Unlocked"),
    )
    mocked_is_hotkey_registered = mocker.patch.object(
        subtensor,
        "is_hotkey_registered",
        return_value=False,
    )
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=(True, ""),
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
    mocked_unlock_key.assert_called_once_with(fake_wallet)
    mocked_is_hotkey_registered.assert_called_once_with(
        netuid=0, hotkey_ss58="fake_hotkey_address"
    )
    mocked_compose_call.assert_called_once()
    mocked_sign_and_send_extrinsic.assert_called_once()
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Uids",
        params=[0, "fake_hotkey_address"],
    )
    assert result is False


@pytest.mark.asyncio
async def test_set_root_weights_extrinsic_success(subtensor, fake_wallet, mocker):
    """Tests successful setting of root weights."""
    fake_wallet.hotkey.ss58_address = "fake_hotkey"
    netuids = [1, 2, 3]
    weights = [0.1, 0.2, 0.7]

    mocker.patch.object(subtensor.substrate, "query", return_value=123)
    mocker.patch.object(
        async_root, "unlock_key", return_value=mocker.Mock(success=True)
    )
    mocker.patch.object(async_root, "_get_limits", return_value=(2, 1.0))
    mocker.patch.object(async_root, "normalize_max_weight", return_value=weights)
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=(True, ""),
    )

    result = await async_root.set_root_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuids=netuids,
        weights=weights,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        use_nonce=True,
    )
    assert result is True


@pytest.mark.asyncio
async def test_set_root_weights_extrinsic_no_waiting(subtensor, fake_wallet, mocker):
    """Tests setting root weights without waiting for inclusion or finalization."""
    fake_wallet.hotkey.ss58_address = "fake_hotkey"
    netuids = [1, 2, 3]
    weights = [0.1, 0.2, 0.7]

    mocker.patch.object(subtensor.substrate, "query", return_value=123)
    mocker.patch.object(
        async_root, "unlock_key", return_value=mocker.Mock(success=True)
    )
    mocker.patch.object(async_root, "_get_limits", return_value=(2, 1.0))
    mocker.patch.object(async_root, "normalize_max_weight", return_value=weights)
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=(True, ""),
    )

    result = await async_root.set_root_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuids=netuids,
        weights=weights,
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )

    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=False,
        wait_for_finalization=False,
        period=None,
        use_nonce=True,
    )
    assert result is True


@pytest.mark.asyncio
async def test_set_root_weights_extrinsic_not_registered(
    subtensor, fake_wallet, mocker
):
    """Tests failure when hotkey is not registered."""
    fake_wallet.hotkey.ss58_address = "fake_hotkey"

    mocker.patch.object(subtensor.substrate, "query", return_value=None)

    result = await async_root.set_root_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuids=[1, 2, 3],
        weights=[0.1, 0.2, 0.7],
    )

    assert result is False


@pytest.mark.asyncio
async def test_set_root_weights_extrinsic_insufficient_weights(
    subtensor, fake_wallet, mocker
):
    """Tests failure when number of weights is less than the minimum allowed."""
    fake_wallet.hotkey.ss58_address = "fake_hotkey"
    netuids = [1, 2]
    weights = [0.5, 0.5]

    mocker.patch.object(subtensor.substrate, "query", return_value=123)
    mocker.patch.object(
        async_root, "unlock_key", return_value=mocker.Mock(success=True)
    )
    mocker.patch.object(async_root, "_get_limits", return_value=(3, 1.0))

    with pytest.raises(ValueError):
        await async_root.set_root_weights_extrinsic(
            subtensor=subtensor,
            wallet=fake_wallet,
            netuids=netuids,
            weights=weights,
        )


@pytest.mark.asyncio
async def test_set_root_weights_extrinsic_unlock_failed(subtensor, fake_wallet, mocker):
    """Tests failure due to unlock key error."""
    fake_wallet.hotkey.ss58_address = "fake_hotkey"

    mocker.patch.object(subtensor.substrate, "query", return_value=123)
    mocker.patch.object(
        async_root,
        "unlock_key",
        return_value=mocker.Mock(success=False, message="Unlock failed"),
    )

    result = await async_root.set_root_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuids=[1, 2, 3],
        weights=[0.1, 0.2, 0.7],
    )

    assert result is False


@pytest.mark.asyncio
async def test_set_root_weights_extrinsic_transaction_failed(
    subtensor, fake_wallet, mocker
):
    """Tests failure when transaction is not successful."""
    fake_wallet.hotkey.ss58_address = "fake_hotkey"

    mocker.patch.object(subtensor.substrate, "query", return_value=123)
    mocker.patch.object(
        async_root, "unlock_key", return_value=mocker.Mock(success=True)
    )
    mocker.patch.object(async_root, "_get_limits", return_value=(2, 1.0))
    mocker.patch.object(
        async_root, "normalize_max_weight", return_value=[0.1, 0.2, 0.7]
    )
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=(True, ""),
    )

    result = await async_root.set_root_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuids=[1, 2, 3],
        weights=[0.1, 0.2, 0.7],
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        use_nonce=True,
    )
    assert result is True


@pytest.mark.asyncio
async def test_set_root_weights_extrinsic_request_exception(
    subtensor, fake_wallet, mocker
):
    """Tests failure due to SubstrateRequestException."""
    fake_wallet.hotkey.ss58_address = "fake_hotkey"

    mocker.patch.object(subtensor.substrate, "query", return_value=123)
    mocker.patch.object(
        async_root, "unlock_key", return_value=mocker.Mock(success=True)
    )
    mocker.patch.object(async_root, "_get_limits", return_value=(2, 1.0))
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        side_effect=SubstrateRequestException("Request failed"),
    )
    mocked_format_error_message = mocker.patch.object(
        async_root, "format_error_message"
    )

    result = await async_root.set_root_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuids=[1, 2, 3],
        weights=[0.1, 0.2, 0.7],
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert result is False
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        use_nonce=True,
    )
    mocked_format_error_message.assert_called_once()
