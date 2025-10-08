import pytest

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
