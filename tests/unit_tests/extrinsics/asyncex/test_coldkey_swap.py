import pytest
from bittensor_wallet import Wallet
from scalecodec.types import GenericCall

from bittensor.core.extrinsics.asyncex import coldkey_swap
from bittensor.core.extrinsics.pallets import SubtensorModule
from bittensor.core.settings import DEFAULT_MEV_PROTECTION
from bittensor.core.types import ExtrinsicResponse
from bittensor.core.chain_data.coldkey_swap import ColdkeySwapAnnouncementInfo


@pytest.mark.asyncio
async def test_announce_coldkey_swap_extrinsic(subtensor, mocker):
    """Verify that async `announce_coldkey_swap_extrinsic` method calls proper methods."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    wallet.coldkeypub.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    new_coldkey_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_keypair = mocker.patch("bittensor.core.extrinsics.asyncex.coldkey_swap.Keypair")
    mocked_keypair_instance = mocker.MagicMock()
    mocked_keypair_instance.public_key = b"\x00" * 32
    mocked_keypair.return_value = mocked_keypair_instance

    mocked_compute_hash = mocker.patch.object(
        coldkey_swap, "compute_coldkey_hash", return_value="0x" + "00" * 32
    )
    mocked_subtensor_module = mocker.patch.object(
        coldkey_swap, "SubtensorModule", return_value=mocker.MagicMock()
    )
    mocked_pallet_instance = mocked_subtensor_module.return_value
    mocked_pallet_instance.announce_coldkey_swap = mocker.AsyncMock(
        return_value=mocker.MagicMock()
    )
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    response = await coldkey_swap.announce_coldkey_swap_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        new_coldkey_ss58=new_coldkey_ss58,
        mev_protection=False,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(wallet, False)
    mocked_keypair.assert_called_once_with(ss58_address=new_coldkey_ss58)
    mocked_compute_hash.assert_called_once_with(mocked_keypair_instance)
    mocked_subtensor_module.assert_called_once_with(subtensor)
    mocked_pallet_instance.announce_coldkey_swap.assert_awaited_once_with(
        new_coldkey_hash="0x" + "00" * 32
    )
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_pallet_instance.announce_coldkey_swap.return_value,
        wallet=wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value


@pytest.mark.asyncio
async def test_announce_coldkey_swap_extrinsic_with_mev_protection(subtensor, mocker):
    """Verify that async `announce_coldkey_swap_extrinsic` uses MEV protection when enabled."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    wallet.coldkeypub.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    new_coldkey_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_keypair = mocker.patch("bittensor.core.extrinsics.asyncex.coldkey_swap.Keypair")
    mocked_keypair_instance = mocker.MagicMock()
    mocked_keypair_instance.public_key = b"\x00" * 32
    mocked_keypair.return_value = mocked_keypair_instance

    mocked_compute_hash = mocker.patch.object(
        coldkey_swap, "compute_coldkey_hash", return_value="0x" + "00" * 32
    )
    mocked_subtensor_module = mocker.patch.object(
        coldkey_swap, "SubtensorModule", return_value=mocker.MagicMock()
    )
    mocked_pallet_instance = mocked_subtensor_module.return_value
    mocked_pallet_instance.announce_coldkey_swap = mocker.AsyncMock(
        return_value=mocker.MagicMock()
    )
    mocked_submit_encrypted = mocker.patch.object(
        coldkey_swap,
        "submit_encrypted_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    response = await coldkey_swap.announce_coldkey_swap_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        new_coldkey_ss58=new_coldkey_ss58,
        mev_protection=True,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(wallet, False)
    mocked_subtensor_module.assert_called_once_with(subtensor)
    mocked_pallet_instance.announce_coldkey_swap.assert_awaited_once_with(
        new_coldkey_hash="0x" + "00" * 32
    )
    mocked_submit_encrypted.assert_awaited_once()
    assert response == mocked_submit_encrypted.return_value


@pytest.mark.asyncio
async def test_swap_coldkey_announced_extrinsic_success(subtensor, mocker):
    """Verify that async `swap_coldkey_announced_extrinsic` method calls proper methods."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    wallet.coldkeypub.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    new_coldkey_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    fake_hash = "0x" + "00" * 32

    announcement = ColdkeySwapAnnouncementInfo(
        coldkey=wallet.coldkeypub.ss58_address,
        execution_block=1000,
        new_coldkey_hash=fake_hash,
    )

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_get_announcement = mocker.patch.object(
        subtensor, "get_coldkey_swap_announcement", return_value=announcement
    )
    mocked_get_current_block = mocker.patch.object(
        subtensor, "get_current_block", return_value=1001
    )
    mocked_keypair = mocker.patch("bittensor.core.extrinsics.asyncex.coldkey_swap.Keypair")
    mocked_keypair_instance = mocker.MagicMock()
    mocked_keypair_instance.public_key = b"\x00" * 32
    mocked_keypair.return_value = mocked_keypair_instance

    mocked_verify_hash = mocker.patch.object(
        coldkey_swap, "verify_coldkey_hash", return_value=True
    )
    mocked_subtensor_module = mocker.patch.object(
        coldkey_swap, "SubtensorModule", return_value=mocker.MagicMock()
    )
    mocked_pallet_instance = mocked_subtensor_module.return_value
    mocked_pallet_instance.swap_coldkey_announced = mocker.AsyncMock(
        return_value=mocker.MagicMock()
    )
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    response = await coldkey_swap.swap_coldkey_announced_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        new_coldkey_ss58=new_coldkey_ss58,
        mev_protection=False,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(wallet, False)
    mocked_get_announcement.assert_awaited_once_with(
        coldkey_ss58=wallet.coldkeypub.ss58_address
    )
    mocked_get_current_block.assert_awaited_once()
    mocked_verify_hash.assert_called_once_with(mocked_keypair_instance, fake_hash)
    mocked_subtensor_module.assert_called_once_with(subtensor)
    mocked_pallet_instance.swap_coldkey_announced.assert_awaited_once_with(
        new_coldkey=new_coldkey_ss58
    )
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_pallet_instance.swap_coldkey_announced.return_value,
        wallet=wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value


@pytest.mark.asyncio
async def test_swap_coldkey_announced_extrinsic_no_announcement(subtensor, mocker):
    """Verify that async `swap_coldkey_announced_extrinsic` returns error when no announcement."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    wallet.coldkeypub.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    new_coldkey_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_get_announcement = mocker.patch.object(
        subtensor, "get_coldkey_swap_announcement", return_value=None
    )

    # Call
    response = await coldkey_swap.swap_coldkey_announced_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        new_coldkey_ss58=new_coldkey_ss58,
        raise_error=False,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(wallet, False)
    mocked_get_announcement.assert_awaited_once_with(
        coldkey_ss58=wallet.coldkeypub.ss58_address
    )
    assert response.success is False
    assert "No coldkey swap announcement found" in response.message


@pytest.mark.asyncio
async def test_swap_coldkey_announced_extrinsic_hash_mismatch(subtensor, mocker):
    """Verify that async `swap_coldkey_announced_extrinsic` returns error when hash doesn't match."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    wallet.coldkeypub.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    new_coldkey_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    fake_hash = "0x" + "00" * 32

    announcement = ColdkeySwapAnnouncementInfo(
        coldkey=wallet.coldkeypub.ss58_address,
        execution_block=1000,
        new_coldkey_hash=fake_hash,
    )

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_get_announcement = mocker.patch.object(
        subtensor, "get_coldkey_swap_announcement", return_value=announcement
    )
    mocked_keypair = mocker.patch("bittensor.core.extrinsics.asyncex.coldkey_swap.Keypair")
    mocked_keypair_instance = mocker.MagicMock()
    mocked_keypair_instance.public_key = b"\x00" * 32
    mocked_keypair.return_value = mocked_keypair_instance

    mocked_verify_hash = mocker.patch.object(
        coldkey_swap, "verify_coldkey_hash", return_value=False
    )
    mocked_compute_hash = mocker.patch.object(
        coldkey_swap, "compute_coldkey_hash", return_value="0x" + "11" * 32
    )

    # Call
    response = await coldkey_swap.swap_coldkey_announced_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        new_coldkey_ss58=new_coldkey_ss58,
        raise_error=False,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(wallet, False)
    mocked_get_announcement.assert_awaited_once_with(
        coldkey_ss58=wallet.coldkeypub.ss58_address
    )
    mocked_verify_hash.assert_called_once_with(mocked_keypair_instance, fake_hash)
    assert response.success is False
    assert "hash does not match" in response.message.lower()


@pytest.mark.asyncio
async def test_swap_coldkey_announced_extrinsic_too_early(subtensor, mocker):
    """Verify that async `swap_coldkey_announced_extrinsic` returns error when too early."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    wallet.coldkeypub.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    new_coldkey_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    fake_hash = "0x" + "00" * 32

    announcement = ColdkeySwapAnnouncementInfo(
        coldkey=wallet.coldkeypub.ss58_address,
        execution_block=1000,
        new_coldkey_hash=fake_hash,
    )

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_get_announcement = mocker.patch.object(
        subtensor, "get_coldkey_swap_announcement", return_value=announcement
    )
    mocked_get_current_block = mocker.patch.object(
        subtensor, "get_current_block", return_value=999
    )
    mocked_keypair = mocker.patch("bittensor.core.extrinsics.asyncex.coldkey_swap.Keypair")
    mocked_keypair_instance = mocker.MagicMock()
    mocked_keypair_instance.public_key = b"\x00" * 32
    mocked_keypair.return_value = mocked_keypair_instance

    mocked_verify_hash = mocker.patch.object(
        coldkey_swap, "verify_coldkey_hash", return_value=True
    )

    # Call
    response = await coldkey_swap.swap_coldkey_announced_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        new_coldkey_ss58=new_coldkey_ss58,
        raise_error=False,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(wallet, False)
    mocked_get_announcement.assert_awaited_once_with(
        coldkey_ss58=wallet.coldkeypub.ss58_address
    )
    mocked_get_current_block.assert_awaited_once()
    mocked_verify_hash.assert_called_once_with(mocked_keypair_instance, fake_hash)
    assert response.success is False
    assert "too early" in response.message.lower()
    assert "999" in response.message
    assert "1000" in response.message


@pytest.mark.asyncio
async def test_remove_coldkey_swap_announcement_extrinsic(subtensor, mocker):
    """Verify that async `remove_coldkey_swap_announcement_extrinsic` method calls proper methods."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    coldkey_ss58 = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_subtensor_module = mocker.patch.object(
        coldkey_swap, "SubtensorModule", return_value=mocker.MagicMock()
    )
    mocked_pallet_instance = mocked_subtensor_module.return_value
    mocked_pallet_instance.remove_coldkey_swap_announcement = mocker.AsyncMock(
        return_value=mocker.MagicMock()
    )
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    response = await coldkey_swap.remove_coldkey_swap_announcement_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        coldkey_ss58=coldkey_ss58,
        mev_protection=False,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(wallet, False)
    mocked_subtensor_module.assert_called_once_with(subtensor)
    mocked_pallet_instance.remove_coldkey_swap_announcement.assert_awaited_once_with(
        coldkey=coldkey_ss58
    )
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_pallet_instance.remove_coldkey_swap_announcement.return_value,
        wallet=wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value

