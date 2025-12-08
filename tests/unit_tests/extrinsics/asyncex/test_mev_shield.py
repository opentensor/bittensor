"""Unit tests for async MEV Shield extrinsics."""

from unittest.mock import AsyncMock

import pytest

from bittensor.core.extrinsics.asyncex import mev_shield
from bittensor.core.types import ExtrinsicResponse

# Constants
ML_KEM_768_KEY_SIZE = 1184
MOCK_COMMITMENT = "0x9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
MOCK_CIPHERTEXT = b"encrypted_data"
MOCK_PAYLOAD_CORE = b"payload_core"
MOCK_EXTRINSIC_HASH = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
MOCK_EXTRINSIC_HASH_HEX = f"0x{MOCK_EXTRINSIC_HASH}"
MOCK_BLOCK_HASH = "0x7d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e"
MOCK_BLOCK_HASH_101 = "0x8e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f"
MOCK_SUBMIT_BLOCK_HASH = "0x6c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d"
MOCK_STARTING_BLOCK = 100


@pytest.mark.asyncio
async def test_submit_encrypted_extrinsic_success(subtensor, fake_wallet, mocker):
    """
    Test description: Verifies successful encryption and submission of an extrinsic.

    Given: A valid wallet, call, and MEV Shield NextKey available
    When: submit_encrypted_extrinsic is called with default parameters
    Then: Commitment and ciphertext are generated and extrinsic is submitted successfully
    """
    # Arrange
    mock_call = mocker.Mock()
    mock_signed_extrinsic = mocker.Mock()
    mock_signed_extrinsic.extrinsic_hash.hex.return_value = MOCK_EXTRINSIC_HASH
    mock_signed_extrinsic.data.data = b"payload_core_data"

    mock_ml_kem_key = b"x" * ML_KEM_768_KEY_SIZE

    mocker.patch.object(
        mev_shield.ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocker.patch.object(
        subtensor,
        "get_mev_shield_next_key",
        new=AsyncMock(return_value=mock_ml_kem_key),
    )
    mocker.patch.object(
        subtensor.substrate,
        "get_account_next_index",
        new=AsyncMock(return_value=0),
    )
    mocker.patch.object(
        subtensor.substrate,
        "create_signed_extrinsic",
        new=AsyncMock(return_value=mock_signed_extrinsic),
    )

    mocker.patch.object(
        mev_shield,
        "get_mev_commitment_and_ciphertext",
        return_value=(MOCK_COMMITMENT, MOCK_CIPHERTEXT, MOCK_PAYLOAD_CORE),
    )

    mock_extrinsic_call = mocker.Mock()
    mock_mev_shield_pallet = mocker.patch.object(
        mev_shield,
        "MevShield",
    )
    mock_mev_shield_pallet.return_value.submit_encrypted = AsyncMock(
        return_value=mock_extrinsic_call
    )

    mock_response = ExtrinsicResponse(success=True, message="")
    mock_response.extrinsic_receipt = mocker.Mock()
    mock_response.extrinsic_receipt.block_hash = MOCK_BLOCK_HASH
    mock_response.extrinsic_receipt.triggered_events = AsyncMock(
        return_value=[
            {
                "module_id": "mevShield",
                "event_id": "EncryptedSubmitted",
                "attributes": {"id": "shield_123"},
            }
        ]
    )()
    mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        new=AsyncMock(return_value=mock_response),
    )

    mock_mev_receipt = mocker.Mock()
    mock_mev_receipt.is_success = AsyncMock(return_value=True)()
    mocker.patch.object(
        mev_shield,
        "wait_for_extrinsic_by_hash",
        new=AsyncMock(return_value=mock_mev_receipt),
    )

    # Act
    result = await mev_shield.submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=mock_call,
        sign_with="coldkey",
        wait_for_inclusion=True,
        wait_for_revealed_execution=True,
    )

    # Assert
    assert result.success is True
    assert result.data["commitment"] == MOCK_COMMITMENT
    assert result.data["ciphertext"] == MOCK_CIPHERTEXT
    assert result.data["ml_kem_768_public_key"] == b"x" * ML_KEM_768_KEY_SIZE
    assert result.data["payload_core"] == MOCK_PAYLOAD_CORE
    assert result.data["signed_extrinsic_hash"] == MOCK_EXTRINSIC_HASH_HEX


@pytest.mark.asyncio
async def test_submit_encrypted_extrinsic_sign_with_hotkey(
    subtensor, fake_wallet, mocker
):
    """
    Test description: Verifies signing with hotkey instead of coldkey.

    Given: A valid wallet with both coldkey and hotkey
    When: submit_encrypted_extrinsic is called with sign_with="hotkey"
    Then: The hotkey is used for signing the inner extrinsic
    """
    # Arrange
    mock_call = mocker.Mock()
    mock_signed_extrinsic = mocker.Mock()
    mock_signed_extrinsic.extrinsic_hash.hex.return_value = MOCK_EXTRINSIC_HASH
    mock_signed_extrinsic.data.data = b"payload_core_data"

    mock_ml_kem_key = b"y" * ML_KEM_768_KEY_SIZE

    mocker.patch.object(
        mev_shield.ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocker.patch.object(
        subtensor,
        "get_mev_shield_next_key",
        new=AsyncMock(return_value=mock_ml_kem_key),
    )
    mocker.patch.object(
        subtensor.substrate,
        "get_account_next_index",
        new=AsyncMock(return_value=0),
    )
    mock_create_signed = mocker.patch.object(
        subtensor.substrate,
        "create_signed_extrinsic",
        new=AsyncMock(return_value=mock_signed_extrinsic),
    )

    mocker.patch.object(
        mev_shield,
        "get_mev_commitment_and_ciphertext",
        return_value=(MOCK_COMMITMENT, MOCK_CIPHERTEXT, MOCK_PAYLOAD_CORE),
    )

    mock_extrinsic_call = mocker.Mock()
    mock_mev_shield_pallet = mocker.patch.object(
        mev_shield,
        "MevShield",
    )
    mock_mev_shield_pallet.return_value.submit_encrypted = AsyncMock(
        return_value=mock_extrinsic_call
    )

    mock_response = ExtrinsicResponse(success=True, message="")
    mock_response.extrinsic_receipt = mocker.Mock()
    mock_response.extrinsic_receipt.block_hash = MOCK_BLOCK_HASH
    mock_response.extrinsic_receipt.triggered_events = AsyncMock(
        return_value=[
            {
                "module_id": "mevShield",
                "event_id": "EncryptedSubmitted",
                "attributes": {"id": "shield_456"},
            }
        ]
    )()
    mock_sign_and_send = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        new=AsyncMock(return_value=mock_response),
    )

    mock_mev_receipt = mocker.Mock()
    mock_mev_receipt.is_success = AsyncMock(return_value=True)()
    mocker.patch.object(
        mev_shield,
        "wait_for_extrinsic_by_hash",
        new=AsyncMock(return_value=mock_mev_receipt),
    )

    # Act
    result = await mev_shield.submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=mock_call,
        sign_with="hotkey",
        wait_for_inclusion=True,
        wait_for_revealed_execution=True,
    )

    # Assert
    assert result.success is True
    # Verify hotkey was used for signing inner extrinsic
    mock_create_signed.assert_awaited_once()
    assert mock_create_signed.call_args.kwargs["keypair"] is fake_wallet.hotkey
    # Verify sign_and_send_extrinsic was called with hotkey
    mock_sign_and_send.assert_awaited_once()
    assert mock_sign_and_send.call_args.kwargs["sign_with"] == "hotkey"


@pytest.mark.asyncio
async def test_wait_for_extrinsic_by_hash_success(subtensor, mocker):
    """
    Test description: Verifies successful extrinsic discovery in subsequent block.

    Given: An extrinsic hash and shield ID
    When: wait_for_extrinsic_by_hash is called and the extrinsic is found
    Then: An AsyncExtrinsicReceipt is returned with correct block info
    """
    # Arrange
    extrinsic_hash = MOCK_EXTRINSIC_HASH_HEX
    shield_id = "shield_001"
    submit_block_hash = MOCK_SUBMIT_BLOCK_HASH

    mocker.patch.object(
        subtensor.substrate,
        "get_block_number",
        new=AsyncMock(return_value=MOCK_STARTING_BLOCK),
    )
    mocker.patch.object(subtensor, "wait_for_block", new=AsyncMock())
    mocker.patch.object(
        subtensor.substrate,
        "get_block_hash",
        new=AsyncMock(return_value=MOCK_BLOCK_HASH_101),
    )

    # Mock extrinsic with matching hash
    mock_extrinsic = mocker.Mock()
    mock_extrinsic.extrinsic_hash.hex.return_value = MOCK_EXTRINSIC_HASH
    mock_extrinsic.value = {"call": {}}

    mocker.patch.object(
        subtensor.substrate,
        "get_extrinsics",
        new=AsyncMock(return_value=[mock_extrinsic]),
    )

    mock_receipt = mocker.Mock()
    mock_async_receipt = mocker.patch.object(
        mev_shield,
        "AsyncExtrinsicReceipt",
        return_value=mock_receipt,
    )

    # Act
    result = await mev_shield.wait_for_extrinsic_by_hash(
        subtensor=subtensor,
        extrinsic_hash=extrinsic_hash,
        shield_id=shield_id,
        submit_block_hash=submit_block_hash,
        timeout_blocks=3,
    )

    # Assert
    assert result is mock_receipt
    mock_async_receipt.assert_called_once_with(
        substrate=subtensor.substrate,
        block_hash=MOCK_BLOCK_HASH_101,
        block_number=MOCK_STARTING_BLOCK + 1,
        extrinsic_idx=0,
    )


@pytest.mark.asyncio
async def test_wait_for_extrinsic_by_hash_decryption_failure(subtensor, mocker):
    """
    Test description: Verifies markDecryptionFailed event detection.

    Given: An extrinsic hash and shield ID
    When: wait_for_extrinsic_by_hash finds a markDecryptionFailed extrinsic
    Then: An AsyncExtrinsicReceipt is returned for the failure extrinsic
    """
    # Arrange
    extrinsic_hash = MOCK_EXTRINSIC_HASH_HEX
    shield_id = "shield_002"
    submit_block_hash = MOCK_SUBMIT_BLOCK_HASH

    mocker.patch.object(
        subtensor.substrate,
        "get_block_number",
        new=AsyncMock(return_value=MOCK_STARTING_BLOCK),
    )
    mocker.patch.object(subtensor, "wait_for_block", new=AsyncMock())
    mocker.patch.object(
        subtensor.substrate,
        "get_block_hash",
        new=AsyncMock(return_value=MOCK_BLOCK_HASH_101),
    )

    # Mock extrinsic with markDecryptionFailed call
    mock_extrinsic = mocker.Mock()
    mock_extrinsic.extrinsic_hash.hex.return_value = "different_hash"
    mock_extrinsic.value = {
        "call": {
            "call_module": "MevShield",
            "call_function": "mark_decryption_failed",
            "call_args": [{"name": "id", "value": shield_id}],
        }
    }

    mocker.patch.object(
        subtensor.substrate,
        "get_extrinsics",
        new=AsyncMock(return_value=[mock_extrinsic]),
    )

    mock_receipt = mocker.Mock()
    mock_async_receipt = mocker.patch.object(
        mev_shield,
        "AsyncExtrinsicReceipt",
        return_value=mock_receipt,
    )

    # Act
    result = await mev_shield.wait_for_extrinsic_by_hash(
        subtensor=subtensor,
        extrinsic_hash=extrinsic_hash,
        shield_id=shield_id,
        submit_block_hash=submit_block_hash,
        timeout_blocks=3,
    )

    # Assert
    assert result is mock_receipt
    mock_async_receipt.assert_called_once_with(
        substrate=subtensor.substrate,
        block_hash=MOCK_BLOCK_HASH_101,
        block_number=MOCK_STARTING_BLOCK + 1,
        extrinsic_idx=0,
    )


@pytest.mark.asyncio
async def test_wait_for_extrinsic_by_hash_timeout(subtensor, mocker):
    """
    Test description: Verifies behavior when extrinsic is not found within timeout_blocks.

    Given: An extrinsic hash and shield ID
    When: wait_for_extrinsic_by_hash polls blocks but never finds the extrinsic
    Then: None is returned after timeout_blocks iterations
    """
    # Arrange
    extrinsic_hash = MOCK_EXTRINSIC_HASH_HEX
    shield_id = "shield_003"
    submit_block_hash = MOCK_SUBMIT_BLOCK_HASH
    timeout_blocks = 2

    mocker.patch.object(
        subtensor.substrate,
        "get_block_number",
        new=AsyncMock(return_value=MOCK_STARTING_BLOCK),
    )
    mocker.patch.object(subtensor, "wait_for_block", new=AsyncMock())

    block_hashes = ["0xblock_101", "0xblock_102"]
    mocker.patch.object(
        subtensor.substrate,
        "get_block_hash",
        new=AsyncMock(side_effect=block_hashes),
    )

    # Mock extrinsics that don't match
    mock_extrinsic = mocker.Mock()
    mock_extrinsic.extrinsic_hash.hex.return_value = "non_matching_hash"
    mock_extrinsic.value = {"call": {}}

    mocker.patch.object(
        subtensor.substrate,
        "get_extrinsics",
        new=AsyncMock(return_value=[mock_extrinsic]),
    )

    # Act
    result = await mev_shield.wait_for_extrinsic_by_hash(
        subtensor=subtensor,
        extrinsic_hash=extrinsic_hash,
        shield_id=shield_id,
        submit_block_hash=submit_block_hash,
        timeout_blocks=timeout_blocks,
    )

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_submit_encrypted_extrinsic_invalid_signer(
    subtensor, fake_wallet, mocker
):
    """
    Test description: Verifies AttributeError is raised for invalid sign_with parameter.

    Given: A valid wallet and call
    When: submit_encrypted_extrinsic is called with sign_with not in ("coldkey", "hotkey")
    Then: AttributeError is raised with appropriate message
    """
    # Arrange
    mock_call = mocker.Mock()

    # Act & Assert
    with pytest.raises(AttributeError) as exc_info:
        await mev_shield.submit_encrypted_extrinsic(
            subtensor=subtensor,
            wallet=fake_wallet,
            call=mock_call,
            sign_with="invalid_key",
            raise_error=True,
        )

    assert str(exc_info.value) == "'sign_with' must be either 'coldkey' or 'hotkey', not 'invalid_key'"
