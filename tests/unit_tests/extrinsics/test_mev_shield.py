from bittensor_wallet import Wallet
from scalecodec.types import GenericCall
from async_substrate_interface import ExtrinsicReceipt

from bittensor.core.extrinsics import mev_shield
from bittensor.core.types import ExtrinsicResponse


def test_wait_for_extrinsic_by_hash_success(subtensor, mocker):
    """Verify that wait_for_extrinsic_by_hash finds the extrinsic by hash."""
    # Preps
    extrinsic_hash = "0x1234567890abcdef"
    shield_id = "shield_id_123"
    submit_block_hash = "0xblockhash"
    starting_block = 100
    current_block = 101

    mocked_get_block_number = mocker.patch.object(
        subtensor.substrate, "get_block_number", return_value=starting_block
    )
    mocked_wait_for_block = mocker.patch.object(subtensor, "wait_for_block")
    mocked_get_block_hash = mocker.patch.object(
        subtensor.substrate, "get_block_hash", return_value="0xblockhash101"
    )

    mock_extrinsic = mocker.MagicMock()
    mock_extrinsic.extrinsic_hash.hex.return_value = "1234567890abcdef"
    mocked_get_extrinsics = mocker.patch.object(
        subtensor.substrate,
        "get_extrinsics",
        return_value=[mock_extrinsic],
    )

    mock_receipt = mocker.MagicMock(spec=ExtrinsicReceipt)
    mocked_extrinsic_receipt = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.ExtrinsicReceipt",
        return_value=mock_receipt,
    )

    # Call
    result = mev_shield.wait_for_extrinsic_by_hash(
        subtensor=subtensor,
        extrinsic_hash=extrinsic_hash,
        shield_id=shield_id,
        submit_block_hash=submit_block_hash,
        timeout_blocks=3,
    )

    # Asserts
    mocked_get_block_number.assert_called_once_with(submit_block_hash)
    mocked_wait_for_block.assert_called_once()
    mocked_get_block_hash.assert_called_once_with(current_block)
    mocked_get_extrinsics.assert_called_once_with("0xblockhash101")
    mocked_extrinsic_receipt.assert_called_once_with(
        substrate=subtensor.substrate,
        block_hash="0xblockhash101",
        block_number=current_block,
        extrinsic_idx=0,
    )
    assert result == mock_receipt


def test_wait_for_extrinsic_by_hash_decryption_failed(subtensor, mocker):
    """Verify that wait_for_extrinsic_by_hash finds mark_decryption_failed extrinsic."""
    # Preps
    extrinsic_hash = "0x1234567890abcdef"
    shield_id = "shield_id_123"
    submit_block_hash = "0xblockhash"
    starting_block = 100
    current_block = 101

    mocked_get_block_number = mocker.patch.object(
        subtensor.substrate, "get_block_number", return_value=starting_block
    )
    mocked_wait_for_block = mocker.patch.object(subtensor, "wait_for_block")
    mocked_get_block_hash = mocker.patch.object(
        subtensor.substrate, "get_block_hash", return_value="0xblockhash101"
    )

    mock_extrinsic = mocker.MagicMock()
    mock_extrinsic.value = {
        "call": {
            "call_module": "MevShield",
            "call_function": "mark_decryption_failed",
            "call_args": [{"name": "id", "value": shield_id}],
        }
    }
    mocked_get_extrinsics = mocker.patch.object(
        subtensor.substrate,
        "get_extrinsics",
        return_value=[mock_extrinsic],
    )

    mock_receipt = mocker.MagicMock(spec=ExtrinsicReceipt)
    mocked_extrinsic_receipt = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.ExtrinsicReceipt",
        return_value=mock_receipt,
    )

    # Call
    result = mev_shield.wait_for_extrinsic_by_hash(
        subtensor=subtensor,
        extrinsic_hash=extrinsic_hash,
        shield_id=shield_id,
        submit_block_hash=submit_block_hash,
        timeout_blocks=3,
    )

    # Asserts
    mocked_get_block_number.assert_called_once_with(submit_block_hash)
    mocked_wait_for_block.assert_called_once()
    mocked_get_block_hash.assert_called_once_with(current_block)
    mocked_get_extrinsics.assert_called_once_with("0xblockhash101")
    mocked_extrinsic_receipt.assert_called_once_with(
        substrate=subtensor.substrate,
        block_hash="0xblockhash101",
        block_number=current_block,
        extrinsic_idx=0,
    )
    assert result == mock_receipt


def test_wait_for_extrinsic_by_hash_timeout(subtensor, mocker):
    """Verify that wait_for_extrinsic_by_hash returns None on timeout."""
    # Preps
    extrinsic_hash = "0x1234567890abcdef"
    shield_id = "shield_id_123"
    submit_block_hash = "0xblockhash"
    starting_block = 100

    mocked_get_block_number = mocker.patch.object(
        subtensor.substrate, "get_block_number", return_value=starting_block
    )
    mocked_wait_for_block = mocker.patch.object(subtensor, "wait_for_block")
    mocked_get_block_hash = mocker.patch.object(
        subtensor.substrate, "get_block_hash", return_value="0xblockhash101"
    )
    mocked_get_extrinsics = mocker.patch.object(
        subtensor.substrate,
        "get_extrinsics",
        return_value=[],
    )

    # Call
    result = mev_shield.wait_for_extrinsic_by_hash(
        subtensor=subtensor,
        extrinsic_hash=extrinsic_hash,
        shield_id=shield_id,
        submit_block_hash=submit_block_hash,
        timeout_blocks=3,
    )

    # Asserts
    mocked_get_block_number.assert_called_once_with(submit_block_hash)
    assert mocked_wait_for_block.call_count == 3
    assert mocked_get_block_hash.call_count == 3
    assert mocked_get_extrinsics.call_count == 3
    assert result is None


def test_submit_encrypted_extrinsic_success_with_revealed_execution(
    subtensor, fake_wallet, mocker
):
    """Verify that submit_encrypted_extrinsic works with wait_for_revealed_execution."""
    # Preps
    call = mocker.MagicMock(spec=GenericCall)
    fake_wallet.coldkey.ss58_address = (
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    )

    ml_kem_768_public_key = b"fake_ml_kem_key" * 74  # 1184 bytes
    mev_commitment = "0xcommitment"
    mev_ciphertext = b"fake_ciphertext"
    payload_core = b"fake_payload"
    signed_extrinsic_hash_hex = "abcdef123456"
    signed_extrinsic_hash = f"0x{signed_extrinsic_hash_hex}"
    current_nonce = 5
    next_nonce = 6
    shield_id = "shield_id_123"
    block_hash = "0xblockhash"

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_get_next_key = mocker.patch.object(
        subtensor, "get_mev_shield_next_key", return_value=ml_kem_768_public_key
    )
    mocked_get_account_next_index = mocker.patch.object(
        subtensor.substrate,
        "get_account_next_index",
        return_value=current_nonce,
    )
    mock_signed_extrinsic = mocker.MagicMock()
    mock_signed_extrinsic.extrinsic_hash.hex.return_value = signed_extrinsic_hash_hex
    mocked_create_signed_extrinsic = mocker.patch.object(
        subtensor.substrate,
        "create_signed_extrinsic",
        return_value=mock_signed_extrinsic,
    )
    mocked_get_mev_commitment = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.get_mev_commitment_and_ciphertext",
        return_value=(mev_commitment, mev_ciphertext, payload_core),
    )
    mocked_mev_shield = mocker.patch("bittensor.core.extrinsics.mev_shield.MevShield")
    mock_mev_shield_instance = mocker.MagicMock()
    mock_extrinsic_call = mocker.MagicMock()
    mock_mev_shield_instance.submit_encrypted.return_value = mock_extrinsic_call
    mocked_mev_shield.return_value = mock_mev_shield_instance

    mock_response = mocker.MagicMock(spec=ExtrinsicResponse)
    mock_response.success = True
    mock_response.extrinsic_receipt = mocker.MagicMock()
    mock_response.extrinsic_receipt.block_hash = block_hash
    mock_response.extrinsic_receipt.triggered_events = [
        {
            "module_id": "mevShield",
            "event_id": "EncryptedSubmitted",
            "attributes": {"id": shield_id},
        }
    ]

    mock_mev_extrinsic = mocker.MagicMock(spec=ExtrinsicReceipt)
    mock_mev_extrinsic.is_success = True
    mocked_wait_for_extrinsic = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.wait_for_extrinsic_by_hash",
        return_value=mock_mev_extrinsic,
    )
    mocked_get_event_data = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.get_event_data_by_event_name",
        return_value={
            "module_id": "mevShield",
            "event_id": "EncryptedSubmitted",
            "attributes": {"id": shield_id},
        },
    )

    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=mock_response,
    )

    # Call
    result = mev_shield.submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=call,
        sign_with="coldkey",
        wait_for_revealed_execution=True,
        blocks_for_revealed_execution=3,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(fake_wallet, False, "coldkey")
    mocked_get_next_key.assert_called_once()
    mocked_get_account_next_index.assert_called_once_with(
        account_address=fake_wallet.coldkey.ss58_address
    )
    mocked_create_signed_extrinsic.assert_called_once_with(
        call=call,
        keypair=fake_wallet.coldkey,
        nonce=next_nonce,
        era="00",
    )
    mocked_get_mev_commitment.assert_called_once_with(
        signed_ext=mock_signed_extrinsic,
        ml_kem_768_public_key=ml_kem_768_public_key,
    )
    mocked_mev_shield.assert_called_once_with(subtensor)
    mock_mev_shield_instance.submit_encrypted.assert_called_once_with(
        commitment=mev_commitment,
        ciphertext=mev_ciphertext,
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        wallet=fake_wallet,
        sign_with="coldkey",
        call=mock_extrinsic_call,
        nonce=current_nonce,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    mocked_get_event_data.assert_called_once_with(
        events=mock_response.extrinsic_receipt.triggered_events,
        event_name="mevShield.EncryptedSubmitted",
    )
    mocked_wait_for_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        extrinsic_hash=signed_extrinsic_hash,
        shield_id=shield_id,
        submit_block_hash=block_hash,
        timeout_blocks=3,
    )
    assert result == mock_response
    assert result.mev_extrinsic == mock_mev_extrinsic
    assert result.data["commitment"] == mev_commitment
    assert result.data["ciphertext"] == mev_ciphertext
    assert result.data["ml_kem_768_public_key"] == ml_kem_768_public_key
    assert result.data["payload_core"] == payload_core
    assert result.data["signed_extrinsic_hash"] == signed_extrinsic_hash


def test_submit_encrypted_extrinsic_success_without_revealed_execution(
    subtensor, fake_wallet, mocker
):
    """Verify that submit_encrypted_extrinsic works without wait_for_revealed_execution."""
    # Preps
    call = mocker.MagicMock(spec=GenericCall)
    fake_wallet.coldkey.ss58_address = (
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    )

    ml_kem_768_public_key = b"fake_ml_kem_key" * 74
    mev_commitment = "0xcommitment"
    mev_ciphertext = b"fake_ciphertext"
    payload_core = b"fake_payload"
    signed_extrinsic_hash_hex = "abcdef123456"
    signed_extrinsic_hash = f"0x{signed_extrinsic_hash_hex}"
    current_nonce = 5
    next_nonce = 6

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_get_next_key = mocker.patch.object(
        subtensor, "get_mev_shield_next_key", return_value=ml_kem_768_public_key
    )
    mocked_get_account_next_index = mocker.patch.object(
        subtensor.substrate,
        "get_account_next_index",
        return_value=current_nonce,
    )
    mock_signed_extrinsic = mocker.MagicMock()
    mock_signed_extrinsic.extrinsic_hash.hex.return_value = signed_extrinsic_hash_hex
    mocked_create_signed_extrinsic = mocker.patch.object(
        subtensor.substrate,
        "create_signed_extrinsic",
        return_value=mock_signed_extrinsic,
    )
    mocked_get_mev_commitment = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.get_mev_commitment_and_ciphertext",
        return_value=(mev_commitment, mev_ciphertext, payload_core),
    )
    mocked_mev_shield = mocker.patch("bittensor.core.extrinsics.mev_shield.MevShield")
    mock_mev_shield_instance = mocker.MagicMock()
    mock_extrinsic_call = mocker.MagicMock()
    mock_mev_shield_instance.submit_encrypted.return_value = mock_extrinsic_call
    mocked_mev_shield.return_value = mock_mev_shield_instance

    mock_response = mocker.MagicMock(spec=ExtrinsicResponse)
    mock_response.success = True

    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=mock_response,
    )

    # Call
    result = mev_shield.submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=call,
        sign_with="coldkey",
        wait_for_revealed_execution=False,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(fake_wallet, False, "coldkey")
    mocked_get_next_key.assert_called_once()
    mocked_get_account_next_index.assert_called_once_with(
        account_address=fake_wallet.coldkey.ss58_address
    )
    mocked_create_signed_extrinsic.assert_called_once_with(
        call=call,
        keypair=fake_wallet.coldkey,
        nonce=next_nonce,
        era="00",
    )
    mocked_get_mev_commitment.assert_called_once_with(
        signed_ext=mock_signed_extrinsic,
        ml_kem_768_public_key=ml_kem_768_public_key,
    )
    mocked_mev_shield.assert_called_once_with(subtensor)
    mock_mev_shield_instance.submit_encrypted.assert_called_once_with(
        commitment=mev_commitment,
        ciphertext=mev_ciphertext,
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        wallet=fake_wallet,
        sign_with="coldkey",
        call=mock_extrinsic_call,
        nonce=current_nonce,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    assert result == mock_response
    assert result.data["commitment"] == mev_commitment
    assert result.data["ciphertext"] == mev_ciphertext
    assert result.data["ml_kem_768_public_key"] == ml_kem_768_public_key
    assert result.data["payload_core"] == payload_core
    assert result.data["signed_extrinsic_hash"] == signed_extrinsic_hash


def test_submit_encrypted_extrinsic_invalid_sign_with(subtensor, fake_wallet, mocker):
    """Verify that submit_encrypted_extrinsic raises error for invalid sign_with."""
    # Preps
    call = mocker.MagicMock(spec=GenericCall)

    # Call
    result = mev_shield.submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=call,
        sign_with="invalid_key",
    )

    # Asserts
    assert result.success is False
    assert isinstance(result.error, AttributeError)
    assert "sign_with" in str(result.error)
    assert "invalid_key" in str(result.error)


def test_submit_encrypted_extrinsic_revealed_execution_without_inclusion(
    subtensor, fake_wallet, mocker
):
    """Verify that submit_encrypted_extrinsic returns error when wait_for_revealed_execution is True but wait_for_inclusion is False."""
    # Preps
    call = mocker.MagicMock(spec=GenericCall)

    # Call
    result = mev_shield.submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=call,
        wait_for_revealed_execution=True,
        wait_for_inclusion=False,
    )

    # Asserts
    assert result.success is False
    assert isinstance(result.error, ValueError)
    assert "wait_for_inclusion" in str(result.error)


def test_submit_encrypted_extrinsic_unlock_failure(subtensor, fake_wallet, mocker):
    """Verify that submit_encrypted_extrinsic returns error when wallet unlock fails."""
    # Preps
    call = mocker.MagicMock(spec=GenericCall)

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=False, message="Unlock failed"),
    )

    # Call
    result = mev_shield.submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=call,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(fake_wallet, False, "coldkey")
    assert result.success is False
    assert result.message == "Unlock failed"


def test_submit_encrypted_extrinsic_next_key_not_available(
    subtensor, fake_wallet, mocker
):
    """Verify that submit_encrypted_extrinsic returns error when NextKey is not available."""
    # Preps
    call = mocker.MagicMock(spec=GenericCall)

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_get_next_key = mocker.patch.object(
        subtensor, "get_mev_shield_next_key", return_value=None
    )

    # Call
    result = mev_shield.submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=call,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(fake_wallet, False, "coldkey")
    mocked_get_next_key.assert_called_once()
    assert result.success is False
    assert isinstance(result.error, ValueError)
    assert "NextKey not available" in str(result.error)


def test_submit_encrypted_extrinsic_encrypted_submitted_event_not_found(
    subtensor, fake_wallet, mocker
):
    """Verify that submit_encrypted_extrinsic returns error when EncryptedSubmitted event is not found."""
    # Preps
    call = mocker.MagicMock(spec=GenericCall)
    fake_wallet.coldkey.ss58_address = (
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    )

    ml_kem_768_public_key = b"fake_ml_kem_key" * 74
    mev_commitment = "0xcommitment"
    mev_ciphertext = b"fake_ciphertext"
    payload_core = b"fake_payload"
    current_nonce = 5
    next_nonce = 6

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_get_next_key = mocker.patch.object(
        subtensor, "get_mev_shield_next_key", return_value=ml_kem_768_public_key
    )
    mocked_get_account_next_index = mocker.patch.object(
        subtensor.substrate,
        "get_account_next_index",
        return_value=current_nonce,
    )
    mock_signed_extrinsic = mocker.MagicMock()
    mock_signed_extrinsic.extrinsic_hash.hex.return_value = "abcdef123456"
    mocked_create_signed_extrinsic = mocker.patch.object(
        subtensor.substrate,
        "create_signed_extrinsic",
        return_value=mock_signed_extrinsic,
    )
    mocked_get_mev_commitment = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.get_mev_commitment_and_ciphertext",
        return_value=(mev_commitment, mev_ciphertext, payload_core),
    )
    mocked_mev_shield = mocker.patch("bittensor.core.extrinsics.mev_shield.MevShield")
    mock_mev_shield_instance = mocker.MagicMock()
    mock_extrinsic_call = mocker.MagicMock()
    mock_mev_shield_instance.submit_encrypted.return_value = mock_extrinsic_call
    mocked_mev_shield.return_value = mock_mev_shield_instance

    mock_response = mocker.MagicMock(spec=ExtrinsicResponse)
    mock_response.success = True
    mock_response.extrinsic_receipt = mocker.MagicMock()
    mock_response.extrinsic_receipt.triggered_events = []

    mocked_get_event_data = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.get_event_data_by_event_name",
        return_value=None,
    )

    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=mock_response,
    )

    # Call
    result = mev_shield.submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=call,
        wait_for_revealed_execution=True,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(fake_wallet, False, "coldkey")
    mocked_get_next_key.assert_called_once()
    mocked_get_account_next_index.assert_called_once_with(
        account_address=fake_wallet.coldkey.ss58_address
    )
    mocked_create_signed_extrinsic.assert_called_once_with(
        call=call,
        keypair=fake_wallet.coldkey,
        nonce=next_nonce,
        era="00",
    )
    mocked_get_mev_commitment.assert_called_once_with(
        signed_ext=mock_signed_extrinsic,
        ml_kem_768_public_key=ml_kem_768_public_key,
    )
    mocked_mev_shield.assert_called_once_with(subtensor)
    mock_mev_shield_instance.submit_encrypted.assert_called_once_with(
        commitment=mev_commitment,
        ciphertext=mev_ciphertext,
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        wallet=fake_wallet,
        sign_with="coldkey",
        call=mock_extrinsic_call,
        nonce=current_nonce,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    mocked_get_event_data.assert_called_once_with(
        events=mock_response.extrinsic_receipt.triggered_events,
        event_name="mevShield.EncryptedSubmitted",
    )
    assert result.success is False
    assert isinstance(result.error, RuntimeError)
    assert "EncryptedSubmitted event not found" in str(result.error)


def test_submit_encrypted_extrinsic_failed_to_find_outcome(
    subtensor, fake_wallet, mocker
):
    """Verify that submit_encrypted_extrinsic returns error when outcome is not found."""
    # Preps
    call = mocker.MagicMock(spec=GenericCall)
    fake_wallet.coldkey.ss58_address = (
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    )

    ml_kem_768_public_key = b"fake_ml_kem_key" * 74
    mev_commitment = "0xcommitment"
    mev_ciphertext = b"fake_ciphertext"
    payload_core = b"fake_payload"
    signed_extrinsic_hash_hex = "abcdef123456"
    signed_extrinsic_hash = f"0x{signed_extrinsic_hash_hex}"
    current_nonce = 5
    next_nonce = 6
    shield_id = "shield_id_123"
    block_hash = "0xblockhash"

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_get_next_key = mocker.patch.object(
        subtensor, "get_mev_shield_next_key", return_value=ml_kem_768_public_key
    )
    mocked_get_account_next_index = mocker.patch.object(
        subtensor.substrate,
        "get_account_next_index",
        return_value=current_nonce,
    )
    mock_signed_extrinsic = mocker.MagicMock()
    mock_signed_extrinsic.extrinsic_hash.hex.return_value = signed_extrinsic_hash_hex
    mocked_create_signed_extrinsic = mocker.patch.object(
        subtensor.substrate,
        "create_signed_extrinsic",
        return_value=mock_signed_extrinsic,
    )
    mocked_get_mev_commitment = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.get_mev_commitment_and_ciphertext",
        return_value=(mev_commitment, mev_ciphertext, payload_core),
    )
    mocked_mev_shield = mocker.patch("bittensor.core.extrinsics.mev_shield.MevShield")
    mock_mev_shield_instance = mocker.MagicMock()
    mock_extrinsic_call = mocker.MagicMock()
    mock_mev_shield_instance.submit_encrypted.return_value = mock_extrinsic_call
    mocked_mev_shield.return_value = mock_mev_shield_instance

    mock_response = mocker.MagicMock(spec=ExtrinsicResponse)
    mock_response.success = True
    mock_response.extrinsic_receipt = mocker.MagicMock()
    mock_response.extrinsic_receipt.block_hash = block_hash
    mock_response.extrinsic_receipt.triggered_events = [
        {
            "module_id": "mevShield",
            "event_id": "EncryptedSubmitted",
            "attributes": {"id": shield_id},
        }
    ]

    mocked_get_event_data = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.get_event_data_by_event_name",
        return_value={
            "module_id": "mevShield",
            "event_id": "EncryptedSubmitted",
            "attributes": {"id": shield_id},
        },
    )
    mocked_wait_for_extrinsic = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.wait_for_extrinsic_by_hash",
        return_value=None,
    )

    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=mock_response,
    )

    # Call
    result = mev_shield.submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=call,
        wait_for_revealed_execution=True,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(fake_wallet, False, "coldkey")
    mocked_get_next_key.assert_called_once()
    mocked_get_account_next_index.assert_called_once_with(
        account_address=fake_wallet.coldkey.ss58_address
    )
    mocked_create_signed_extrinsic.assert_called_once_with(
        call=call,
        keypair=fake_wallet.coldkey,
        nonce=next_nonce,
        era="00",
    )
    mocked_get_mev_commitment.assert_called_once_with(
        signed_ext=mock_signed_extrinsic,
        ml_kem_768_public_key=ml_kem_768_public_key,
    )
    mocked_mev_shield.assert_called_once_with(subtensor)
    mock_mev_shield_instance.submit_encrypted.assert_called_once_with(
        commitment=mev_commitment,
        ciphertext=mev_ciphertext,
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        wallet=fake_wallet,
        sign_with="coldkey",
        call=mock_extrinsic_call,
        nonce=current_nonce,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    mocked_get_event_data.assert_called_once_with(
        events=mock_response.extrinsic_receipt.triggered_events,
        event_name="mevShield.EncryptedSubmitted",
    )
    mocked_wait_for_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        extrinsic_hash=signed_extrinsic_hash,
        shield_id=shield_id,
        submit_block_hash=block_hash,
        timeout_blocks=3,
    )
    assert result.success is False
    assert isinstance(result.error, RuntimeError)
    assert "Failed to find outcome" in str(result.error)


def test_submit_encrypted_extrinsic_execution_failure(subtensor, fake_wallet, mocker):
    """Verify that submit_encrypted_extrinsic handles execution failure."""
    # Preps
    call = mocker.MagicMock(spec=GenericCall)
    fake_wallet.coldkey.ss58_address = (
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    )

    ml_kem_768_public_key = b"fake_ml_kem_key" * 74
    mev_commitment = "0xcommitment"
    mev_ciphertext = b"fake_ciphertext"
    payload_core = b"fake_payload"
    signed_extrinsic_hash_hex = "abcdef123456"
    signed_extrinsic_hash = f"0x{signed_extrinsic_hash_hex}"
    current_nonce = 5
    next_nonce = 6
    shield_id = "shield_id_123"
    block_hash = "0xblockhash"
    error_message = "Execution failed"
    formatted_error = "Formatted error: Execution failed"

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_get_next_key = mocker.patch.object(
        subtensor, "get_mev_shield_next_key", return_value=ml_kem_768_public_key
    )
    mocked_get_account_next_index = mocker.patch.object(
        subtensor.substrate,
        "get_account_next_index",
        return_value=current_nonce,
    )
    mock_signed_extrinsic = mocker.MagicMock()
    mock_signed_extrinsic.extrinsic_hash.hex.return_value = signed_extrinsic_hash_hex
    mocked_create_signed_extrinsic = mocker.patch.object(
        subtensor.substrate,
        "create_signed_extrinsic",
        return_value=mock_signed_extrinsic,
    )
    mocked_get_mev_commitment = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.get_mev_commitment_and_ciphertext",
        return_value=(mev_commitment, mev_ciphertext, payload_core),
    )
    mocked_mev_shield = mocker.patch("bittensor.core.extrinsics.mev_shield.MevShield")
    mock_mev_shield_instance = mocker.MagicMock()
    mock_extrinsic_call = mocker.MagicMock()
    mock_mev_shield_instance.submit_encrypted.return_value = mock_extrinsic_call
    mocked_mev_shield.return_value = mock_mev_shield_instance

    mock_response = mocker.MagicMock(spec=ExtrinsicResponse)
    mock_response.success = True
    mock_response.extrinsic_receipt = mocker.MagicMock()
    mock_response.extrinsic_receipt.block_hash = block_hash
    mock_response.extrinsic_receipt.triggered_events = [
        {
            "module_id": "mevShield",
            "event_id": "EncryptedSubmitted",
            "attributes": {"id": shield_id},
        }
    ]

    mock_mev_extrinsic = mocker.MagicMock(spec=ExtrinsicReceipt)
    mock_mev_extrinsic.is_success = False
    mock_mev_extrinsic.error_message = error_message
    mocked_wait_for_extrinsic = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.wait_for_extrinsic_by_hash",
        return_value=mock_mev_extrinsic,
    )
    mocked_get_event_data = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.get_event_data_by_event_name",
        return_value={
            "module_id": "mevShield",
            "event_id": "EncryptedSubmitted",
            "attributes": {"id": shield_id},
        },
    )
    mocked_format_error_message = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.format_error_message",
        return_value=formatted_error,
    )

    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=mock_response,
    )

    # Call
    result = mev_shield.submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=call,
        wait_for_revealed_execution=True,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(fake_wallet, False, "coldkey")
    mocked_get_next_key.assert_called_once()
    mocked_get_account_next_index.assert_called_once_with(
        account_address=fake_wallet.coldkey.ss58_address
    )
    mocked_create_signed_extrinsic.assert_called_once_with(
        call=call,
        keypair=fake_wallet.coldkey,
        nonce=next_nonce,
        era="00",
    )
    mocked_get_mev_commitment.assert_called_once_with(
        signed_ext=mock_signed_extrinsic,
        ml_kem_768_public_key=ml_kem_768_public_key,
    )
    mocked_mev_shield.assert_called_once_with(subtensor)
    mock_mev_shield_instance.submit_encrypted.assert_called_once_with(
        commitment=mev_commitment,
        ciphertext=mev_ciphertext,
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        wallet=fake_wallet,
        sign_with="coldkey",
        call=mock_extrinsic_call,
        nonce=current_nonce,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    mocked_get_event_data.assert_called_once_with(
        events=mock_response.extrinsic_receipt.triggered_events,
        event_name="mevShield.EncryptedSubmitted",
    )
    mocked_wait_for_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        extrinsic_hash=signed_extrinsic_hash,
        shield_id=shield_id,
        submit_block_hash=block_hash,
        timeout_blocks=3,
    )
    mocked_format_error_message.assert_called_once_with(error_message)
    assert result.success is False
    assert isinstance(result.error, RuntimeError)
    assert result.message == formatted_error


def test_submit_encrypted_extrinsic_sign_and_send_failure(
    subtensor, fake_wallet, mocker
):
    """Verify that submit_encrypted_extrinsic handles sign_and_send_extrinsic failure."""
    # Preps
    call = mocker.MagicMock(spec=GenericCall)
    fake_wallet.coldkey.ss58_address = (
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    )

    ml_kem_768_public_key = b"fake_ml_kem_key" * 74
    mev_commitment = "0xcommitment"
    mev_ciphertext = b"fake_ciphertext"
    payload_core = b"fake_payload"
    current_nonce = 5
    next_nonce = 6

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_get_next_key = mocker.patch.object(
        subtensor, "get_mev_shield_next_key", return_value=ml_kem_768_public_key
    )
    mocked_get_account_next_index = mocker.patch.object(
        subtensor.substrate,
        "get_account_next_index",
        return_value=current_nonce,
    )
    mock_signed_extrinsic = mocker.MagicMock()
    mock_signed_extrinsic.extrinsic_hash.hex.return_value = "abcdef123456"
    mocked_create_signed_extrinsic = mocker.patch.object(
        subtensor.substrate,
        "create_signed_extrinsic",
        return_value=mock_signed_extrinsic,
    )
    mocked_get_mev_commitment = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.get_mev_commitment_and_ciphertext",
        return_value=(mev_commitment, mev_ciphertext, payload_core),
    )
    mocked_mev_shield = mocker.patch("bittensor.core.extrinsics.mev_shield.MevShield")
    mock_mev_shield_instance = mocker.MagicMock()
    mock_extrinsic_call = mocker.MagicMock()
    mock_mev_shield_instance.submit_encrypted.return_value = mock_extrinsic_call
    mocked_mev_shield.return_value = mock_mev_shield_instance

    mock_response = mocker.MagicMock(spec=ExtrinsicResponse)
    mock_response.success = False
    mock_response.message = "Transaction failed"

    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=mock_response,
    )

    # Call
    result = mev_shield.submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=call,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(fake_wallet, False, "coldkey")
    mocked_get_next_key.assert_called_once()
    mocked_get_account_next_index.assert_called_once_with(
        account_address=fake_wallet.coldkey.ss58_address
    )
    mocked_create_signed_extrinsic.assert_called_once_with(
        call=call,
        keypair=fake_wallet.coldkey,
        nonce=next_nonce,
        era="00",
    )
    mocked_get_mev_commitment.assert_called_once_with(
        signed_ext=mock_signed_extrinsic,
        ml_kem_768_public_key=ml_kem_768_public_key,
    )
    mocked_mev_shield.assert_called_once_with(subtensor)
    mock_mev_shield_instance.submit_encrypted.assert_called_once_with(
        commitment=mev_commitment,
        ciphertext=mev_ciphertext,
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        wallet=fake_wallet,
        sign_with="coldkey",
        call=mock_extrinsic_call,
        nonce=current_nonce,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    assert result == mock_response
    assert result.success is False
    assert result.message == "Transaction failed"


def test_submit_encrypted_extrinsic_with_hotkey(subtensor, fake_wallet, mocker):
    """Verify that submit_encrypted_extrinsic works with hotkey signing."""
    # Preps
    call = mocker.MagicMock(spec=GenericCall)
    fake_wallet.hotkey.ss58_address = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"

    ml_kem_768_public_key = b"fake_ml_kem_key" * 74
    mev_commitment = "0xcommitment"
    mev_ciphertext = b"fake_ciphertext"
    payload_core = b"fake_payload"
    current_nonce = 5
    next_nonce = 6

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_get_next_key = mocker.patch.object(
        subtensor, "get_mev_shield_next_key", return_value=ml_kem_768_public_key
    )
    mocked_get_account_next_index = mocker.patch.object(
        subtensor.substrate,
        "get_account_next_index",
        return_value=current_nonce,
    )
    mock_signed_extrinsic = mocker.MagicMock()
    mock_signed_extrinsic.extrinsic_hash.hex.return_value = "abcdef123456"
    mocked_create_signed_extrinsic = mocker.patch.object(
        subtensor.substrate,
        "create_signed_extrinsic",
        return_value=mock_signed_extrinsic,
    )
    mocked_get_mev_commitment = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.get_mev_commitment_and_ciphertext",
        return_value=(mev_commitment, mev_ciphertext, payload_core),
    )
    mocked_mev_shield = mocker.patch("bittensor.core.extrinsics.mev_shield.MevShield")
    mock_mev_shield_instance = mocker.MagicMock()
    mock_extrinsic_call = mocker.MagicMock()
    mock_mev_shield_instance.submit_encrypted.return_value = mock_extrinsic_call
    mocked_mev_shield.return_value = mock_mev_shield_instance

    mock_response = mocker.MagicMock(spec=ExtrinsicResponse)
    mock_response.success = True

    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=mock_response,
    )

    # Call
    result = mev_shield.submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=call,
        sign_with="hotkey",
        wait_for_revealed_execution=False,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(fake_wallet, False, "hotkey")
    mocked_get_next_key.assert_called_once()
    mocked_get_account_next_index.assert_called_once_with(
        account_address=fake_wallet.hotkey.ss58_address
    )
    mocked_create_signed_extrinsic.assert_called_once_with(
        call=call,
        keypair=fake_wallet.hotkey,
        nonce=next_nonce,
        era="00",
    )
    mocked_get_mev_commitment.assert_called_once_with(
        signed_ext=mock_signed_extrinsic,
        ml_kem_768_public_key=ml_kem_768_public_key,
    )
    mocked_mev_shield.assert_called_once_with(subtensor)
    mock_mev_shield_instance.submit_encrypted.assert_called_once_with(
        commitment=mev_commitment,
        ciphertext=mev_ciphertext,
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        wallet=fake_wallet,
        sign_with="hotkey",
        call=mock_extrinsic_call,
        nonce=current_nonce,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    assert result == mock_response


def test_submit_encrypted_extrinsic_with_period(subtensor, fake_wallet, mocker):
    """Verify that submit_encrypted_extrinsic works with period parameter."""
    # Preps
    call = mocker.MagicMock(spec=GenericCall)
    fake_wallet.coldkey.ss58_address = (
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    )

    ml_kem_768_public_key = b"fake_ml_kem_key" * 74
    mev_commitment = "0xcommitment"
    mev_ciphertext = b"fake_ciphertext"
    payload_core = b"fake_payload"
    current_nonce = 5
    next_nonce = 6
    period = 64

    mocked_unlock_wallet = mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(success=True, message="Unlocked"),
    )
    mocked_get_next_key = mocker.patch.object(
        subtensor, "get_mev_shield_next_key", return_value=ml_kem_768_public_key
    )
    mocked_get_account_next_index = mocker.patch.object(
        subtensor.substrate,
        "get_account_next_index",
        return_value=current_nonce,
    )
    mock_signed_extrinsic = mocker.MagicMock()
    mock_signed_extrinsic.extrinsic_hash.hex.return_value = "abcdef123456"
    mocked_create_signed_extrinsic = mocker.patch.object(
        subtensor.substrate,
        "create_signed_extrinsic",
        return_value=mock_signed_extrinsic,
    )
    mocked_get_mev_commitment = mocker.patch(
        "bittensor.core.extrinsics.mev_shield.get_mev_commitment_and_ciphertext",
        return_value=(mev_commitment, mev_ciphertext, payload_core),
    )
    mocked_mev_shield = mocker.patch("bittensor.core.extrinsics.mev_shield.MevShield")
    mock_mev_shield_instance = mocker.MagicMock()
    mock_extrinsic_call = mocker.MagicMock()
    mock_mev_shield_instance.submit_encrypted.return_value = mock_extrinsic_call
    mocked_mev_shield.return_value = mock_mev_shield_instance

    mock_response = mocker.MagicMock(spec=ExtrinsicResponse)
    mock_response.success = True

    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=mock_response,
    )

    # Call
    result = mev_shield.submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=call,
        period=period,
        wait_for_revealed_execution=False,
    )

    # Asserts
    mocked_unlock_wallet.assert_called_once_with(fake_wallet, False, "coldkey")
    mocked_get_next_key.assert_called_once()
    mocked_get_account_next_index.assert_called_once_with(
        account_address=fake_wallet.coldkey.ss58_address
    )
    mocked_create_signed_extrinsic.assert_called_once_with(
        call=call,
        keypair=fake_wallet.coldkey,
        nonce=next_nonce,
        era={"period": period},
    )
    mocked_get_mev_commitment.assert_called_once_with(
        signed_ext=mock_signed_extrinsic,
        ml_kem_768_public_key=ml_kem_768_public_key,
    )
    mocked_mev_shield.assert_called_once_with(subtensor)
    mock_mev_shield_instance.submit_encrypted.assert_called_once_with(
        commitment=mev_commitment,
        ciphertext=mev_ciphertext,
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        wallet=fake_wallet,
        sign_with="coldkey",
        call=mock_extrinsic_call,
        nonce=current_nonce,
        period=period,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    assert result == mock_response
