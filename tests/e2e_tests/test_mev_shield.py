"""E2E tests for MEV Shield functionality."""

import pytest
import hashlib

from bittensor_drand import generate_mlkem768_keypair
from bittensor_wallet import Wallet

from bittensor.core.extrinsics.mev_shield import submit_encrypted_extrinsic
from bittensor.core.extrinsics.pallets import SubtensorModule
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils import (
    AdminUtils,
    NETUID,
    SUDO_SET_TEMPO,
    TestSubnet,
)


def add_balance_to_wallet_hk(
    subtensor, wallet: "Wallet", tao_amount: int
) -> ExtrinsicResponse:
    """Adds 100 TAO to Alice's HK balance."""
    return subtensor.extrinsics.transfer(
        wallet=wallet,
        destination_ss58=wallet.hotkeypub.ss58_address,
        amount=Balance.from_tao(tao_amount),
    )


def test_mev_shield_storage_queries(subtensor):
    """Tests querying MevShield storage items.

    Steps:
    1. Query CurrentKey (may be None)
    2. Query NextKey (may be None)
    3. Query Epoch (always >= 0)
    4. Query Submissions (all, returns dict)
    """

    # Query Epoch - should always be available
    epoch = subtensor.mev_shield.get_mev_shield_epoch()
    assert isinstance(epoch, int), f"Epoch should be int, got {type(epoch)}"
    assert epoch >= 0, f"Epoch should be non-negative, got {epoch}"

    # Query CurrentKey - may be None if no key announced yet
    current_key = subtensor.mev_shield.get_mev_shield_current_key()
    if current_key is not None:
        public_key_bytes, epoch_val = current_key
        assert isinstance(public_key_bytes, bytes), "Public key should be bytes"
        assert len(public_key_bytes) == 1184, (
            f"ML-KEM-768 public key should be 1184 bytes, got {len(public_key_bytes)}"
        )
        assert isinstance(epoch_val, int), "Epoch should be int"
        logging.debug(
            f"CurrentKey found: epoch={epoch_val}, key_size={len(public_key_bytes)}"
        )

    # Query NextKey - may be None if no key announced yet
    next_key = subtensor.mev_shield.get_mev_shield_next_key()
    if next_key is not None:
        public_key_bytes, epoch_val = next_key
        assert isinstance(public_key_bytes, bytes), "Public key should be bytes"
        assert len(public_key_bytes) == 1184, (
            f"ML-KEM-768 public key should be 1184 bytes, got {len(public_key_bytes)}"
        )
        assert isinstance(epoch_val, int), "Epoch should be int"
        logging.debug(
            f"NextKey found: epoch={epoch_val}, key_size={len(public_key_bytes)}"
        )

    # Query all submissions - should always return a dict
    all_submissions = subtensor.mev_shield.get_mev_shield_submission()
    assert isinstance(all_submissions, dict), (
        f"Should return dict of all submissions, got {type(all_submissions)}"
    )
    logging.debug(f"Found {len(all_submissions)} submission(s) in storage")


def test_mev_shield_announce_next_key(subtensor, alice_wallet):
    """Tests announcing next key as a validator.

    Steps:
    1. Generate ML-KEM-768 keypair using bittensor_drand
    2. Get current epoch
    3. Announce next key with next epoch
    4. Wait for next block
    5. Verify NextKey is set correctly
    6. Verify public_key size is 1184 bytes
    7. Verify epoch matches
    """
    # add founds to alice HK
    response = add_balance_to_wallet_hk(subtensor, alice_wallet, 100)
    assert response.success, response.message

    # Set tempo to 100 blocks for root network (netuid=0) to ensure stable epoch timing
    # MEV Shield epoch depends on root network tempo, so we need predictable epoch boundaries
    TEMPO_TO_SET = 100
    root_sn = TestSubnet(subtensor, netuid=0)
    tempo_response = root_sn.execute_one(
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, 0, TEMPO_TO_SET)
    )
    assert tempo_response.success, f"Failed to set tempo: {tempo_response.message}"

    # Wait for tempo transaction to be included in a block
    # This ensures the tempo change is processed before we submit the announce transaction
    current_block = subtensor.block
    subtensor.wait_for_block(current_block + 1)

    # Generate ML-KEM-768 keypair using bittensor_drand
    public_key_bytes = generate_mlkem768_keypair()
    assert len(public_key_bytes) == 1184, (
        f"Generated key should be 1184 bytes, got {len(public_key_bytes)}"
    )

    # Get current epoch and prepare next epoch
    # We use current_epoch + 1 because we're announcing the key for the NEXT epoch
    current_epoch = subtensor.mev_shield.get_mev_shield_epoch()
    next_epoch = current_epoch + 1

    logging.info(
        f"Announcing next key for epoch {next_epoch} (current epoch: {current_epoch})"
    )

    # Announce next key with explicit period to avoid transaction expiration
    # Use a longer period (256 blocks) to ensure transaction doesn't expire during epoch transitions
    response = announce_next_key_extrinsic(
        subtensor=subtensor,
        wallet=alice_wallet,
        public_key=public_key_bytes,
        epoch=next_epoch,
        period=256,  # Longer period to avoid "Transaction is outdated" errors
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert response.success, f"Failed to announce next key: {response.message}"

    # Verify that extrinsic was actually executed successfully (not just included)
    assert response.extrinsic_receipt is not None, (
        "Extrinsic receipt should be available"
    )
    assert response.extrinsic_receipt.is_success, (
        f"Extrinsic execution failed: {response.extrinsic_receipt.error_message}"
    )

    logging.info("Next key announced successfully, waiting for block...")

    # Wait for next block to ensure the announcement is processed
    current_block = subtensor.block
    subtensor.wait_for_block(current_block + 1)

    # Verify NextKey is set
    next_key_result = subtensor.mev_shield.get_mev_shield_next_key()
    assert next_key_result is not None, "NextKey should be set after announcement"

    pk_bytes, epoch = next_key_result
    assert pk_bytes == public_key_bytes, "Public key should match announced key"
    assert len(pk_bytes) == 1184, (
        f"Public key size should be 1184 bytes, got {len(pk_bytes)}"
    )
    assert epoch == next_epoch, (
        f"Epoch should match announced epoch {next_epoch}, got {epoch}"
    )

    logging.info(f"NextKey verified: epoch={epoch}, key_size={len(pk_bytes)}")


def test_mev_shield_submit_encrypted_full_flow(subtensor, bob_wallet, alice_wallet):
    """Tests submitting an encrypted extrinsic with full verification.

    Steps:
    1. Ensure NextKey exists (announce if needed)
    2. Create a simple transfer call
    3. Submit encrypted extrinsic
    4. Wait for inclusion
    5. Verify submission exists in storage
    6. Verify commitment matches computed value
    7. Verify author matches wallet address
    8. Verify submitted_in block number
    """

    # add founds to alice, bob HK
    for w in [alice_wallet, bob_wallet]:
        response = add_balance_to_wallet_hk(subtensor, w, 100)
        assert response.success, response.message

    # Ensure NextKey exists - if not, we need to announce it first using alice_wallet (validator)
    next_key_result = subtensor.mev_shield.get_mev_shield_next_key()
    if next_key_result is None:
        # Set tempo to 100 blocks for root network (netuid=0) to ensure stable epoch timing
        # MEV Shield epoch depends on root network tempo, so we need predictable epoch boundaries
        TEMPO_TO_SET = 100
        root_sn = TestSubnet(subtensor, netuid=0)
        tempo_response = root_sn.execute_one(
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, 0, TEMPO_TO_SET)
        )
        assert tempo_response.success, f"Failed to set tempo: {tempo_response.message}"

        # Wait a few blocks after setting tempo to ensure state is synchronized
        # This prevents "Transaction is outdated" errors by allowing the chain to process
        # the tempo change and stabilize before we submit the announce transaction.
        current_block = subtensor.block
        subtensor.wait_for_block(current_block + 5)

        # Generate and announce a key first using alice_wallet (which is a validator in localnet)
        public_key_bytes = generate_mlkem768_keypair()
        current_epoch = subtensor.mev_shield.get_mev_shield_epoch()
        next_epoch = current_epoch + 1

        logging.info(
            f"NextKey not found, announcing key for epoch {next_epoch} using Alice wallet (validator)"
        )

        # Announce key using validator wallet with explicit period to avoid transaction expiration
        # Use a longer period (256 blocks) to ensure transaction doesn't expire during epoch transitions
        announce_response = announce_next_key_extrinsic(
            subtensor=subtensor,
            wallet=alice_wallet,
            public_key=public_key_bytes,
            epoch=next_epoch,
            period=256,  # Longer period to avoid "Transaction is outdated" errors
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )

        assert announce_response.success, (
            f"Failed to announce next key: {announce_response.message}"
        )
        assert announce_response.extrinsic_receipt is not None, (
            "Extrinsic receipt should be available"
        )
        assert announce_response.extrinsic_receipt.is_success, (
            f"Extrinsic execution failed: {announce_response.extrinsic_receipt.error_message}"
        )

        # Wait for block to ensure key is set
        current_block = subtensor.block
        subtensor.wait_for_block(current_block + 1)

        # Verify NextKey is now set
        next_key_result = subtensor.mev_shield.get_mev_shield_next_key()
        assert next_key_result is not None, "NextKey should be set after announcement"

    pk_bytes, epoch = next_key_result
    logging.info(f"Using NextKey with epoch {epoch} for encryption")

    # Create a simple transfer call
    transfer_call = SubtensorModule(subtensor).transfer(
        dest=bob_wallet.coldkey.ss58_address,
        value=1000000,  # 1 TAO in RAO
    )

    # Get nonce for commitment calculation
    nonce = subtensor.substrate.get_account_next_index(bob_wallet.coldkey.ss58_address)
    signer_bytes = bob_wallet.coldkey.public_key
    nonce_bytes = nonce.to_bytes(4, "little")
    scale_call_bytes = transfer_call.encode()
    payload_core = signer_bytes + nonce_bytes + scale_call_bytes

    # Compute expected commitment (blake2_256 = blake2b with digest_size=32)
    expected_commitment_hash = hashlib.blake2b(payload_core, digest_size=32).digest()
    expected_commitment_hex = "0x" + expected_commitment_hash.hex()

    logging.info(
        f"Submitting encrypted extrinsic with expected commitment: {expected_commitment_hex}"
    )

    # Submit encrypted extrinsic
    response = submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=bob_wallet,
        call=transfer_call,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert response.success, f"Failed to submit encrypted extrinsic: {response.message}"

    logging.info("Encrypted extrinsic submitted successfully, waiting for block...")

    # Wait for block to be finalized
    current_block = subtensor.block
    subtensor.wait_for_block(current_block + 1)

    # Find submission by commitment
    all_submissions = subtensor.mev_shield.get_mev_shield_submission()
    submission = None
    for sub_id, sub_data in all_submissions.items():
        if sub_data["commitment"] == expected_commitment_hex:
            submission = sub_data
            break

    assert submission is not None, (
        f"Submission not found in storage. Expected commitment: {expected_commitment_hex}"
    )

    assert submission["commitment"] == expected_commitment_hex, (
        f"Commitment mismatch: expected {expected_commitment_hex}, got {submission['commitment']}"
    )
    assert submission["author"] == bob_wallet.coldkey.ss58_address, (
        f"Author mismatch: expected {bob_wallet.coldkey.ss58_address}, got {submission['author']}"
    )
    assert isinstance(submission["submitted_in"], int), "submitted_in should be int"
    assert submission["submitted_in"] > 0, "submitted_in should be positive"
    assert isinstance(submission["ciphertext"], bytes), "ciphertext should be bytes"
    assert len(submission["ciphertext"]) > 0, "ciphertext should not be empty"

    logging.info(
        f"Submission verified: commitment={submission['commitment']}, "
        f"author={submission['author']}, submitted_in={submission['submitted_in']}"
    )


def test_mev_shield_key_rotation(subtensor, alice_wallet):
    """Tests key rotation from NextKey to CurrentKey.

    Steps:
    1. Generate ML-KEM-768 keypair
    2. Announce NextKey
    3. Verify NextKey is set
    4. Wait for next block (triggers rotation)
    5. Verify CurrentKey now contains the old NextKey
    6. Verify Epoch has been updated
    """
    # add founds to alice HK
    response = add_balance_to_wallet_hk(subtensor, alice_wallet, 100)
    assert response.success, response.message

    # Set tempo to 100 blocks for root network (netuid=0) to ensure stable epoch timing
    # MEV Shield epoch depends on root network tempo, so we need predictable epoch boundaries
    TEMPO_TO_SET = 100
    root_sn = TestSubnet(subtensor, netuid=0)
    tempo_response = root_sn.execute_one(
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, 0, TEMPO_TO_SET)
    )
    assert tempo_response.success, f"Failed to set tempo: {tempo_response.message}"

    # Wait a few blocks after setting tempo to ensure state is synchronized
    # This prevents "Transaction is outdated" errors by allowing the chain to process
    # the tempo change and stabilize before we submit the announce transaction.
    current_block = subtensor.block
    subtensor.wait_for_block(current_block + 5)

    # Generate ML-KEM-768 keypair using bittensor_drand
    public_key_bytes = generate_mlkem768_keypair()

    # Get current epoch
    current_epoch = subtensor.mev_shield.get_mev_shield_epoch()
    next_epoch = current_epoch + 1

    logging.info(f"Testing key rotation: announcing NextKey for epoch {next_epoch}")

    # Announce NextKey with explicit period to avoid transaction expiration
    # Use a longer period (256 blocks) to ensure transaction doesn't expire during epoch transitions
    response = announce_next_key_extrinsic(
        subtensor=subtensor,
        wallet=alice_wallet,
        public_key=public_key_bytes,
        epoch=next_epoch,
        period=256,  # Longer period to avoid "Transaction is outdated" errors
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert response.success, f"Failed to announce next key: {response.message}"

    # Verify that extrinsic was actually executed successfully (not just included)
    assert response.extrinsic_receipt is not None, (
        "Extrinsic receipt should be available"
    )
    assert response.extrinsic_receipt.is_success, (
        f"Extrinsic execution failed: {response.extrinsic_receipt.error_message}"
    )

    # Verify NextKey is set
    next_key_before = subtensor.mev_shield.get_mev_shield_next_key()
    assert next_key_before is not None, "NextKey should be set after announcement"
    pk_bytes_before, epoch_before = next_key_before
    assert pk_bytes_before == public_key_bytes, (
        "NextKey public key should match announced key"
    )
    assert epoch_before == next_epoch, (
        f"NextKey epoch should be {next_epoch}, got {epoch_before}"
    )

    logging.info(f"NextKey set: epoch={epoch_before}, waiting for rotation...")

    # Wait for next block (this triggers rotation: NextKey -> CurrentKey)
    current_block = subtensor.block
    subtensor.wait_for_block(current_block + 1)

    # Verify CurrentKey now contains the old NextKey
    current_key_after = subtensor.mev_shield.get_mev_shield_current_key()
    assert current_key_after is not None, "CurrentKey should be set after rotation"

    pk_bytes_after, epoch_after = current_key_after
    assert pk_bytes_after == public_key_bytes, (
        "CurrentKey should contain the old NextKey"
    )
    assert epoch_after == next_epoch, (
        f"CurrentKey epoch should be {next_epoch}, got {epoch_after}"
    )

    # Verify Epoch has been updated
    new_epoch = subtensor.mev_shield.get_mev_shield_epoch()
    assert new_epoch == next_epoch, (
        f"Epoch should be updated to {next_epoch}, got {new_epoch}"
    )

    logging.info(
        f"Key rotation verified: CurrentKey epoch={epoch_after}, global epoch={new_epoch}"
    )


def test_mev_shield_commitment_verification(subtensor, bob_wallet, alice_wallet):
    """Tests commitment verification by recomputing it from payload.

    Steps:
    1. Ensure NextKey exists
    2. Create a call
    3. Submit encrypted extrinsic
    4. Get submission from storage
    5. Recompute payload_core from call parameters
    6. Recompute commitment (blake2b)
    7. Verify commitment from storage matches recomputed value
    """

    # add founds to alice, bob HK
    for w in [alice_wallet, bob_wallet]:
        response = add_balance_to_wallet_hk(subtensor, w, 100)
        assert response.success, response.message

    # Ensure NextKey exists - if not, we need to announce it first using alice_wallet (validator)
    next_key_result = subtensor.mev_shield.get_mev_shield_next_key()
    if next_key_result is None:
        # Set tempo to 100 blocks for root network (netuid=0) to ensure stable epoch timing
        # MEV Shield epoch depends on root network tempo, so we need predictable epoch boundaries
        TEMPO_TO_SET = 100
        root_sn = TestSubnet(subtensor, netuid=0)
        tempo_response = root_sn.execute_one(
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, 0, TEMPO_TO_SET)
        )
        assert tempo_response.success, f"Failed to set tempo: {tempo_response.message}"

        # Wait a few blocks after setting tempo to ensure state is synchronized
        # This prevents "Transaction is outdated" errors by allowing the chain to process
        # the tempo change and stabilize before we submit the announce transaction.
        current_block = subtensor.block
        subtensor.wait_for_block(current_block + 5)

        # Generate and announce a key first using alice_wallet (which is a validator in localnet)
        public_key_bytes = generate_mlkem768_keypair()
        current_epoch = subtensor.mev_shield.get_mev_shield_epoch()
        next_epoch = current_epoch + 1

        logging.info(
            f"NextKey not found, announcing key for epoch {next_epoch} using Alice wallet (validator)"
        )

        # Announce key using validator wallet with explicit period to avoid transaction expiration
        # Use a longer period (256 blocks) to ensure transaction doesn't expire during epoch transitions
        announce_response = announce_next_key_extrinsic(
            subtensor=subtensor,
            wallet=alice_wallet,
            public_key=public_key_bytes,
            epoch=next_epoch,
            period=256,  # Longer period to avoid "Transaction is outdated" errors
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )

        assert announce_response.success, (
            f"Failed to announce next key: {announce_response.message}"
        )
        assert announce_response.extrinsic_receipt is not None, (
            "Extrinsic receipt should be available"
        )
        assert announce_response.extrinsic_receipt.is_success, (
            f"Extrinsic execution failed: {announce_response.extrinsic_receipt.error_message}"
        )

        # Wait for block to ensure key is set
        current_block = subtensor.block
        subtensor.wait_for_block(current_block + 1)

        # Verify NextKey is now set
        next_key_result = subtensor.mev_shield.get_mev_shield_next_key()
        assert next_key_result is not None, "NextKey should be set after announcement"

    # Create a simple transfer call
    transfer_call = SubtensorModule(subtensor).transfer(
        dest=bob_wallet.coldkey.ss58_address,
        value=2000000,  # 2 TAO in RAO
    )

    # Get nonce that will be used
    nonce = subtensor.substrate.get_account_next_index(bob_wallet.coldkey.ss58_address)

    # Compute payload_core: signer (32B) + nonce (u32 LE, 4B) + SCALE(call)
    signer_bytes = bob_wallet.coldkey.public_key
    nonce_bytes = nonce.to_bytes(4, "little")
    scale_call_bytes = transfer_call.encode()
    payload_core = signer_bytes + nonce_bytes + scale_call_bytes

    # Compute expected commitment (blake2_256 = blake2b with digest_size=32)
    expected_commitment_hash = hashlib.blake2b(payload_core, digest_size=32).digest()
    expected_commitment_hex = "0x" + expected_commitment_hash.hex()

    logging.info(
        f"Submitting encrypted extrinsic, expected commitment: {expected_commitment_hex}"
    )

    # Submit encrypted extrinsic
    response = submit_encrypted_extrinsic(
        subtensor=subtensor,
        wallet=bob_wallet,
        call=transfer_call,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert response.success, f"Failed to submit encrypted extrinsic: {response.message}"

    # Wait for block
    current_block = subtensor.block
    subtensor.wait_for_block(current_block + 1)

    # Find submission by commitment
    all_submissions = subtensor.mev_shield.get_mev_shield_submission()
    submission = None
    for sub_id, sub_data in all_submissions.items():
        if sub_data["commitment"] == expected_commitment_hex:
            submission = sub_data
            break

    assert submission is not None, (
        f"Submission not found with commitment {expected_commitment_hex}"
    )

    # Verify commitment matches
    storage_commitment = submission["commitment"]
    assert storage_commitment == expected_commitment_hex, (
        f"Commitment mismatch: storage has {storage_commitment}, "
        f"expected {expected_commitment_hex}"
    )

    # Recompute commitment to double-check
    recomputed_commitment_hash = hashlib.blake2b(payload_core, digest_size=32).digest()
    recomputed_commitment_hex = "0x" + recomputed_commitment_hash.hex()

    assert recomputed_commitment_hex == expected_commitment_hex, (
        "Recomputed commitment should match expected"
    )
    assert storage_commitment == recomputed_commitment_hex, (
        f"Storage commitment {storage_commitment} should match recomputed {recomputed_commitment_hex}"
    )

    logging.info(f"Commitment verification passed: {storage_commitment}")
