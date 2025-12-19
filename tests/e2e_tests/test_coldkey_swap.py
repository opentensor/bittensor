import pytest

from bittensor import logging
from bittensor.utils.balance import Balance


def test_coldkey_swap(subtensor, alice_wallet, bob_wallet, charlie_wallet):
    """
    Sync test for coldkey swap extrinsics.

    This comprehensive test covers:
    1. Happy Path - Successful swap flow:
       - Step 1: Announce coldkey swap from Alice to Bob
       - Step 2: Verify announcement was created and contains correct data
       - Step 3: Verify coldkey swap constants are accessible
       - Step 4: Wait for execution block (50 blocks delay)
       - Step 5: Execute the swap
       - Step 6: Verify announcement was removed after successful swap

    2. Error cases for swap_coldkey_announced:
       - Error 1: Attempt to execute swap without prior announcement
       - Error 2: Attempt to execute swap with incorrect coldkey hash (mismatch)
       - Error 3: Attempt to execute swap too early (before execution block)

    3. Error cases for announce_coldkey_swap:
       - Error 4: Attempt to create duplicate announcement (reannouncement behavior)

    4. Transaction blocking after announcement:
       - Step 1: Create announcement
       - Step 2: Attempt to execute other transaction (transfer) from announced coldkey
       - Step 3: Verify transaction is blocked (except swap_coldkey_announced)

    Notes:
        - Uses fast blocks mode (50 blocks delay instead of 5 days)
        - All operations use async_subtensor for async execution
        - Each test section cleans up after itself
    """
    logging.console.info("Starting coldkey swap E2E test")

    # === 1. Happy Path - Successful swap ===
    logging.console.info("Testing Happy Path - successful swap")

    # Step 1: Alice announces swap to new coldkey (Bob)
    logging.console.info("Step 1: Alice announces swap to Bob")
    response = subtensor.extrinsics.announce_coldkey_swap(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to announce swap: {response.message}"

    # Step 2: Verify announcement was created
    logging.console.info("Step 2: Verify announcement was created")
    announcement = subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address
    )
    assert announcement is not None, "Announcement should exist"
    assert announcement.coldkey == alice_wallet.coldkeypub.ss58_address
    assert announcement.execution_block > subtensor.chain.get_current_block()

    # Step 3: Verify constants and storage values
    logging.console.info("Step 3: Verify constants and storage values")
    constants = subtensor.wallets.get_coldkey_swap_constants()
    assert constants.KeySwapCost is not None

    announcement_delay = subtensor.wallets.get_coldkey_swap_announcement_delay()
    reannouncement_delay = subtensor.wallets.get_coldkey_swap_reannouncement_delay()

    assert announcement_delay is not None
    assert reannouncement_delay is not None
    logging.console.info(
        f"Constants: AnnouncementDelay={announcement_delay}, "
        f"ReannouncementDelay={reannouncement_delay}, "
        f"KeySwapCost={constants.KeySwapCost}"
    )

    # Step 4: Wait for 50 blocks (execution_block)
    logging.console.info("Step 4: Waiting for execution block")
    current_block = subtensor.chain.get_current_block()
    execution_block = announcement.execution_block
    logging.console.info(
        f"Current block: {current_block}, Execution block: {execution_block}"
    )
    subtensor.wait_for_block(execution_block + 1)

    # Step 5: Execute swap
    logging.console.info("Step 5: Executing swap")
    response = subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to execute swap: {response.message}"

    # Step 6: Verify announcement was removed after swap
    logging.console.info("Step 6: Verify announcement was removed after swap")
    announcement_after = subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address
    )
    assert announcement_after is None, "Announcement should be removed after swap"

    logging.console.info("Happy Path completed successfully")

    # Refund Alice balance for further tests (Bob now has all Alice's funds after swap)
    logging.console.info("Refunding Alice balance for further tests")
    bob_balance = subtensor.wallets.get_balance(bob_wallet.coldkeypub.ss58_address)
    refund_amount = Balance.from_tao(10)
    assert bob_balance > refund_amount, (
        f"Bob balance ({bob_balance}) too low to refund Alice ({refund_amount})"
    )
    response = subtensor.extrinsics.transfer(
        wallet=bob_wallet,
        destination_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=refund_amount,
    )
    assert response.success, f"Failed to refund Alice: {response.message}"
    logging.console.info("Alice balance refunded successfully")

    # === 2. Error cases for swap_coldkey_announced ===
    logging.console.info("Testing errors for swap_coldkey_announced")

    # Error 1: Attempt to execute swap without announcement
    logging.console.info("Error 1: Attempting swap without announcement")
    response = subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
        raise_error=False,
    )
    assert not response.success, "Should fail without announcement"
    assert "No coldkey swap announcement found" in response.message
    logging.console.info("Error 1 passed: No announcement error")

    # Error 2: Hash mismatch
    logging.console.info("Error 2: Testing hash mismatch")
    # Alice announces swap to Bob
    response = subtensor.extrinsics.announce_coldkey_swap(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to announce swap: {response.message}"

    # Attempt to execute swap with incorrect coldkey (Charlie instead of Bob)
    response = subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=charlie_wallet.coldkeypub.ss58_address,
        raise_error=False,
    )
    assert not response.success, "Should fail with hash mismatch"
    assert "hash does not match" in response.message.lower()
    logging.console.info("Error 2 passed: Hash mismatch error")

    # Cleanup: Remove announcement from Error 2 by waiting and executing swap
    logging.console.info("Cleaning up announcement from Error 2")
    announcement_from_error2 = subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address
    )
    assert announcement_from_error2 is not None, (
        "Announcement from Error 2 should exist"
    )
    # Wait for execution block (wait_for_block is safe even if block already passed)
    subtensor.wait_for_block(announcement_from_error2.execution_block + 1)
    response = subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to cleanup announcement: {response.message}"
    logging.console.info("Cleaned up announcement from Error 2")
    # Refund Alice balance after swap
    bob_balance = subtensor.wallets.get_balance(bob_wallet.coldkeypub.ss58_address)
    refund_amount = Balance.from_tao(10)
    assert bob_balance > refund_amount, (
        f"Bob balance ({bob_balance}) too low to refund Alice ({refund_amount})"
    )
    response = subtensor.extrinsics.transfer(
        wallet=bob_wallet,
        destination_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=refund_amount,
    )
    assert response.success, f"Failed to refund Alice: {response.message}"
    logging.console.info("Refunded Alice balance after cleanup")

    # Error 3: Too early (before execution block)
    logging.console.info("Error 3: Testing too early error")
    # Create new announcement for this test
    response = subtensor.extrinsics.announce_coldkey_swap(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to announce swap: {response.message}"
    announcement = subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address
    )
    assert announcement is not None, "Announcement should exist for Error 3 test"

    # Attempt to execute swap immediately (before execution_block)
    current_block = subtensor.chain.get_current_block()
    assert current_block < announcement.execution_block, (
        "Current block should be before execution block"
    )
    response = subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
        raise_error=False,
    )
    assert not response.success, "Should fail with too early error"
    assert "too early" in response.message.lower()
    assert str(announcement.execution_block) in response.message
    logging.console.info("Error 3 passed: Too early error")

    # Wait for execution_block and execute swap for cleanup
    subtensor.wait_for_block(announcement.execution_block)
    response = subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, response.message
    logging.console.info("Cleaned up announcement by executing swap")
    # Refund Alice balance after swap

    refund_amount = Balance.from_tao(10)
    response = subtensor.extrinsics.transfer(
        wallet=bob_wallet,
        destination_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=refund_amount,
    )
    assert response.success, response.message

    # === 3. Error cases for announce_coldkey_swap ===
    logging.console.info("Testing errors for announce_coldkey_swap")

    # Error 4: Duplicate announcement (reannouncement)
    logging.console.info("Error 4: Testing duplicate announcement")
    # Create first announcement
    response = subtensor.extrinsics.announce_coldkey_swap(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to announce swap: {response.message}"

    # Attempt to create second announcement (to Charlie)
    response = subtensor.extrinsics.announce_coldkey_swap(
        wallet=alice_wallet,
        new_coldkey_ss58=charlie_wallet.coldkeypub.ss58_address,
        raise_error=False,
    )
    assert not response.success, "Should fail with duplicate announcement"

    # Verify that there is an active announcement
    announcement = subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address
    )
    assert announcement is not None, "Should have an active announcement"
    logging.console.info("Error 4: Duplicate announcement handled")

    # Cleanup: Remove announcement from Error 4 by waiting and executing swap
    logging.console.info("Cleaning up announcement from Error 4")
    from bittensor.core.extrinsics.utils import verify_coldkey_hash
    from bittensor_wallet import Keypair

    # Determine which coldkey matches the announcement hash
    bob_keypair = Keypair(ss58_address=bob_wallet.coldkeypub.ss58_address)
    charlie_keypair = Keypair(ss58_address=charlie_wallet.coldkeypub.ss58_address)

    assert verify_coldkey_hash(
        bob_keypair, announcement.new_coldkey_hash
    ) or verify_coldkey_hash(charlie_keypair, announcement.new_coldkey_hash), (
        "Announcement hash should match either Bob or Charlie"
    )

    # Use the matching coldkey
    target_coldkey = (
        bob_wallet.coldkeypub.ss58_address
        if verify_coldkey_hash(bob_keypair, announcement.new_coldkey_hash)
        else charlie_wallet.coldkeypub.ss58_address
    )
    refund_wallet = (
        bob_wallet
        if verify_coldkey_hash(bob_keypair, announcement.new_coldkey_hash)
        else charlie_wallet
    )

    # Wait for execution block
    subtensor.wait_for_block(announcement.execution_block + 1)
    response = subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=target_coldkey,
    )
    assert response.success, f"Failed to cleanup announcement: {response.message}"
    logging.console.info("Cleaned up announcement from Error 4")

    # Refund Alice balance after swap
    refund_amount = Balance.from_tao(10)
    refund_balance = subtensor.wallets.get_balance(
        refund_wallet.coldkeypub.ss58_address
    )
    assert refund_balance > refund_amount, (
        f"{refund_wallet.name} balance ({refund_balance}) too low to refund Alice ({refund_amount})"
    )
    response = subtensor.extrinsics.transfer(
        wallet=refund_wallet,
        destination_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=refund_amount,
    )
    assert response.success, f"Failed to refund Alice: {response.message}"
    logging.console.info(
        f"Refunded Alice balance from {refund_wallet.name} after cleanup"
    )

    # === 4. Transaction blocking after announcement ===
    logging.console.info("Testing transaction blocking after announcement")

    # Step 1: Alice announces swap to Bob
    logging.console.info("Step 1: Alice announces swap to Bob")
    # Verify no existing announcement
    existing_announcement = subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address
    )
    assert existing_announcement is None, (
        "No announcement should exist before creating new one"
    )
    response = subtensor.extrinsics.announce_coldkey_swap(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to announce swap: {response.message}"

    # Step 2: Attempt to execute other transaction from Alice (transfer)
    logging.console.info("Step 2: Attempting transfer transaction (should be blocked)")
    transfer_value = Balance.from_tao(1)
    dest_coldkey = charlie_wallet.coldkeypub.ss58_address

    response = subtensor.extrinsics.transfer(
        wallet=alice_wallet,
        destination_ss58=dest_coldkey,
        amount=transfer_value,
        raise_error=False,
    )

    # Step 3: Verify transaction is blocked
    assert not response.success, "Transfer should be blocked after announcement"
    # Error code 0 corresponds to ColdkeySwapAnnounced (see CustomTransactionError enum)
    # The message may contain "Custom error: 0" or specific text about swap
    assert (
        "Custom error: 0" in response.message
        or "ColdkeySwapAnnounced" in response.message
        or "swap" in response.message.lower()
    ), (
        f"Expected transaction to be blocked by ColdkeySwapAnnounced, got: {response.message}"
    )
    logging.console.info("Transaction blocking test passed")

    # Cleanup: wait for execution_block and execute swap
    announcement = subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address
    )
    assert announcement is not None, "Announcement should exist for cleanup"
    subtensor.wait_for_block(announcement.execution_block + 1)
    response = subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to cleanup announcement: {response.message}"
    logging.console.info("Cleaned up announcement by executing swap")

    logging.console.info("All coldkey swap E2E tests completed successfully")


@pytest.mark.asyncio
async def test_coldkey_swap_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet
):
    """
    Async test for coldkey swap extrinsics.

    This comprehensive test covers:
    1. Happy Path - Successful swap flow:
       - Step 1: Announce coldkey swap from Alice to Bob
       - Step 2: Verify announcement was created and contains correct data
       - Step 3: Verify coldkey swap constants are accessible
       - Step 4: Wait for execution block (50 blocks delay)
       - Step 5: Execute the swap
       - Step 6: Verify announcement was removed after successful swap

    2. Error cases for swap_coldkey_announced:
       - Error 1: Attempt to execute swap without prior announcement
       - Error 2: Attempt to execute swap with incorrect coldkey hash (mismatch)
       - Error 3: Attempt to execute swap too early (before execution block)

    3. Error cases for announce_coldkey_swap:
       - Error 4: Attempt to create duplicate announcement (reannouncement behavior)

    4. Transaction blocking after announcement:
       - Step 1: Create announcement
       - Step 2: Attempt to execute other transaction (transfer) from announced coldkey
       - Step 3: Verify transaction is blocked (except swap_coldkey_announced)

    Notes:
        - Uses fast blocks mode (50 blocks delay instead of 5 days)
        - All operations use async_subtensor for async execution
        - Each test section cleans up after itself
    """
    logging.console.info("Starting coldkey swap E2E test")

    # === 1. Happy Path - Successful swap ===
    logging.console.info("Testing Happy Path - successful swap")

    # Step 1: Alice announces swap to new coldkey (Bob)
    logging.console.info("Step 1: Alice announces swap to Bob")
    response = await async_subtensor.extrinsics.announce_coldkey_swap(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to announce swap: {response.message}"

    # Step 2: Verify announcement was created
    logging.console.info("Step 2: Verify announcement was created")
    announcement = await async_subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address
    )
    assert announcement is not None, "Announcement should exist"
    assert announcement.coldkey == alice_wallet.coldkeypub.ss58_address
    assert (
        announcement.execution_block > await async_subtensor.chain.get_current_block()
    )

    # Step 3: Verify constants and storage values
    logging.console.info("Step 3: Verify constants and storage values")
    constants = await async_subtensor.wallets.get_coldkey_swap_constants()
    assert constants.KeySwapCost is not None

    announcement_delay = (
        await async_subtensor.wallets.get_coldkey_swap_announcement_delay()
    )
    reannouncement_delay = (
        await async_subtensor.wallets.get_coldkey_swap_reannouncement_delay()
    )

    assert announcement_delay is not None
    assert reannouncement_delay is not None
    logging.console.info(
        f"Constants: AnnouncementDelay={announcement_delay}, "
        f"ReannouncementDelay={reannouncement_delay}, "
        f"KeySwapCost={constants.KeySwapCost}"
    )

    # Step 4: Wait for 50 blocks (execution_block)
    logging.console.info("Step 4: Waiting for execution block")
    current_block = await async_subtensor.chain.get_current_block()
    execution_block = announcement.execution_block
    logging.console.info(
        f"Current block: {current_block}, Execution block: {execution_block}"
    )
    await async_subtensor.wait_for_block(execution_block + 1)

    # Step 5: Execute swap
    logging.console.info("Step 5: Executing swap")
    response = await async_subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to execute swap: {response.message}"

    # Step 6: Verify announcement was removed after swap
    logging.console.info("Step 6: Verify announcement was removed after swap")
    announcement_after = await async_subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address
    )
    assert announcement_after is None, "Announcement should be removed after swap"

    logging.console.info("Happy Path completed successfully")

    # Refund Alice balance for further tests (Bob now has all Alice's funds after swap)
    logging.console.info("Refunding Alice balance for further tests")
    bob_balance = await async_subtensor.wallets.get_balance(
        bob_wallet.coldkeypub.ss58_address
    )
    refund_amount = Balance.from_tao(10)
    assert bob_balance > refund_amount, (
        f"Bob balance ({bob_balance}) too low to refund Alice ({refund_amount})"
    )
    response = await async_subtensor.extrinsics.transfer(
        wallet=bob_wallet,
        destination_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=refund_amount,
    )
    assert response.success, f"Failed to refund Alice: {response.message}"
    logging.console.info("Alice balance refunded successfully")

    # === 2. Error cases for swap_coldkey_announced ===
    logging.console.info("Testing errors for swap_coldkey_announced")

    # Error 1: Attempt to execute swap without announcement
    logging.console.info("Error 1: Attempting swap without announcement")
    response = await async_subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
        raise_error=False,
    )
    assert not response.success, "Should fail without announcement"
    assert "No coldkey swap announcement found" in response.message
    logging.console.info("Error 1 passed: No announcement error")

    # Error 2: Hash mismatch
    logging.console.info("Error 2: Testing hash mismatch")
    # Alice announces swap to Bob
    response = await async_subtensor.extrinsics.announce_coldkey_swap(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to announce swap: {response.message}"

    # Attempt to execute swap with incorrect coldkey (Charlie instead of Bob)
    response = await async_subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=charlie_wallet.coldkeypub.ss58_address,
        raise_error=False,
    )
    assert not response.success, "Should fail with hash mismatch"
    assert "hash does not match" in response.message.lower()
    logging.console.info("Error 2 passed: Hash mismatch error")

    # Cleanup: Remove announcement from Error 2 by waiting and executing swap
    logging.console.info("Cleaning up announcement from Error 2")
    announcement_from_error2 = (
        await async_subtensor.wallets.get_coldkey_swap_announcement(
            coldkey_ss58=alice_wallet.coldkeypub.ss58_address
        )
    )
    assert announcement_from_error2 is not None, (
        "Announcement from Error 2 should exist"
    )
    # Wait for execution block (wait_for_block is safe even if block already passed)
    await async_subtensor.wait_for_block(announcement_from_error2.execution_block + 1)
    response = await async_subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to cleanup announcement: {response.message}"
    logging.console.info("Cleaned up announcement from Error 2")
    # Refund Alice balance after swap
    bob_balance = await async_subtensor.wallets.get_balance(
        bob_wallet.coldkeypub.ss58_address
    )
    refund_amount = Balance.from_tao(10)
    assert bob_balance > refund_amount, (
        f"Bob balance ({bob_balance}) too low to refund Alice ({refund_amount})"
    )
    response = await async_subtensor.extrinsics.transfer(
        wallet=bob_wallet,
        destination_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=refund_amount,
    )
    assert response.success, f"Failed to refund Alice: {response.message}"
    logging.console.info("Refunded Alice balance after cleanup")

    # Error 3: Too early (before execution block)
    logging.console.info("Error 3: Testing too early error")
    # Create new announcement for this test
    response = await async_subtensor.extrinsics.announce_coldkey_swap(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to announce swap: {response.message}"
    announcement = await async_subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address
    )
    assert announcement is not None, "Announcement should exist for Error 3 test"

    # Attempt to execute swap immediately (before execution_block)
    current_block = await async_subtensor.chain.get_current_block()
    assert current_block < announcement.execution_block, (
        "Current block should be before execution block"
    )
    response = await async_subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
        raise_error=False,
    )
    assert not response.success, "Should fail with too early error"
    assert "too early" in response.message.lower()
    assert str(announcement.execution_block) in response.message
    logging.console.info("Error 3 passed: Too early error")

    # Wait for execution_block and execute swap for cleanup
    await async_subtensor.wait_for_block(announcement.execution_block)
    response = await async_subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, response.message
    logging.console.info("Cleaned up announcement by executing swap")
    # Refund Alice balance after swap

    refund_amount = Balance.from_tao(10)
    response = await async_subtensor.extrinsics.transfer(
        wallet=bob_wallet,
        destination_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=refund_amount,
    )
    assert response.success, response.message

    # === 3. Error cases for announce_coldkey_swap ===
    logging.console.info("Testing errors for announce_coldkey_swap")

    # Error 4: Duplicate announcement (reannouncement)
    logging.console.info("Error 4: Testing duplicate announcement")
    # Create first announcement
    response = await async_subtensor.extrinsics.announce_coldkey_swap(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to announce swap: {response.message}"

    # Attempt to create second announcement (to Charlie)
    response = await async_subtensor.extrinsics.announce_coldkey_swap(
        wallet=alice_wallet,
        new_coldkey_ss58=charlie_wallet.coldkeypub.ss58_address,
        raise_error=False,
    )
    assert not response.success, "Should fail with duplicate announcement"

    # Verify that there is an active announcement
    announcement = await async_subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address
    )
    assert announcement is not None, "Should have an active announcement"
    logging.console.info("Error 4: Duplicate announcement handled")

    # Cleanup: Remove announcement from Error 4 by waiting and executing swap
    logging.console.info("Cleaning up announcement from Error 4")
    from bittensor.core.extrinsics.utils import verify_coldkey_hash
    from bittensor_wallet import Keypair

    # Determine which coldkey matches the announcement hash
    bob_keypair = Keypair(ss58_address=bob_wallet.coldkeypub.ss58_address)
    charlie_keypair = Keypair(ss58_address=charlie_wallet.coldkeypub.ss58_address)

    assert verify_coldkey_hash(
        bob_keypair, announcement.new_coldkey_hash
    ) or verify_coldkey_hash(charlie_keypair, announcement.new_coldkey_hash), (
        "Announcement hash should match either Bob or Charlie"
    )

    # Use the matching coldkey
    target_coldkey = (
        bob_wallet.coldkeypub.ss58_address
        if verify_coldkey_hash(bob_keypair, announcement.new_coldkey_hash)
        else charlie_wallet.coldkeypub.ss58_address
    )
    refund_wallet = (
        bob_wallet
        if verify_coldkey_hash(bob_keypair, announcement.new_coldkey_hash)
        else charlie_wallet
    )

    # Wait for execution block
    await async_subtensor.wait_for_block(announcement.execution_block + 1)
    response = await async_subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=target_coldkey,
    )
    assert response.success, f"Failed to cleanup announcement: {response.message}"
    logging.console.info("Cleaned up announcement from Error 4")

    # Refund Alice balance after swap
    refund_amount = Balance.from_tao(10)
    refund_balance = await async_subtensor.wallets.get_balance(
        refund_wallet.coldkeypub.ss58_address
    )
    assert refund_balance > refund_amount, (
        f"{refund_wallet.name} balance ({refund_balance}) too low to refund Alice ({refund_amount})"
    )
    response = await async_subtensor.extrinsics.transfer(
        wallet=refund_wallet,
        destination_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=refund_amount,
    )
    assert response.success, f"Failed to refund Alice: {response.message}"
    logging.console.info(
        f"Refunded Alice balance from {refund_wallet.name} after cleanup"
    )

    # === 4. Transaction blocking after announcement ===
    logging.console.info("Testing transaction blocking after announcement")

    # Step 1: Alice announces swap to Bob
    logging.console.info("Step 1: Alice announces swap to Bob")
    # Verify no existing announcement
    existing_announcement = await async_subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address
    )
    assert existing_announcement is None, (
        "No announcement should exist before creating new one"
    )
    response = await async_subtensor.extrinsics.announce_coldkey_swap(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to announce swap: {response.message}"

    # Step 2: Attempt to execute other transaction from Alice (transfer)
    logging.console.info("Step 2: Attempting transfer transaction (should be blocked)")
    transfer_value = Balance.from_tao(1)
    dest_coldkey = charlie_wallet.coldkeypub.ss58_address

    response = await async_subtensor.extrinsics.transfer(
        wallet=alice_wallet,
        destination_ss58=dest_coldkey,
        amount=transfer_value,
        raise_error=False,
    )

    # Step 3: Verify transaction is blocked
    assert not response.success, "Transfer should be blocked after announcement"
    # Error code 0 corresponds to ColdkeySwapAnnounced (see CustomTransactionError enum)
    # The message may contain "Custom error: 0" or specific text about swap
    assert (
        "Custom error: 0" in response.message
        or "ColdkeySwapAnnounced" in response.message
        or "swap" in response.message.lower()
    ), (
        f"Expected transaction to be blocked by ColdkeySwapAnnounced, got: {response.message}"
    )
    logging.console.info("Transaction blocking test passed")

    # Cleanup: wait for execution_block and execute swap
    announcement = await async_subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address
    )
    assert announcement is not None, "Announcement should exist for cleanup"
    await async_subtensor.wait_for_block(announcement.execution_block + 1)
    response = await async_subtensor.extrinsics.swap_coldkey_announced(
        wallet=alice_wallet,
        new_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to cleanup announcement: {response.message}"
    logging.console.info("Cleaned up announcement by executing swap")

    logging.console.info("All coldkey swap E2E tests completed successfully")
