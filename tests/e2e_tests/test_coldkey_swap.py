import pytest

from bittensor import logging
from bittensor.core.extrinsics.asyncex.sudo import (
    reset_coldkey_swap_extrinsic as async_reset_coldkey_swap_extrinsic,
    swap_coldkey_extrinsic as async_swap_coldkey_extrinsic,
)
from bittensor.core.extrinsics.sudo import (
    reset_coldkey_swap_extrinsic,
    swap_coldkey_extrinsic,
)
from bittensor.utils.balance import Balance


def test_coldkey_swap(subtensor, alice_wallet, bob_wallet, charlie_wallet, dave_wallet):
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

    5. Dispute and root reset:
       - Step 1: Dave announces swap, then disputes it (dispute_coldkey_swap)
       - Step 2: Verify dispute is recorded (get_coldkey_swap_dispute)
       - Step 3: Verify account is blocked (transfer fails)
       - Step 4: Root resets coldkey swap (reset_coldkey_swap)
       - Step 5: Verify dispute and announcement are cleared
       - Step 6: Verify transfers are unblocked after reset

    6. Root swap override:
       - Step 1: Root swaps Dave to Charlie without announcement
       - Step 2: Verify announcement and dispute are cleared
       - Step 3: Verify old coldkey is reaped

    Notes:
        - Uses fast blocks mode (50 blocks delay instead of 5 days)
        - All operations use subtensor for sync execution
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
    swap_cost = Balance.from_rao(constants.KeySwapCost)
    swap_cost_rao = int(constants.KeySwapCost)
    existential_deposit = subtensor.chain.get_existential_deposit()
    logging.console.info(
        f"Constants: AnnouncementDelay={announcement_delay}, "
        f"ReannouncementDelay={reannouncement_delay}, "
        f"KeySwapCost={constants.KeySwapCost}"
    )

    def assert_coldkey_reaped(coldkey_ss58: str, label: str) -> None:
        balance_after = subtensor.wallets.get_balance(coldkey_ss58)
        assert balance_after <= existential_deposit, (
            f"{label} balance after swap ({balance_after}) should be <= "
            f"ED ({existential_deposit})"
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
    dispute_after_swap = subtensor.wallets.get_coldkey_swap_dispute(
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address
    )
    assert dispute_after_swap is None, "Dispute should not exist after swap"
    assert_coldkey_reaped(alice_wallet.coldkeypub.ss58_address, "Alice")

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
    assert_coldkey_reaped(alice_wallet.coldkeypub.ss58_address, "Alice")
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
    assert_coldkey_reaped(alice_wallet.coldkeypub.ss58_address, "Alice")
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
    from bittensor_wallet import Keypair

    from bittensor.core.extrinsics.utils import verify_coldkey_hash

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
    assert_coldkey_reaped(alice_wallet.coldkeypub.ss58_address, "Alice")

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
    assert_coldkey_reaped(alice_wallet.coldkeypub.ss58_address, "Alice")

    # Ensure Alice has enough balance to pay root transaction fees
    root_min_balance = Balance.from_tao(5)
    alice_balance = subtensor.wallets.get_balance(alice_wallet.coldkeypub.ss58_address)
    if alice_balance < root_min_balance:
        top_up_amount = Balance.from_tao(10)
        logging.console.info(
            f"Top up Alice for root operations by {top_up_amount} "
            f"(current {alice_balance})"
        )
        response = subtensor.extrinsics.transfer(
            wallet=bob_wallet,
            destination_ss58=alice_wallet.coldkeypub.ss58_address,
            amount=top_up_amount,
        )
        assert response.success, (
            f"Failed to fund Alice for root ops: {response.message}"
        )

    # Ensure Dave has enough balance for dispute flow
    dave_min_balance = swap_cost + Balance.from_tao(10)
    dave_balance = subtensor.wallets.get_balance(dave_wallet.coldkeypub.ss58_address)
    if dave_balance < dave_min_balance:
        top_up_amount = dave_min_balance - dave_balance
        logging.console.info(
            f"Top up Dave for dispute flow by {top_up_amount} (current {dave_balance})"
        )
        response = subtensor.extrinsics.transfer(
            wallet=bob_wallet,
            destination_ss58=dave_wallet.coldkeypub.ss58_address,
            amount=top_up_amount,
        )
        assert response.success, f"Failed to fund Dave: {response.message}"

    # === 5. Dispute and root reset ===
    logging.console.info("Testing dispute and root reset")

    # Step 1: Dave announces swap to Charlie
    logging.console.info("Step 1: Dave announces swap to Charlie")
    existing_announcement = subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=dave_wallet.coldkeypub.ss58_address
    )
    existing_dispute = subtensor.wallets.get_coldkey_swap_dispute(
        coldkey_ss58=dave_wallet.coldkeypub.ss58_address
    )
    assert existing_announcement is None, (
        "No announcement should exist before dispute test"
    )
    assert existing_dispute is None, "No dispute should exist before dispute test"
    response = subtensor.extrinsics.announce_coldkey_swap(
        wallet=dave_wallet,
        new_coldkey_ss58=charlie_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to announce swap: {response.message}"

    # Step 2: Dave disputes the swap
    logging.console.info("Step 2: Dave disputes the swap")
    response = subtensor.extrinsics.dispute_coldkey_swap(wallet=dave_wallet)
    assert response.success, f"Failed to dispute swap: {response.message}"

    # Step 3: Verify dispute is recorded
    logging.console.info("Step 3: Verify dispute is recorded")
    dispute = subtensor.wallets.get_coldkey_swap_dispute(
        coldkey_ss58=dave_wallet.coldkeypub.ss58_address
    )
    assert dispute is not None, "Dispute should exist"
    assert dispute.coldkey == dave_wallet.coldkeypub.ss58_address
    logging.console.info(f"Dispute recorded at block {dispute.disputed_block}")

    # Step 4: Verify account is blocked (transfer from Dave should fail)
    logging.console.info("Step 4: Verify account is blocked")
    response = subtensor.extrinsics.transfer(
        wallet=dave_wallet,
        destination_ss58=charlie_wallet.coldkeypub.ss58_address,
        amount=Balance.from_tao(1),
        raise_error=False,
    )
    assert not response.success, "Transfer should be blocked while disputed"
    logging.console.info("Account blocking verified")

    # Step 5: Root resets the coldkey swap (alice_wallet is //Alice, root)
    logging.console.info("Step 5: Root resets the coldkey swap")
    response = reset_coldkey_swap_extrinsic(
        subtensor=subtensor.inner_subtensor,
        wallet=alice_wallet,
        coldkey_ss58=dave_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to reset coldkey swap: {response.message}"

    # Step 6: Verify dispute and announcement are cleared
    logging.console.info("Step 6: Verify dispute and announcement are cleared")
    dispute_after = subtensor.wallets.get_coldkey_swap_dispute(
        coldkey_ss58=dave_wallet.coldkeypub.ss58_address
    )
    announcement_after_reset = subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=dave_wallet.coldkeypub.ss58_address
    )
    assert dispute_after is None, "Dispute should be cleared after reset"
    assert announcement_after_reset is None, (
        "Announcement should be cleared after reset"
    )

    # Step 7: Verify transfers work again after reset
    logging.console.info("Step 7: Verify transfers are unblocked after reset")
    response = subtensor.extrinsics.transfer(
        wallet=dave_wallet,
        destination_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=Balance.from_tao(1),
        raise_error=False,
    )
    assert response.success, "Transfer should be allowed after reset"
    logging.console.info("Dispute scenario completed successfully")

    # === 6. Root swap override ===
    logging.console.info("Testing root swap override")

    # Ensure Dave has enough balance for root swap cost
    dave_min_balance = Balance.from_rao(swap_cost_rao) + Balance.from_tao(1)
    dave_balance = subtensor.wallets.get_balance(dave_wallet.coldkeypub.ss58_address)
    if dave_balance < dave_min_balance:
        top_up_amount = dave_min_balance - dave_balance
        logging.console.info(
            f"Top up Dave for root swap by {top_up_amount} (current {dave_balance})"
        )
        response = subtensor.extrinsics.transfer(
            wallet=bob_wallet,
            destination_ss58=dave_wallet.coldkeypub.ss58_address,
            amount=top_up_amount,
        )
        assert response.success, f"Failed to fund Dave: {response.message}"

    response = swap_coldkey_extrinsic(
        subtensor=subtensor.inner_subtensor,
        wallet=alice_wallet,
        old_coldkey_ss58=dave_wallet.coldkeypub.ss58_address,
        new_coldkey_ss58=charlie_wallet.coldkeypub.ss58_address,
        swap_cost=swap_cost_rao,
    )
    assert response.success, f"Failed to swap coldkey via root: {response.message}"

    announcement_after_root_swap = subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=dave_wallet.coldkeypub.ss58_address
    )
    dispute_after_root_swap = subtensor.wallets.get_coldkey_swap_dispute(
        coldkey_ss58=dave_wallet.coldkeypub.ss58_address
    )
    assert announcement_after_root_swap is None, (
        "Announcement should be cleared after root swap"
    )
    assert dispute_after_root_swap is None, "Dispute should be cleared after root swap"
    assert_coldkey_reaped(dave_wallet.coldkeypub.ss58_address, "Dave")

    logging.console.info("All coldkey swap E2E tests completed successfully")


@pytest.mark.asyncio
async def test_coldkey_swap_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet, dave_wallet
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

    5. Dispute and root reset:
       - Step 1: Dave announces swap, then disputes it (dispute_coldkey_swap)
       - Step 2: Verify dispute is recorded (get_coldkey_swap_dispute)
       - Step 3: Verify account is blocked (transfer fails)
       - Step 4: Root resets coldkey swap (reset_coldkey_swap)
       - Step 5: Verify dispute and announcement are cleared
       - Step 6: Verify transfers are unblocked after reset

    6. Root swap override:
       - Step 1: Root swaps Dave to Charlie without announcement
       - Step 2: Verify announcement and dispute are cleared
       - Step 3: Verify old coldkey is reaped

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
    swap_cost = Balance.from_rao(constants.KeySwapCost)
    swap_cost_rao = int(constants.KeySwapCost)
    existential_deposit = await async_subtensor.chain.get_existential_deposit()
    logging.console.info(
        f"Constants: AnnouncementDelay={announcement_delay}, "
        f"ReannouncementDelay={reannouncement_delay}, "
        f"KeySwapCost={constants.KeySwapCost}"
    )

    async def assert_coldkey_reaped(coldkey_ss58: str, label: str) -> None:
        balance_after = await async_subtensor.wallets.get_balance(coldkey_ss58)
        assert balance_after <= existential_deposit, (
            f"{label} balance after swap ({balance_after}) should be <= "
            f"ED ({existential_deposit})"
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
    dispute_after_swap = await async_subtensor.wallets.get_coldkey_swap_dispute(
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address
    )
    assert dispute_after_swap is None, "Dispute should not exist after swap"
    await assert_coldkey_reaped(alice_wallet.coldkeypub.ss58_address, "Alice")

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
    await assert_coldkey_reaped(alice_wallet.coldkeypub.ss58_address, "Alice")
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
    await assert_coldkey_reaped(alice_wallet.coldkeypub.ss58_address, "Alice")
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
    from bittensor_wallet import Keypair

    from bittensor.core.extrinsics.utils import verify_coldkey_hash

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
    await assert_coldkey_reaped(alice_wallet.coldkeypub.ss58_address, "Alice")

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
    await assert_coldkey_reaped(alice_wallet.coldkeypub.ss58_address, "Alice")

    # Ensure Alice has enough balance to pay root transaction fees
    root_min_balance = Balance.from_tao(5)
    alice_balance = await async_subtensor.wallets.get_balance(
        alice_wallet.coldkeypub.ss58_address
    )
    if alice_balance < root_min_balance:
        top_up_amount = Balance.from_tao(10)
        logging.console.info(
            f"Top up Alice for root operations by {top_up_amount} "
            f"(current {alice_balance})"
        )
        response = await async_subtensor.extrinsics.transfer(
            wallet=bob_wallet,
            destination_ss58=alice_wallet.coldkeypub.ss58_address,
            amount=top_up_amount,
        )
        assert response.success, (
            f"Failed to fund Alice for root ops: {response.message}"
        )

    # Ensure Dave has enough balance for dispute flow
    dave_min_balance = swap_cost + Balance.from_tao(10)
    dave_balance = await async_subtensor.wallets.get_balance(
        dave_wallet.coldkeypub.ss58_address
    )
    if dave_balance < dave_min_balance:
        top_up_amount = dave_min_balance - dave_balance
        logging.console.info(
            f"Top up Dave for dispute flow by {top_up_amount} (current {dave_balance})"
        )
        response = await async_subtensor.extrinsics.transfer(
            wallet=bob_wallet,
            destination_ss58=dave_wallet.coldkeypub.ss58_address,
            amount=top_up_amount,
        )
        assert response.success, f"Failed to fund Dave: {response.message}"

    # === 5. Dispute and root reset ===
    logging.console.info("Testing dispute and root reset")

    # Step 1: Dave announces swap to Charlie
    logging.console.info("Step 1: Dave announces swap to Charlie")
    existing_announcement = await async_subtensor.wallets.get_coldkey_swap_announcement(
        coldkey_ss58=dave_wallet.coldkeypub.ss58_address
    )
    existing_dispute = await async_subtensor.wallets.get_coldkey_swap_dispute(
        coldkey_ss58=dave_wallet.coldkeypub.ss58_address
    )
    assert existing_announcement is None, (
        "No announcement should exist before dispute test"
    )
    assert existing_dispute is None, "No dispute should exist before dispute test"
    response = await async_subtensor.extrinsics.announce_coldkey_swap(
        wallet=dave_wallet,
        new_coldkey_ss58=charlie_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to announce swap: {response.message}"

    # Step 2: Dave disputes the swap
    logging.console.info("Step 2: Dave disputes the swap")
    response = await async_subtensor.extrinsics.dispute_coldkey_swap(wallet=dave_wallet)
    assert response.success, f"Failed to dispute swap: {response.message}"

    # Step 3: Verify dispute is recorded
    logging.console.info("Step 3: Verify dispute is recorded")
    dispute = await async_subtensor.wallets.get_coldkey_swap_dispute(
        coldkey_ss58=dave_wallet.coldkeypub.ss58_address
    )
    assert dispute is not None, "Dispute should exist"
    assert dispute.coldkey == dave_wallet.coldkeypub.ss58_address
    logging.console.info(f"Dispute recorded at block {dispute.disputed_block}")

    # Step 4: Verify account is blocked (transfer from Dave should fail)
    logging.console.info("Step 4: Verify account is blocked")
    response = await async_subtensor.extrinsics.transfer(
        wallet=dave_wallet,
        destination_ss58=charlie_wallet.coldkeypub.ss58_address,
        amount=Balance.from_tao(1),
        raise_error=False,
    )
    assert not response.success, "Transfer should be blocked while disputed"
    logging.console.info("Account blocking verified")

    # Step 5: Root resets the coldkey swap (alice_wallet is //Alice, root)
    logging.console.info("Step 5: Root resets the coldkey swap")
    response = await async_reset_coldkey_swap_extrinsic(
        subtensor=async_subtensor.inner_subtensor,
        wallet=alice_wallet,
        coldkey_ss58=dave_wallet.coldkeypub.ss58_address,
    )
    assert response.success, f"Failed to reset coldkey swap: {response.message}"

    # Step 6: Verify dispute and announcement are cleared
    logging.console.info("Step 6: Verify dispute and announcement are cleared")
    dispute_after = await async_subtensor.wallets.get_coldkey_swap_dispute(
        coldkey_ss58=dave_wallet.coldkeypub.ss58_address
    )
    announcement_after_reset = (
        await async_subtensor.wallets.get_coldkey_swap_announcement(
            coldkey_ss58=dave_wallet.coldkeypub.ss58_address
        )
    )
    assert dispute_after is None, "Dispute should be cleared after reset"
    assert announcement_after_reset is None, (
        "Announcement should be cleared after reset"
    )

    # Step 7: Verify transfers work again after reset
    logging.console.info("Step 7: Verify transfers are unblocked after reset")
    response = await async_subtensor.extrinsics.transfer(
        wallet=dave_wallet,
        destination_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=Balance.from_tao(1),
        raise_error=False,
    )
    assert response.success, "Transfer should be allowed after reset"
    logging.console.info("Dispute scenario completed successfully")

    # === 6. Root swap override ===
    logging.console.info("Testing root swap override")

    # Ensure Dave has enough balance for root swap cost
    dave_min_balance = Balance.from_rao(swap_cost_rao) + Balance.from_tao(1)
    dave_balance = await async_subtensor.wallets.get_balance(
        dave_wallet.coldkeypub.ss58_address
    )
    if dave_balance < dave_min_balance:
        top_up_amount = dave_min_balance - dave_balance
        logging.console.info(
            f"Top up Dave for root swap by {top_up_amount} (current {dave_balance})"
        )
        response = await async_subtensor.extrinsics.transfer(
            wallet=bob_wallet,
            destination_ss58=dave_wallet.coldkeypub.ss58_address,
            amount=top_up_amount,
        )
        assert response.success, f"Failed to fund Dave: {response.message}"

    response = await async_swap_coldkey_extrinsic(
        subtensor=async_subtensor.inner_subtensor,
        wallet=alice_wallet,
        old_coldkey_ss58=dave_wallet.coldkeypub.ss58_address,
        new_coldkey_ss58=charlie_wallet.coldkeypub.ss58_address,
        swap_cost=swap_cost_rao,
    )
    assert response.success, f"Failed to swap coldkey via root: {response.message}"

    announcement_after_root_swap = (
        await async_subtensor.wallets.get_coldkey_swap_announcement(
            coldkey_ss58=dave_wallet.coldkeypub.ss58_address
        )
    )
    dispute_after_root_swap = await async_subtensor.wallets.get_coldkey_swap_dispute(
        coldkey_ss58=dave_wallet.coldkeypub.ss58_address
    )
    assert announcement_after_root_swap is None, (
        "Announcement should be cleared after root swap"
    )
    assert dispute_after_root_swap is None, "Dispute should be cleared after root swap"
    await assert_coldkey_reaped(dave_wallet.coldkeypub.ss58_address, "Dave")

    logging.console.info("All coldkey swap E2E tests completed successfully")
