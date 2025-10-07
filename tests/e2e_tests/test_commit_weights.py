import time

import numpy as np
import pytest
import retry

from bittensor.utils import get_mechid_storage_index
from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from tests.e2e_tests.utils import (
    execute_and_wait_for_next_nonce,
    TestSubnet,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
    NETUID,
    SUDO_SET_ADMIN_FREEZE_WINDOW,
    SUDO_SET_TEMPO,
    SUDO_SET_MECHANISM_COUNT,
    SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED,
    SUDO_SET_WEIGHTS_SET_RATE_LIMIT,
    AdminUtils,
)

TESTED_SUB_SUBNETS = 2


def test_commit_and_reveal_weights_legacy(subtensor, alice_wallet):
    """
    Tests the commit/reveal weights mechanism with subprocess disabled (CR1.0)

    Steps:
        1. Register a subnet through Alice
        2. Enable the commit-reveal mechanism on subnet
        3. Lower the commit_reveal interval and rate limit
        4. Commit weights and verify
        5. Wait interval & reveal weights and verify
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    TEMPO_TO_SET = 50 if subtensor.chain.is_fast_blocks() else 20

    # Create and prepare subnet
    alice_sn = TestSubnet(subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(alice_wallet),
        SUDO_SET_MECHANISM_COUNT(
            alice_wallet, AdminUtils, True, NETUID, TESTED_SUB_SUBNETS
        ),
        ACTIVATE_SUBNET(alice_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
        SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
            alice_wallet, AdminUtils, True, NETUID, True
        ),
    ]
    alice_sn.execute_steps(steps)

    assert subtensor.subnets.commit_reveal_enabled(alice_sn.netuid), (
        "Failed to enable commit/reveal"
    )

    assert (
        subtensor.subnets.get_subnet_hyperparameters(
            alice_sn.netuid
        ).commit_reveal_period
        == 1
    ), "Failed to set commit/reveal periods"

    assert subtensor.subnets.weights_rate_limit(alice_sn.netuid) > 0, (
        "Weights rate limit is below 0"
    )

    # set weights rate limit
    response = alice_sn.execute_one(
        SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, 0)
    )
    assert response.success, response.message
    assert subtensor.subnets.weights_rate_limit(netuid=alice_sn.netuid) == 0

    for mechid in range(TESTED_SUB_SUBNETS):
        logging.console.info(
            f"[magenta]Testing subnet mechanism {alice_sn.netuid}.{mechid}[/magenta]"
        )

        # Commit-reveal values
        uids = np.array([0], dtype=np.int64)
        weights = np.array([0.1], dtype=np.float32)
        salt = [18, 179, 107, 0, 165, 211, 141, 197]
        weight_uids, weight_vals = convert_weights_and_uids_for_emit(
            uids=uids, weights=weights
        )

        # Commit weights
        response = subtensor.extrinsics.commit_weights(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            mechid=mechid,
            salt=salt,
            uids=weight_uids,
            weights=weight_vals,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )

        assert response.success, response.message

        storage_index = get_mechid_storage_index(alice_sn.netuid, mechid)
        weight_commits = subtensor.queries.query_module(
            module="SubtensorModule",
            name="WeightCommits",
            params=[storage_index, alice_wallet.hotkey.ss58_address],
        )
        logging.console.info(f"weight_commits: {weight_commits}")

        # Assert that the committed weights are set correctly
        assert weight_commits is not None, "Weight commit not found in storage"
        commit_hash, commit_block, reveal_block, expire_block = weight_commits[0]
        assert commit_block > 0, f"Invalid block number: {commit_block}"

        # Query the WeightCommitRevealInterval storage map
        assert subtensor.subnets.get_subnet_reveal_period_epochs(alice_sn.netuid) > 0, (
            "Invalid RevealPeriodEpochs"
        )

        # Wait until the reveal block range
        subtensor.wait_for_block(
            subtensor.subnets.get_next_epoch_start_block(alice_sn.netuid) + 1
        )

        # Reveal weights
        success, message = subtensor.extrinsics.reveal_weights(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            mechid=mechid,
            uids=weight_uids,
            weights=weight_vals,
            salt=salt,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )

        assert success is True, message

        revealed_weights = subtensor.subnets.weights(alice_sn.netuid, mechid=mechid)

        # Assert that the revealed weights are set correctly
        assert revealed_weights is not None, "Weight reveal not found in storage"

        alice_weights = revealed_weights[0][1]
        assert weight_vals[0] == alice_weights[0][1], (
            f"Incorrect revealed weights. Expected: {weights[0]}, Actual: {revealed_weights[0][1]}"
        )


@pytest.mark.asyncio
async def test_commit_and_reveal_weights_legacy_async(async_subtensor, alice_wallet):
    """
    Tests the commit/reveal weights mechanism with subprocess disabled (CR1.0) with AsyncSubtensor.

    Steps:
        1. Register a subnet through Alice
        2. Enable the commit-reveal mechanism on subnet
        3. Lower the commit_reveal interval and rate limit
        4. Commit weights and verify
        5. Wait interval & reveal weights and verify
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    TEMPO_TO_SET = 50 if await async_subtensor.chain.is_fast_blocks() else 20

    # Create and prepare subnet
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(alice_wallet),
        SUDO_SET_MECHANISM_COUNT(
            alice_wallet, AdminUtils, True, NETUID, TESTED_SUB_SUBNETS
        ),
        ACTIVATE_SUBNET(alice_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
        SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
            alice_wallet, AdminUtils, True, NETUID, True
        ),
    ]
    await alice_sn.async_execute_steps(steps)

    assert await async_subtensor.subnets.commit_reveal_enabled(alice_sn.netuid), (
        "Failed to enable commit/reveal"
    )

    assert (
        await async_subtensor.subnets.get_subnet_hyperparameters(netuid=alice_sn.netuid)
    ).commit_reveal_period == 1, "Failed to set commit/reveal periods"

    assert (
        await async_subtensor.subnets.weights_rate_limit(netuid=alice_sn.netuid) > 0
    ), "Weights rate limit is below 0"

    # set weights rate limit
    response = await alice_sn.async_execute_one(
        SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, 0)
    )
    assert response.success, response.message
    assert await async_subtensor.subnets.weights_rate_limit(netuid=alice_sn.netuid) == 0

    # Commit-reveal values
    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.1], dtype=np.float32)
    salt = [18, 179, 107, 0, 165, 211, 141, 197]
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )

    # Commit weights
    success, message = await async_subtensor.extrinsics.commit_weights(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        salt=salt,
        uids=weight_uids,
        weights=weight_vals,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success is True

    weight_commits = await async_subtensor.queries.query_module(
        module="SubtensorModule",
        name="WeightCommits",
        params=[alice_sn.netuid, alice_wallet.hotkey.ss58_address],
    )
    logging.console.info(f"weight_commits: {weight_commits}")

    # Assert that the committed weights are set correctly
    assert weight_commits is not None, "Weight commit not found in storage"
    commit_hash, commit_block, reveal_block, expire_block = weight_commits[0]
    assert commit_block > 0, f"Invalid block number: {commit_block}"

    # Query the WeightCommitRevealInterval storage map
    assert (
        await async_subtensor.subnets.get_subnet_reveal_period_epochs(alice_sn.netuid)
        > 0
    ), "Invalid RevealPeriodEpochs"

    # Reveal weights
    success, message = await async_subtensor.extrinsics.reveal_weights(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        uids=weight_uids,
        weights=weight_vals,
        salt=salt,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert success is True, message

    # Query the Weights storage map
    revealed_weights = await async_subtensor.queries.query_module(
        module="SubtensorModule",
        name="Weights",
        params=[alice_sn.netuid, 0],  # netuid and uid
    )

    # Assert that the revealed weights are set correctly
    assert revealed_weights is not None, "Weight reveal not found in storage"

    assert weight_vals[0] == revealed_weights[0][1], (
        f"Incorrect revealed weights. Expected: {weights[0]}, Actual: {revealed_weights[0][1]}"
    )


# Create different committed data to avoid coming into the pool's blacklist with the error
#   Failed to commit weights: Subtensor returned `Custom type(1012)` error. This means: `Transaction is temporarily
#   banned`.Failed to commit weights: Subtensor returned `Custom type(1012)` error. This means: `Transaction is
#   temporarily banned`.`
def get_weights_and_salt(counter: int):
    # Commit-reveal values
    uids_ = np.array([0], dtype=np.int64)
    weights_ = np.array([(counter + 1) / 10], dtype=np.float32)
    weight_uids_, weight_vals_ = convert_weights_and_uids_for_emit(
        uids=uids_, weights=weights_
    )
    salt_ = [18, 179, 107, counter, 165, 211, 141, 197]
    return weight_uids_, weight_vals_, salt_


def test_commit_weights_uses_next_nonce(subtensor, alice_wallet):
    """
    Tests that committing weights doesn't re-use nonce in the transaction pool.

    Steps:
        1. Register a subnet through Alice
        2. Register Alice's neuron and add stake
        3. Enable the commit-reveal mechanism on subnet
        4. Lower the commit_reveal interval and rate limit
        5. Commit weights three times
        6. Assert that all commits succeeded
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    TEMPO_TO_SET = 100 if subtensor.chain.is_fast_blocks() else 20

    # Create and prepare subnet
    alice_sn = TestSubnet(subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
        SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
            alice_wallet, AdminUtils, True, NETUID, True
        ),
    ]
    alice_sn.execute_steps(steps)

    # Wait for 2 tempos to pass as CR3 only reveals weights after 2 tempos
    subtensor.wait_for_block(subtensor.block + (TEMPO_TO_SET * 2) + 1)

    assert subtensor.commitments.commit_reveal_enabled(alice_sn.netuid), (
        "Failed to enable commit/reveal."
    )

    assert (
        subtensor.subnets.get_subnet_hyperparameters(
            netuid=alice_sn.netuid
        ).commit_reveal_period
        == 1
    ), "Commit reveal period is not 1."

    assert subtensor.subnets.weights_rate_limit(netuid=alice_sn.netuid) > 0, (
        "Weights rate limit is below 0"
    )

    # set weights rate limit
    response = alice_sn.execute_one(
        SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, 0)
    )
    assert response.success, response.message
    assert subtensor.subnets.weights_rate_limit(netuid=alice_sn.netuid) == 0

    # wait while weights_rate_limit changes applied.
    subtensor.wait_for_block()

    logging.console.info(
        f"[orange]Nonce before first commit_weights: "
        f"{subtensor.substrate.get_account_next_index(alice_wallet.hotkey.ss58_address)}[/orange]"
    )

    # 3 time doing call if nonce wasn't updated, then raise the error
    @retry.retry(exceptions=Exception, tries=3, delay=1)
    @execute_and_wait_for_next_nonce(subtensor=subtensor, wallet=alice_wallet)
    def send_commit(salt_, weight_uids_, weight_vals_):
        success, message = subtensor.extrinsics.commit_weights(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            salt=salt_,
            uids=weight_uids_,
            weights=weight_vals_,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        assert success is True, message

    # Send some number of commit weights
    AMOUNT_OF_COMMIT_WEIGHTS = 3
    for call in range(AMOUNT_OF_COMMIT_WEIGHTS):
        weight_uids, weight_vals, salt = get_weights_and_salt(call)

        logging.console.info(
            f"Sending commit with uids: {weight_uids}, weight: {weight_vals}"
        )
        send_commit(salt, weight_uids, weight_vals)

        # let's wait for 3 (12 fast blocks) seconds between transactions, next block for non-fast-blocks
        waiting_block = (
            (subtensor.block + 12) if subtensor.chain.is_fast_blocks() else None
        )
        subtensor.wait_for_block(waiting_block)

    logging.console.info(
        f"[orange]Nonce after third commit_weights: "
        f"{subtensor.substrate.get_account_next_index(alice_wallet.hotkey.ss58_address)}[/orange]"
    )

    # Wait a few blocks
    waiting_block = (
        (subtensor.block + subtensor.subnets.tempo(alice_sn.netuid) * 2)
        if subtensor.chain.is_fast_blocks()
        else None
    )
    subtensor.wait_for_block(waiting_block)

    # Query the WeightCommits storage map for all three salts
    weight_commits = subtensor.queries.query_module(
        module="SubtensorModule",
        name="WeightCommits",
        params=[alice_sn.netuid, alice_wallet.hotkey.ss58_address],
    )
    # Assert that the committed weights are set correctly
    assert weight_commits.value is not None, "Weight commit not found in storage"
    commit_hash, commit_block, reveal_block, expire_block = weight_commits.value[0]
    assert commit_block > 0, f"Invalid block number: {commit_block}"

    # Check for three commits in the WeightCommits storage map
    assert len(weight_commits.value) == AMOUNT_OF_COMMIT_WEIGHTS, (
        "Expected exact list of weight commits"
    )


@pytest.mark.asyncio
async def test_commit_weights_uses_next_nonce_async(async_subtensor, alice_wallet):
    """
    Tests that committing weights doesn't re-use nonce in the transaction pool with AsyncSubtensor.

    Steps:
        1. Register a subnet through Alice
        2. Register Alice's neuron and add stake
        3. Enable the commit-reveal mechanism on subnet
        4. Lower the commit_reveal interval and rate limit
        5. Commit weights three times
        6. Assert that all commits succeeded
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    TEMPO_TO_SET = 100 if await async_subtensor.chain.is_fast_blocks() else 20

    # Create and prepare subnet
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
        SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
            alice_wallet, AdminUtils, True, NETUID, True
        ),
    ]
    await alice_sn.async_execute_steps(steps)

    # Wait for 2 tempos to pass as CR3 only reveals weights after 2 tempos
    await async_subtensor.wait_for_block(
        await async_subtensor.block + (TEMPO_TO_SET * 2) + 1
    )

    assert await async_subtensor.commitments.commit_reveal_enabled(alice_sn.netuid), (
        "Failed to enable commit/reveal."
    )

    assert (
        await async_subtensor.subnets.get_subnet_hyperparameters(netuid=alice_sn.netuid)
    ).commit_reveal_period == 1, "Failed to set commit/reveal periods"

    assert (
        await async_subtensor.subnets.weights_rate_limit(netuid=alice_sn.netuid) > 0
    ), "Weights rate limit is below 0"

    # set weights rate limit
    response = await alice_sn.async_execute_one(
        SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, 0)
    )
    assert response.success, response.message
    assert await async_subtensor.subnets.weights_rate_limit(netuid=alice_sn.netuid) == 0

    # wait while weights_rate_limit changes applied.
    await async_subtensor.wait_for_block(alice_sn.netuid + 1)

    logging.console.info(
        f"[orange]Nonce before first commit_weights: "
        f"{await async_subtensor.substrate.get_account_nonce(alice_wallet.hotkey.ss58_address)}[/orange]"
    )

    # 3 time doing call if nonce wasn't updated, then raise the error

    async def send_commit(salt_, weight_uids_, weight_vals_):
        """
        To avoid adding asynchronous retrieval to dependencies, we implement a retrieval behavior with asynchronous
        behavior.
        """

        async def send_commit_():
            success_, message_ = await async_subtensor.extrinsics.commit_weights(
                wallet=alice_wallet,
                netuid=alice_sn.netuid,
                salt=salt_,
                uids=weight_uids_,
                weights=weight_vals_,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            return success_, message_

        max_retries = 3
        timeout = 60.0
        sleep = 0.25 if async_subtensor.chain.is_fast_blocks() else 12.0

        for attempt in range(1, max_retries + 1):
            try:
                start_nonce = await async_subtensor.substrate.get_account_nonce(
                    alice_wallet.hotkey.ss58_address
                )

                result = await send_commit_()

                start = time.time()
                while (time.time() - start) < timeout:
                    current_nonce = await async_subtensor.substrate.get_account_nonce(
                        alice_wallet.hotkey.ss58_address
                    )

                    if current_nonce != start_nonce:
                        logging.console.info(
                            f"✅ Nonce changed from {start_nonce} to {current_nonce}"
                        )
                        return result
                    logging.console.info(
                        f"⏳ Waiting for nonce increment. Current: {current_nonce}"
                    )
                    time.sleep(sleep)
            except Exception as e:
                raise e
        raise Exception(f"Failed to commit weights after {max_retries} attempts.")

    # Send some number of commit weights
    AMOUNT_OF_COMMIT_WEIGHTS = 3
    for call in range(AMOUNT_OF_COMMIT_WEIGHTS):
        weight_uids, weight_vals, salt = get_weights_and_salt(call)

        logging.console.info(
            f"Sending commit with uids: {weight_uids}, weight: {weight_vals}"
        )
        await send_commit(salt, weight_uids, weight_vals)

        # let's wait for 3 (12 fast blocks) seconds between transactions, next block for non-fast-blocks
        waiting_block = (
            (await async_subtensor.block + 12)
            if await async_subtensor.chain.is_fast_blocks()
            else None
        )
        await async_subtensor.wait_for_block(waiting_block)

    logging.console.info(
        f"[orange]Nonce after third commit_weights: "
        f"{await async_subtensor.substrate.get_account_next_index(alice_wallet.hotkey.ss58_address)}[/orange]"
    )

    # Wait a few blocks
    waiting_block = (
        (
            await async_subtensor.block
            + await async_subtensor.subnets.tempo(alice_sn.netuid) * 2
        )
        + 15
        if await async_subtensor.chain.is_fast_blocks()
        else None
    )
    await async_subtensor.wait_for_block(waiting_block)

    weight_commits = None
    counter = 0
    # Query the WeightCommits storage map for all three salts
    while not weight_commits or not len(weight_commits) == AMOUNT_OF_COMMIT_WEIGHTS:
        weight_commits = await async_subtensor.queries.query_module(
            module="SubtensorModule",
            name="WeightCommits",
            params=[alice_sn.netuid, alice_wallet.hotkey.ss58_address],
        )
        await async_subtensor.wait_for_block()
        logging.console.info(f"len(weight_commits): {len(weight_commits)}")
        counter += 1
        if counter > TEMPO_TO_SET:
            break

    # Assert that the committed weights are set correctly
    assert weight_commits.value is not None, "Weight commit not found in storage"
    commit_hash, commit_block, reveal_block, expire_block = weight_commits.value[0]
    assert commit_block > 0, f"Invalid block number: {commit_block}"

    # Check for three commits in the WeightCommits storage map
    assert len(weight_commits.value) == AMOUNT_OF_COMMIT_WEIGHTS, (
        "Expected exact list of weight commits"
    )
