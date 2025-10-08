import asyncio
import time

import numpy as np
import pytest

from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from tests.e2e_tests.utils import (
    TestSubnet,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
    SUDO_SET_ADMIN_FREEZE_WINDOW,
    SUDO_SET_TEMPO,
    SUDO_SET_MECHANISM_COUNT,
    SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED,
    SUDO_SET_WEIGHTS_SET_RATE_LIMIT,
    NETUID,
    AdminUtils,
)

TESTED_MECHANISMS = 2
EXPECTED_CR_VERSION = 4


def test_commit_and_reveal_weights_cr4(subtensor, alice_wallet):
    """
    Tests the commit/reveal weights mechanism (CRv4)

    Steps:
        1. Register a subnet through Alice
        2. Register Alice's neuron and add stake
        3. Enable a commit-reveal mechanism on subnet
        4. Lower weights rate limit
        5. Change the tempo for subnet 1
        5. Commit weights and ensure they are committed.
        6. Wait interval & reveal weights and verify
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    # 12 for non-fast-block, 0.25 for fast block
    BLOCK_TIME, TEMPO_TO_SET = (
        (0.25, 100) if subtensor.chain.is_fast_blocks() else (12.0, 20)
    )
    logging.console.info(f"Using block time: {BLOCK_TIME}")

    # Create and prepare subnet
    alice_sn = TestSubnet(subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(alice_wallet),
        SUDO_SET_MECHANISM_COUNT(
            alice_wallet, AdminUtils, True, NETUID, TESTED_MECHANISMS
        ),
        ACTIVATE_SUBNET(alice_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
        SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
            alice_wallet, AdminUtils, True, NETUID, True
        ),
        SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, 0),
    ]
    alice_sn.execute_steps(steps)

    cr_version = subtensor.substrate.query(
        module="SubtensorModule", storage_function="CommitRevealWeightsVersion"
    )
    assert cr_version == EXPECTED_CR_VERSION, (
        f"Commit reveal version is not {EXPECTED_CR_VERSION}, got {cr_version}"
    )

    # Verify weights rate limit was changed
    assert subtensor.subnets.weights_rate_limit(netuid=alice_sn.netuid) == 0
    logging.console.success("sudo_set_weights_set_rate_limit executed: set to 0")

    tempo = subtensor.subnets.get_subnet_hyperparameters(netuid=alice_sn.netuid).tempo
    assert tempo == TEMPO_TO_SET, "SN tempos has not been changed."

    # Commit-reveal values - setting weights to self
    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.1], dtype=np.float32)

    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )
    logging.console.info(
        f"Committing weights: uids {weight_uids}, weights {weight_vals}"
    )

    # Fetch current block and calculate next tempo for the subnet
    current_block = subtensor.chain.get_current_block()
    upcoming_tempo = subtensor.subnets.get_next_epoch_start_block(alice_sn.netuid)
    logging.console.info(
        f"Checking if window is too low with current block: {current_block}, next tempo: {upcoming_tempo}"
    )

    # Lower than this might mean weights will get revealed before we can check them
    if upcoming_tempo - current_block < 6:
        alice_sn.wait_next_epoch()
    current_block = subtensor.chain.get_current_block()
    latest_drand_round = subtensor.chain.last_drand_round()
    upcoming_tempo = subtensor.subnets.get_next_epoch_start_block(alice_sn.netuid)
    logging.console.info(
        f"Post first wait_interval (to ensure window isn't too low): {current_block}, next tempo: {upcoming_tempo}, drand: {latest_drand_round}"
    )

    for mechid in range(TESTED_MECHANISMS):
        logging.console.info(
            f"[magenta]Testing subnet mechanism: {alice_sn.netuid}.{mechid}[/magenta]"
        )

        # commit_block is the block when weights were committed on the chain (transaction block)
        expected_commit_block = subtensor.block + 1
        # Commit weights
        response = subtensor.extrinsics.set_weights(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            mechid=mechid,
            uids=weight_uids,
            weights=weight_vals,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            block_time=BLOCK_TIME,
            period=16,
        )

        # Assert committing was a success
        assert response.success is True, response.message
        assert response.data.get("reveal_round") is not None

        # Parse expected reveal_round
        expected_reveal_round = response.data.get("reveal_round")
        logging.console.success(
            f"Successfully set weights: uids {weight_uids}, weights {weight_vals}, reveal_round: {expected_reveal_round}"
        )

        # let chain to update
        subtensor.wait_for_block(subtensor.block + 1)

        # Fetch current commits pending on the chain
        commits_on_chain = subtensor.commitments.get_timelocked_weight_commits(
            netuid=alice_sn.netuid, mechid=mechid
        )
        address, commit_block, commit, reveal_round = commits_on_chain[0]

        # Assert correct values are committed on the chain
        assert expected_reveal_round == reveal_round
        assert address == alice_wallet.hotkey.ss58_address

        # bc of the drand delay, the commit block can be either the previous block or the current block
        assert expected_commit_block in [
            commit_block - 1,
            commit_block,
            commit_block + 1,
        ]

        # Ensure no weights are available as of now
        assert subtensor.subnets.weights(netuid=alice_sn.netuid, mechid=mechid) == []
        logging.console.success("No weights are available before next epoch.")

        # 5 is safety drand offset
        expected_reveal_block = (
            subtensor.subnets.get_next_epoch_start_block(alice_sn.netuid) + 5
        )

        logging.console.info(
            f"Waiting for the next epoch to ensure weights are revealed: block {expected_reveal_block}"
        )
        subtensor.wait_for_block(expected_reveal_block)

        # Fetch the latest drand pulse
        latest_drand_round = 0

        while latest_drand_round <= expected_reveal_round:
            latest_drand_round = subtensor.chain.last_drand_round()
            logging.console.info(
                f"Latest drand round: {latest_drand_round}, waiting for next round..."
            )
            # drand round is 2
            time.sleep(3)

        # Fetch weights on the chain as they should be revealed now
        subnet_weights = subtensor.subnets.weights(
            netuid=alice_sn.netuid, mechid=mechid
        )
        assert subnet_weights != [], "Weights are not available yet."

        logging.console.info(f"Revealed weights: {subnet_weights}")

        revealed_weights = subnet_weights[0][1]
        # Assert correct weights were revealed
        assert weight_uids[0] == revealed_weights[0][0]
        assert weight_vals[0] == revealed_weights[0][1]

        logging.console.success(
            f"Successfully revealed weights: uids {weight_uids}, weights {weight_vals}"
        )

        # Now that the commit has been revealed, there shouldn't be any pending commits
        assert (
            subtensor.commitments.get_timelocked_weight_commits(
                netuid=alice_sn.netuid, mechid=mechid
            )
            == []
        )

        # Ensure the drand_round is always in the positive w.r.t expected when revealed
        assert latest_drand_round - expected_reveal_round >= -3, (
            f"latest_drand_round ({latest_drand_round}) is less than expected_reveal_round ({expected_reveal_round})"
        )


@pytest.mark.asyncio
async def test_commit_and_reveal_weights_cr4_async(async_subtensor, alice_wallet):
    """
    Tests the commit/reveal weights mechanism (CRv4)

    Steps:
        1. Register a subnet through Alice
        2. Register Alice's neuron and add stake
        3. Enable a commit-reveal mechanism on subnet
        4. Lower weights rate limit
        5. Change the tempo for subnet 1
        5. Commit weights and ensure they are committed.
        6. Wait interval & reveal weights and verify
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    # 12 for non-fast-block, 0.25 for fast block
    BLOCK_TIME, TEMPO_TO_SET = (
        (0.25, 100) if await async_subtensor.chain.is_fast_blocks() else (12.0, 20)
    )
    logging.console.info(f"Using block time: {BLOCK_TIME}")

    # Create and prepare subnet
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(alice_wallet),
        SUDO_SET_MECHANISM_COUNT(
            alice_wallet, AdminUtils, True, NETUID, TESTED_MECHANISMS
        ),
        ACTIVATE_SUBNET(alice_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
        SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
            alice_wallet, AdminUtils, True, NETUID, True
        ),
        SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, 0),
    ]
    await alice_sn.async_execute_steps(steps)

    cr_version = await async_subtensor.substrate.query(
        module="SubtensorModule", storage_function="CommitRevealWeightsVersion"
    )
    assert cr_version == EXPECTED_CR_VERSION, (
        f"Commit reveal version is not {EXPECTED_CR_VERSION}, got {cr_version}"
    )

    # Verify weights rate limit was changed
    assert await async_subtensor.subnets.weights_rate_limit(netuid=alice_sn.netuid) == 0
    logging.console.success("sudo_set_weights_set_rate_limit executed: set to 0")

    tempo = (
        await async_subtensor.subnets.get_subnet_hyperparameters(netuid=alice_sn.netuid)
    ).tempo
    assert tempo == TEMPO_TO_SET, "SN tempos has not been changed."

    # Commit-reveal values - setting weights to self
    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.1], dtype=np.float32)

    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )
    logging.console.info(
        f"Committing weights: uids {weight_uids}, weights {weight_vals}"
    )

    # Fetch current block and calculate next tempo for the subnet
    current_block, upcoming_tempo = await asyncio.gather(
        async_subtensor.chain.get_current_block(),
        async_subtensor.subnets.get_next_epoch_start_block(alice_sn.netuid),
    )
    logging.console.info(
        f"Checking if window is too low with Current block: {current_block}, next tempo: {upcoming_tempo}"
    )

    # Lower than this might mean weights will get revealed before we can check them
    if upcoming_tempo - current_block < 6:
        await alice_sn.async_wait_next_epoch()

    current_block, latest_drand_round = await asyncio.gather(
        async_subtensor.chain.get_current_block(),
        async_subtensor.subnets.get_next_epoch_start_block(
            alice_sn.netuid
        )
    )
    logging.console.info(
        f"Post first wait_interval (to ensure window isn't too low): {current_block}, next tempo: {upcoming_tempo}, drand: {latest_drand_round}"
    )

    for mechid in range(TESTED_MECHANISMS):
        logging.console.info(
            f"[magenta]Testing subnet mechanism: {alice_sn.netuid}.{mechid}[/magenta]"
        )

        # Commit weights
        response = await async_subtensor.extrinsics.set_weights(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            mechid=mechid,
            uids=weight_uids,
            weights=weight_vals,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            block_time=BLOCK_TIME,
            period=16,
        )

        # Assert committing was a success
        assert response.success is True, response.message
        assert response.data.get("reveal_round") is not None

        # Parse expected reveal_round
        expected_reveal_round = response.data.get("reveal_round")
        logging.console.success(
            f"Successfully set weights: uids {weight_uids}, weights {weight_vals}, reveal_round: {expected_reveal_round}"
        )

        # Fetch current commits pending on the chain
        await async_subtensor.wait_for_block(await async_subtensor.block + 1)

        commits_on_chain = (
            await async_subtensor.commitments.get_timelocked_weight_commits(
                netuid=alice_sn.netuid, mechid=mechid
            )
        )
        # commits_on_chain = []
        # counter = TEMPO_TO_SET
        # while commits_on_chain == []:
        #     counter -= 1
        #     if counter == 0:
        #         break
        #     commits_on_chain = (
        #         await async_subtensor.commitments.get_timelocked_weight_commits(
        #             netuid=alice_sn.netuid, mechid=mechid
        #         )
        #     )
        #     logging.console.info(
        #         f"block: {await async_subtensor.block}, commits: {commits_on_chain}, waiting for next round..."
        #     )
        #     await async_subtensor.wait_for_block()
        #
        # logging.console.info(
        #     f"block: {await async_subtensor.block}, commits: {commits_on_chain}, waiting for next round..."
        # )
        address, commit_block, commit, reveal_round = commits_on_chain[0]

        # Assert correct values are committed on the chain
        assert expected_reveal_round == reveal_round
        assert address == alice_wallet.hotkey.ss58_address

        # Ensure no weights are available as of now
        assert (
            await async_subtensor.subnets.weights(netuid=alice_sn.netuid, mechid=mechid)
            == []
        )
        logging.console.success("No weights are available before next epoch.")

        # 5 is safety drand offset
        expected_reveal_block = (
            await async_subtensor.subnets.get_next_epoch_start_block(alice_sn.netuid)
            + 5
        )

        logging.console.info(
            f"Waiting for the next epoch to ensure weights are revealed: block {expected_reveal_block}"
        )
        await async_subtensor.wait_for_block(expected_reveal_block)

        # Fetch the latest drand pulse
        latest_drand_round = 0

        while latest_drand_round <= expected_reveal_round:
            latest_drand_round = await async_subtensor.chain.last_drand_round()
            logging.console.info(
                f"Latest drand round: {latest_drand_round}, waiting for next round..."
            )
            # drand round is 2
            await asyncio.sleep(3)

        # Fetch weights on the chain as they should be revealed now
        subnet_weights = await async_subtensor.subnets.weights(
            netuid=alice_sn.netuid, mechid=mechid
        )
        assert subnet_weights != [], "Weights are not available yet."

        logging.console.info(f"Revealed weights: {subnet_weights}")

        revealed_weights = subnet_weights[0][1]
        # Assert correct weights were revealed
        assert weight_uids[0] == revealed_weights[0][0]
        assert weight_vals[0] == revealed_weights[0][1]

        logging.console.success(
            f"Successfully revealed weights: uids {weight_uids}, weights {weight_vals}"
        )

        # Now that the commit has been revealed, there shouldn't be any pending commits
        assert (
            await async_subtensor.commitments.get_timelocked_weight_commits(
                netuid=alice_sn.netuid, mechid=mechid
            )
            == []
        )

        # Ensure the drand_round is always in the positive w.r.t expected when revealed
        assert latest_drand_round - expected_reveal_round >= -3, (
            f"latest_drand_round ({latest_drand_round}) is less than expected_reveal_round ({expected_reveal_round})"
        )
