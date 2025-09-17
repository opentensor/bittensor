import asyncio
import re
import time

import numpy as np
import pytest

from bittensor.core.extrinsics.asyncex.sudo import (
    sudo_set_mechanism_count_extrinsic as async_sudo_set_mechanism_count_extrinsic,
    sudo_set_admin_freez_window_extrinsic as async_sudo_set_admin_freez_window_extrinsic,
)
from bittensor.core.extrinsics.sudo import (
    sudo_set_mechanism_count_extrinsic,
    sudo_set_admin_freez_window_extrinsic,
)
from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from tests.e2e_tests.utils.chain_interactions import (
    async_wait_interval,
    sudo_set_admin_utils,
    sudo_set_hyperparameter_bool,
    wait_interval,
    next_tempo,
)

TESTED_SUB_SUBNETS = 2


# @pytest.mark.parametrize("local_chain", [True], indirect=True)
@pytest.mark.asyncio
async def test_commit_and_reveal_weights_cr4(local_chain, subtensor, alice_wallet):
    """
    Tests the commit/reveal weights mechanism (CR3)

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
    logging.console.info("Testing `test_commit_and_reveal_weights_cr4`")

    # turn off admin freeze window limit for testing
    assert sudo_set_admin_freez_window_extrinsic(subtensor, alice_wallet, 0)

    # 12 for non-fast-block, 0.25 for fast block
    BLOCK_TIME, TEMPO_TO_SET = (
        (0.25, 100) if subtensor.chain.is_fast_blocks() else (12.0, 20)
    )

    logging.console.info(f"Using block time: {BLOCK_TIME}")

    alice_subnet_netuid = subtensor.subnets.get_total_subnets()  # 2

    # Register root as Alice
    assert subtensor.extrinsics.register_subnet(alice_wallet), (
        "Unable to register the subnet"
    )

    # Verify subnet 2 created successfully
    assert subtensor.subnet_exists(alice_subnet_netuid), (
        f"SN #{alice_subnet_netuid} wasn't created successfully"
    )

    assert sudo_set_mechanism_count_extrinsic(
        subtensor, alice_wallet, alice_subnet_netuid, TESTED_SUB_SUBNETS
    ), "Cannot create sub-subnets."

    logging.console.success(f"SN #{alice_subnet_netuid} is registered.")

    # Enable commit_reveal on the subnet
    assert sudo_set_hyperparameter_bool(
        substrate=local_chain,
        wallet=alice_wallet,
        call_function="sudo_set_commit_reveal_weights_enabled",
        value=True,
        netuid=alice_subnet_netuid,
    ), f"Unable to enable commit reveal on the SN #{alice_subnet_netuid}"

    # Verify commit_reveal was enabled
    assert subtensor.subnets.commit_reveal_enabled(alice_subnet_netuid), (
        "Failed to enable commit/reveal"
    )
    logging.console.success("Commit reveal enabled")

    cr_version = subtensor.substrate.query(
        module="SubtensorModule", storage_function="CommitRevealWeightsVersion"
    )
    assert cr_version == 4, f"Commit reveal version is not 3, got {cr_version}"

    # Change the weights rate limit on the subnet
    status, error = sudo_set_admin_utils(
        substrate=local_chain,
        wallet=alice_wallet,
        call_function="sudo_set_weights_set_rate_limit",
        call_params={"netuid": alice_subnet_netuid, "weights_set_rate_limit": "0"},
    )

    assert status is True
    assert error is None

    # Verify weights rate limit was changed
    assert (
        subtensor.subnets.get_subnet_hyperparameters(
            netuid=alice_subnet_netuid
        ).weights_rate_limit
        == 0
    ), "Failed to set weights_rate_limit"
    assert subtensor.weights_rate_limit(netuid=alice_subnet_netuid) == 0
    logging.console.success("sudo_set_weights_set_rate_limit executed: set to 0")

    # Change the tempo of the subnet
    assert (
        sudo_set_admin_utils(
            local_chain,
            alice_wallet,
            call_function="sudo_set_tempo",
            call_params={"netuid": alice_subnet_netuid, "tempo": TEMPO_TO_SET},
        )[0]
        is True
    )

    tempo = subtensor.subnets.get_subnet_hyperparameters(
        netuid=alice_subnet_netuid
    ).tempo
    assert tempo == TEMPO_TO_SET, "SN tempos has not been changed."
    logging.console.success(f"SN #{alice_subnet_netuid} tempo set to {TEMPO_TO_SET}")

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
    upcoming_tempo = next_tempo(current_block, tempo)
    logging.console.info(
        f"Checking if window is too low with Current block: {current_block}, next tempo: {upcoming_tempo}"
    )

    # Lower than this might mean weights will get revealed before we can check them
    if upcoming_tempo - current_block < 6:
        await wait_interval(
            tempo,
            subtensor,
            netuid=alice_subnet_netuid,
            reporting_interval=1,
        )
    current_block = subtensor.chain.get_current_block()
    latest_drand_round = subtensor.chain.last_drand_round()
    upcoming_tempo = next_tempo(current_block, tempo)
    logging.console.info(
        f"Post first wait_interval (to ensure window isn't too low): {current_block}, next tempo: {upcoming_tempo}, drand: {latest_drand_round}"
    )

    for mechid in range(TESTED_SUB_SUBNETS):
        logging.console.info(
            f"[magenta]Testing subnet mechanism: {alice_subnet_netuid}.{mechid}[/magenta]"
        )

        # commit_block is the block when weights were committed on the chain (transaction block)
        expected_commit_block = subtensor.block + 1
        # Commit weights
        success, message = subtensor.extrinsics.set_weights(
            wallet=alice_wallet,
            netuid=alice_subnet_netuid,
            mechid=mechid,
            uids=weight_uids,
            weights=weight_vals,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            block_time=BLOCK_TIME,
            period=16,
        )

        # Assert committing was a success
        assert success is True, message
        assert bool(re.match(r"reveal_round:\d+", message))

        # Parse expected reveal_round
        expected_reveal_round = int(message.split(":")[1])
        logging.console.success(
            f"Successfully set weights: uids {weight_uids}, weights {weight_vals}, reveal_round: {expected_reveal_round}"
        )

        # let chain to update
        subtensor.wait_for_block(subtensor.block + 1)

        # Fetch current commits pending on the chain
        commits_on_chain = subtensor.commitments.get_timelocked_weight_commits(
            netuid=alice_subnet_netuid, mechid=mechid
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
        assert subtensor.weights(netuid=alice_subnet_netuid, mechid=mechid) == []
        logging.console.success("No weights are available before next epoch.")

        # 5 is safety drand offset
        expected_reveal_block = (
            subtensor.subnets.get_next_epoch_start_block(alice_subnet_netuid) + 5
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
            netuid=alice_subnet_netuid, mechid=mechid
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
                netuid=alice_subnet_netuid, mechid=mechid
            )
            == []
        )

        # Ensure the drand_round is always in the positive w.r.t expected when revealed
        assert latest_drand_round - expected_reveal_round >= -3, (
            f"latest_drand_round ({latest_drand_round}) is less than expected_reveal_round ({expected_reveal_round})"
        )

    logging.console.success("✅ Passed `test_commit_and_reveal_weights_cr4`")


@pytest.mark.asyncio
async def test_async_commit_and_reveal_weights_cr4(
    local_chain, async_subtensor, alice_wallet
):
    """
    Tests the commit/reveal weights mechanism (CR3)

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

    logging.console.info("Testing `test_commit_and_reveal_weights_cr4`")

    # turn off admin freeze window limit for testing
    assert await async_sudo_set_admin_freez_window_extrinsic(
        async_subtensor, alice_wallet, 0
    )

    # 12 for non-fast-block, 0.25 for fast block
    BLOCK_TIME, TEMPO_TO_SET = (
        (0.25, 100) if (await async_subtensor.chain.is_fast_blocks()) else (12.0, 20)
    )

    logging.console.info(f"Using block time: {BLOCK_TIME}")

    alice_subnet_netuid = await async_subtensor.subnets.get_total_subnets()  # 2

    # Register root as Alice
    assert await async_subtensor.extrinsics.register_subnet(alice_wallet), (
        "Unable to register the subnet"
    )

    # Verify subnet 2 created successfully
    assert await async_subtensor.subnet_exists(alice_subnet_netuid), (
        f"SN #{alice_subnet_netuid} wasn't created successfully"
    )

    assert await async_sudo_set_mechanism_count_extrinsic(
        async_subtensor, alice_wallet, alice_subnet_netuid, TESTED_SUB_SUBNETS
    ), "Cannot create sub-subnets."

    logging.console.success(f"SN #{alice_subnet_netuid} is registered.")

    # Enable commit_reveal on the subnet
    assert sudo_set_hyperparameter_bool(
        substrate=local_chain,
        wallet=alice_wallet,
        call_function="sudo_set_commit_reveal_weights_enabled",
        value=True,
        netuid=alice_subnet_netuid,
    ), f"Unable to enable commit reveal on the SN #{alice_subnet_netuid}"

    # Verify commit_reveal was enabled
    assert await async_subtensor.subnets.commit_reveal_enabled(alice_subnet_netuid), (
        "Failed to enable commit/reveal"
    )
    logging.console.success("Commit reveal enabled")

    cr_version = await async_subtensor.substrate.query(
        module="SubtensorModule", storage_function="CommitRevealWeightsVersion"
    )
    assert cr_version == 4, f"Commit reveal version is not 3, got {cr_version}"

    # Change the weights rate limit on the subnet
    status, error = sudo_set_admin_utils(
        substrate=local_chain,
        wallet=alice_wallet,
        call_function="sudo_set_weights_set_rate_limit",
        call_params={"netuid": alice_subnet_netuid, "weights_set_rate_limit": "0"},
    )

    assert status is True
    assert error is None

    # Verify weights rate limit was changed
    assert (
        await async_subtensor.subnets.get_subnet_hyperparameters(
            netuid=alice_subnet_netuid
        )
    ).weights_rate_limit == 0, "Failed to set weights_rate_limit"
    assert await async_subtensor.weights_rate_limit(netuid=alice_subnet_netuid) == 0
    logging.console.success("sudo_set_weights_set_rate_limit executed: set to 0")

    # Change the tempo of the subnet
    assert (
        sudo_set_admin_utils(
            local_chain,
            alice_wallet,
            call_function="sudo_set_tempo",
            call_params={"netuid": alice_subnet_netuid, "tempo": TEMPO_TO_SET},
        )[0]
        is True
    )

    tempo = (
        await async_subtensor.subnets.get_subnet_hyperparameters(
            netuid=alice_subnet_netuid
        )
    ).tempo
    assert tempo == TEMPO_TO_SET, "SN tempos has not been changed."
    logging.console.success(f"SN #{alice_subnet_netuid} tempo set to {TEMPO_TO_SET}")

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
    current_block = await async_subtensor.chain.get_current_block()
    upcoming_tempo = next_tempo(current_block, tempo)
    logging.console.info(
        f"Checking if window is too low with Current block: {current_block}, next tempo: {upcoming_tempo}"
    )

    # Lower than this might mean weights will get revealed before we can check them
    if upcoming_tempo - current_block < 6:
        await async_wait_interval(
            tempo,
            async_subtensor,
            netuid=alice_subnet_netuid,
            reporting_interval=1,
        )
    current_block = await async_subtensor.chain.get_current_block()
    latest_drand_round = await async_subtensor.chain.last_drand_round()
    upcoming_tempo = next_tempo(current_block, tempo)
    logging.console.info(
        f"Post first wait_interval (to ensure window isn't too low): {current_block}, next tempo: {upcoming_tempo}, drand: {latest_drand_round}"
    )

    for mechid in range(TESTED_SUB_SUBNETS):
        logging.console.info(
            f"[magenta]Testing subnet mechanism: {alice_subnet_netuid}.{mechid}[/magenta]"
        )

        # commit_block is the block when weights were committed on the chain (transaction block)
        expected_commit_block = await async_subtensor.block + 1
        # Commit weights
        success, message = await async_subtensor.extrinsics.set_weights(
            wallet=alice_wallet,
            netuid=alice_subnet_netuid,
            mechid=mechid,
            uids=weight_uids,
            weights=weight_vals,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            block_time=BLOCK_TIME,
            period=16,
        )

        # Assert committing was a success
        assert success is True, message
        assert bool(re.match(r"reveal_round:\d+", message))

        # Parse expected reveal_round
        expected_reveal_round = int(message.split(":")[1])
        logging.console.success(
            f"Successfully set weights: uids {weight_uids}, weights {weight_vals}, reveal_round: {expected_reveal_round}"
        )

        # Fetch current commits pending on the chain
        await async_subtensor.wait_for_block(await async_subtensor.block + 12)

        commits_on_chain = (
            await async_subtensor.commitments.get_timelocked_weight_commits(
                netuid=alice_subnet_netuid, mechid=mechid
            )
        )
        address, commit_block, commit, reveal_round = commits_on_chain[0]

        # Assert correct values are committed on the chain
        assert expected_reveal_round == reveal_round
        assert address == alice_wallet.hotkey.ss58_address

        # bc of the drand delay, the commit block can be either the previous block or the current block
        # assert expected_commit_block in [commit_block - 1, commit_block, commit_block + 1]

        # Ensure no weights are available as of now
        assert (
            await async_subtensor.weights(netuid=alice_subnet_netuid, mechid=mechid)
            == []
        )
        logging.console.success("No weights are available before next epoch.")

        # 5 is safety drand offset
        expected_reveal_block = (
            await async_subtensor.subnets.get_next_epoch_start_block(
                alice_subnet_netuid
            )
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
            netuid=alice_subnet_netuid, mechid=mechid
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
                netuid=alice_subnet_netuid, mechid=mechid
            )
            == []
        )

        # Ensure the drand_round is always in the positive w.r.t expected when revealed
        assert latest_drand_round - expected_reveal_round >= -3, (
            f"latest_drand_round ({latest_drand_round}) is less than expected_reveal_round ({expected_reveal_round})"
        )

        logging.console.success("✅ Passed `test_commit_and_reveal_weights_cr4`")
