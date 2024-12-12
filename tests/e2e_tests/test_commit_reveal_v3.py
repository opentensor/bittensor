import re

import numpy as np
import pytest
from bittensor.utils.btlogging import logging
from bittensor.core.subtensor import Subtensor
from bittensor.utils.balance import Balance
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from tests.e2e_tests.utils.chain_interactions import (
    add_stake,
    register_subnet,
    sudo_set_hyperparameter_bool,
    sudo_set_hyperparameter_values,
    wait_interval,
    sudo_set_admin_utils,
    next_tempo,
)
from tests.e2e_tests.utils.e2e_test_utils import setup_wallet


@pytest.mark.parametrize("local_chain", [False], indirect=True)
@pytest.mark.asyncio
async def test_commit_and_reveal_weights_cr3(local_chain):
    """
    Tests the commit/reveal weights mechanism (CR3)

    Steps:
        1. Register a subnet through Alice
        2. Register Alice's neuron and add stake
        3. Enable commit-reveal mechanism on the subnet
        4. Lower weights rate limit
        5. Change the tempo for subnet 1
        5. Commit weights and ensure they are committed.
        6. Wait interval & reveal weights and verify
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    netuid = 1
    logging.console.info("Testing test_commit_and_reveal_weights")

    # Register root as Alice
    keypair, alice_wallet = setup_wallet("//Alice")
    assert register_subnet(local_chain, alice_wallet), "Unable to register the subnet"

    # Verify subnet 1 created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [1]
    ).serialize(), "Subnet wasn't created successfully"

    logging.console.info("Subnet 1 is registered")

    subtensor = Subtensor(network="ws://localhost:9945")

    # Register Alice to the subnet
    assert subtensor.burned_register(
        alice_wallet, netuid
    ), "Unable to register Alice as a neuron"
    logging.console.info("Registered Alice to subnet 1")

    # Stake to become to top neuron after the first epoch
    add_stake(local_chain, alice_wallet, Balance.from_tao(100_000))
    logging.console.info("Stake added by Alice")

    # Enable commit_reveal on the subnet
    assert sudo_set_hyperparameter_bool(
        local_chain,
        alice_wallet,
        "sudo_set_commit_reveal_weights_enabled",
        True,
        netuid,
    ), "Unable to enable commit reveal on the subnet"

    # Verify commit_reveal was enabled
    assert subtensor.get_subnet_hyperparameters(
        netuid=netuid,
    ).commit_reveal_weights_enabled, "Failed to enable commit/reveal"
    logging.console.info("Commit reveal enabled")

    # Change the weights rate limit on the subnet
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_weights_set_rate_limit",
        call_params={"netuid": netuid, "weights_set_rate_limit": "0"},
        return_error_message=True,
    )

    # Verify weights rate limit was changed
    assert (
        subtensor.get_subnet_hyperparameters(netuid=netuid).weights_rate_limit == 0
    ), "Failed to set weights_rate_limit"
    assert subtensor.weights_rate_limit(netuid=netuid) == 0
    logging.console.info("sudo_set_weights_set_rate_limit executed: set to 0")

    # Change the tempo of the subnet from default 360
    # Since this is in normal blocks, this is necessary
    tempo_set = 10
    assert sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_function="sudo_set_tempo",
        call_params={"netuid": netuid, "tempo": tempo_set},
        return_error_message=True,
    )
    tempo = subtensor.get_subnet_hyperparameters(netuid=netuid).tempo
    assert tempo_set == tempo
    logging.console.info(f"sudo_set_tempo executed: set to {tempo_set}")

    # Commit-reveal values - setting weights to self
    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.1], dtype=np.float32)
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )

    # Fetch current block and calculate next tempo for the subnet
    current_block = subtensor.get_current_block()
    upcoming_tempo = next_tempo(current_block, tempo, netuid)
    logging.console.info(
        f"Checking if window is too low with Current block: {current_block}, next tempo: {upcoming_tempo}"
    )
    # Lower than this might mean weights will get revealed before we can check them
    if upcoming_tempo - current_block < 3:
        await wait_interval(
            tempo,
            subtensor,
            netuid=netuid,
            reporting_interval=1,
        )
    current_block = subtensor.get_current_block()
    latest_drand_round = subtensor.last_drand_round()
    upcoming_tempo = next_tempo(current_block, tempo, netuid)
    logging.console.info(
        f"Post first wait_interval (to ensure window isnt too low): {current_block}, next tempo: {upcoming_tempo}, drand: {latest_drand_round}"
    )

    # Commit weights
    success, message = subtensor.set_weights(
        alice_wallet,
        netuid,
        uids=weight_uids,
        weights=weight_vals,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Assert committing was a success
    assert success is True
    assert bool(re.match(r"reveal_round:\d+", message))

    # Parse expected reveal_round
    expected_reveal_round = int(message.split(":")[1])
    logging.console.info(
        f"Successfully set weights: uids {weight_uids}, weights {weight_vals}, reveal_round: {expected_reveal_round}"
    )

    current_block = subtensor.get_current_block()
    latest_drand_round = subtensor.last_drand_round()
    upcoming_tempo = next_tempo(current_block, tempo, netuid)
    logging.console.info(
        f"After setting weights: Current block: {current_block}, next tempo: {upcoming_tempo}, drand: {latest_drand_round}"
    )

    # Fetch current commits pending on the chain
    commits_on_chain = subtensor.get_weight_commits(netuid=netuid)
    address, commit, reveal_round = commits_on_chain[0]

    # Assert correct values are committed on the chain
    assert expected_reveal_round == reveal_round
    assert address == alice_wallet.hotkey.ss58_address

    # Ensure no weights are available as of now
    assert subtensor.weights(netuid=netuid) == []

    # Wait for the next tempo so weights can be revealed
    await wait_interval(
        subtensor.get_subnet_hyperparameters(netuid=netuid).tempo,
        subtensor,
        netuid=netuid,
        reporting_interval=1,
    )

    # Fetch the latest drand pulse
    latest_drand_round = subtensor.last_drand_round()
    logging.console.info(
        f"Latest drand round after waiting for tempo: {latest_drand_round}"
    )

    # Fetch weights on the chain as they should be revealed now
    revealed_weights = subtensor.weights(netuid=netuid)[0][1]

    # Assert correct weights were revealed
    assert weight_uids[0] == revealed_weights[0][0]
    assert weight_vals[0] == revealed_weights[0][1]

    # Now that the commit has been revealed, there shouldn't be any pending commits
    assert subtensor.get_weight_commits(netuid=netuid) == []

    # Ensure the drand_round is always in the positive w.r.t expected when revealed
    assert (
        latest_drand_round - expected_reveal_round >= 0
    ), f"latest_drand_round ({latest_drand_round}) is less than expected_reveal_round ({expected_reveal_round})"

    logging.console.info("✅ Passed commit_reveal v3")
