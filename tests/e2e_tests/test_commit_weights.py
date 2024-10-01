import time

import numpy as np
import pytest

import bittensor
from bittensor import logging
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from tests.e2e_tests.utils.chain_interactions import (
    add_stake,
    register_neuron,
    register_subnet,
    sudo_set_hyperparameter_bool,
    sudo_set_hyperparameter_values,
    wait_interval,
)
from tests.e2e_tests.utils.e2e_test_utils import setup_wallet


@pytest.mark.asyncio
async def test_commit_and_reveal_weights(local_chain):
    """
    Tests the commit/reveal weights mechanism

    Steps:
        1. Register a subnet through Alice
        2. Register Alice's neuron and add stake
        3. Enable commit-reveal mechanism on the subnet
        4. Lower the commit_reveal interval and rate limit
        5. Commit weights and verify
        6. Wait interval & reveal weights and verify
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    netuid = 1
    logging.info("Testing test_commit_and_reveal_weights")
    # Register root as Alice
    keypair, alice_wallet = setup_wallet("//Alice")
    assert register_subnet(local_chain, alice_wallet), "Unable to register the subnet"

    # Verify subnet 1 created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [1]
    ).serialize(), "Subnet wasn't created successfully"

    assert register_neuron(
        local_chain, alice_wallet, netuid
    ), "Unable to register Alice as a neuron"

    # Stake to become to top neuron after the first epoch
    add_stake(local_chain, alice_wallet, bittensor.Balance.from_tao(100_000))

    # Enable commit_reveal on the subnet
    assert sudo_set_hyperparameter_bool(
        local_chain,
        alice_wallet,
        "sudo_set_commit_reveal_weights_enabled",
        True,
        netuid,
    ), "Unable to enable commit reveal on the subnet"

    subtensor = bittensor.Subtensor(network="ws://localhost:9945")
    assert subtensor.get_subnet_hyperparameters(
        netuid=netuid
    ).commit_reveal_weights_enabled, "Failed to enable commit/reveal"

    # Lower the commit_reveal interval
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_commit_reveal_weights_interval",
        call_params={"netuid": netuid, "interval": "370"},
        return_error_message=True,
    )

    subtensor = bittensor.Subtensor(network="ws://localhost:9945")
    assert (
        subtensor.get_subnet_hyperparameters(
            netuid=netuid
        ).commit_reveal_weights_interval
        == 370
    ), "Failed to set commit/reveal interval"

    assert (
        subtensor.weights_rate_limit(netuid=netuid) > 0
    ), "Weights rate limit is below 0"
    # Lower the rate limit
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_weights_set_rate_limit",
        call_params={"netuid": netuid, "weights_set_rate_limit": "0"},
        return_error_message=True,
    )
    subtensor = bittensor.Subtensor(network="ws://localhost:9945")
    assert (
        subtensor.get_subnet_hyperparameters(netuid=netuid).weights_rate_limit == 0
    ), "Failed to set weights_rate_limit"
    assert subtensor.weights_rate_limit(netuid=netuid) == 0

    # Commit-reveal values
    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.1], dtype=np.float32)
    salt = [18, 179, 107, 0, 165, 211, 141, 197]
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )

    # Commit weights
    success, message = subtensor.commit_weights(
        alice_wallet,
        netuid,
        salt=salt,
        uids=weight_uids,
        weights=weight_vals,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    weight_commits = subtensor.query_module(
        module="SubtensorModule",
        name="WeightCommits",
        params=[netuid, alice_wallet.hotkey.ss58_address],
    )
    # Assert that the committed weights are set correctly
    assert weight_commits.value is not None, "Weight commit not found in storage"
    commit_hash, commit_block = weight_commits.value
    assert commit_block > 0, f"Invalid block number: {commit_block}"

    # Query the WeightCommitRevealInterval storage map
    weight_commit_reveal_interval = subtensor.query_module(
        module="SubtensorModule", name="WeightCommitRevealInterval", params=[netuid]
    )
    interval = weight_commit_reveal_interval.value
    assert interval > 0, "Invalid WeightCommitRevealInterval"

    # Wait until the reveal block range
    await wait_interval(interval, subtensor)

    # Reveal weights
    success, message = subtensor.reveal_weights(
        alice_wallet,
        netuid,
        uids=weight_uids,
        weights=weight_vals,
        salt=salt,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    time.sleep(10)

    # Query the Weights storage map
    revealed_weights = subtensor.query_module(
        module="SubtensorModule",
        name="Weights",
        params=[netuid, 0],  # netuid and uid
    )

    # Assert that the revealed weights are set correctly
    assert revealed_weights.value is not None, "Weight reveal not found in storage"

    assert (
        weight_vals[0] == revealed_weights.value[0][1]
    ), f"Incorrect revealed weights. Expected: {weights[0]}, Actual: {revealed_weights.value[0][1]}"
    logging.info("✅ Passed test_commit_and_reveal_weights")
