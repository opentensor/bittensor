import time

import numpy as np
import pytest
import bittensor.utils.subprocess.commit_reveal as commit_reveal_subprocess
import bittensor
from bittensor import logging
from bittensor.utils import subprocess_utils
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
    Tests the commit/reveal weights mechanism with a subprocess doing the reveal function

    Steps:
        1. Register a subnet through Alice
        2. Register Alice's neuron and add stake
        3. Enable commit-reveal mechanism on the subnet
        4. Lower the commit_reveal interval and rate limit
        5. Commit weights and verify
        6. Wait interval & see if subprocess did the reveal weights and verify
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

    subtensor = bittensor.Subtensor(
        network="ws://localhost:9945", subprocess_sleep_interval=0.25
    )
    assert subtensor.get_subnet_hyperparameters(
        netuid=netuid
    ).commit_reveal_weights_enabled, "Failed to enable commit/reveal"

    # Lower the commit_reveal interval
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_commit_reveal_weights_periods",
        call_params={"netuid": netuid, "periods": "1"},
        return_error_message=True,
    )

    assert (
        subtensor.get_subnet_hyperparameters(netuid=netuid).commit_reveal_periods == 1
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
    commit_hash, commit_block, reveal_block, expire_block = weight_commits.value[0]
    assert commit_block > 0, f"Invalid block number: {commit_block}"

    # Query the WeightCommitRevealInterval storage map
    reveal_periods = subtensor.query_module(
        module="SubtensorModule", name="RevealPeriodEpochs", params=[netuid]
    )
    periods = reveal_periods.value
    assert periods > 0, "Invalid RevealPeriodEpochs"

    # Verify that sqlite has entry
    assert commit_reveal_subprocess.is_table_empty("commits") is False

    # Wait until the reveal block range
    await wait_interval(
        subtensor.get_subnet_hyperparameters(netuid=netuid).tempo, subtensor
    )

    # allow one more block to pass
    time.sleep(12)

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


@pytest.mark.asyncio
async def test_set_and_reveal_weights(local_chain):
    """
    Tests the commit/reveal weights mechanism with a subprocess doing the reveal function

    Steps:
        1. Register a subnet through Alice
        2. Register Alice's neuron and add stake
        3. Enable commit-reveal mechanism on the subnet
        4. Lower the commit_reveal interval and rate limit
        5. Commit weights and verify
        6. Wait interval & see if subprocess did the reveal weights and verify
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    netuid = 1
    logging.info("Testing test_set_and_reveal_weights")
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

    subtensor = bittensor.Subtensor(
        network="ws://localhost:9945", subprocess_sleep_interval=0.25
    )  # Subprocess works with fast blocks
    assert subtensor.get_subnet_hyperparameters(
        netuid=netuid
    ).commit_reveal_weights_enabled, "Failed to enable commit/reveal"

    # Lower the commit_reveal interval
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_commit_reveal_weights_periods",
        call_params={"netuid": netuid, "periods": "1"},
        return_error_message=True,
    )

    assert (
        subtensor.get_subnet_hyperparameters(netuid=netuid).commit_reveal_periods == 1
    ), "Failed to set commit/reveal period"

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

    assert (
        subtensor.get_subnet_hyperparameters(netuid=netuid).weights_rate_limit == 0
    ), "Failed to set weights_rate_limit"
    assert subtensor.weights_rate_limit(netuid=netuid) == 0

    # Commit-reveal values
    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.1], dtype=np.float32)
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )

    # Set weights with CR enabled
    success, message = subtensor.set_weights(
        alice_wallet,
        netuid,
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
    commit_hash, commit_block, reveal_block, expire_block = weight_commits.value[0]
    assert commit_block > 0, f"Invalid block number: {commit_block}"

    # Query the WeightCommitRevealInterval storage map
    reveal_periods = subtensor.query_module(
        module="SubtensorModule", name="RevealPeriodEpochs", params=[netuid]
    )
    periods = reveal_periods.value
    assert periods > 0, "Invalid RevealPeriodEpochs"

    # Verify that sqlite has entry
    assert commit_reveal_subprocess.is_table_empty("commits") is False

    # Wait until the reveal block range
    await wait_interval(
        subtensor.get_subnet_hyperparameters(netuid=netuid).tempo, subtensor
    )

    # allow one more block to pass
    time.sleep(12)

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


@pytest.mark.asyncio
async def test_set_and_reveal_batch_weights(local_chain):
    """
    Tests the commit/reveal batch weights mechanism with a subprocess doing the reveal function

    Steps:
        1. Register a subnet through Alice
        2. Register Alice's neuron and add stake
        3. Enable commit-reveal mechanism on the subnet
        4. Lower the commit_reveal interval and rate limit
        5. Commit weights and verify
        6. Wait interval & see if subprocess did the reveal weights and verify
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    netuid = 1
    logging.info("Testing test_set_and_reveal_weights")
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

    subtensor = bittensor.Subtensor(
        network="ws://localhost:9945", subprocess_sleep_interval=0.25
    )  # Subprocess works with fast blocks
    assert subtensor.get_subnet_hyperparameters(
        netuid=netuid
    ).commit_reveal_weights_enabled, "Failed to enable commit/reveal"

    # Lower the commit_reveal interval
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_commit_reveal_weights_periods",
        call_params={"netuid": netuid, "periods": "1"},
        return_error_message=True,
    )

    assert (
        subtensor.get_subnet_hyperparameters(netuid=netuid).commit_reveal_periods == 1
    ), "Failed to set commit/reveal periods"

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
    assert (
        subtensor.get_subnet_hyperparameters(netuid=netuid).weights_rate_limit == 0
    ), "Failed to set weights_rate_limit"
    assert subtensor.weights_rate_limit(netuid=netuid) == 0

    # Commit-reveal values and weights for different steps
    weights_steps = [
        (np.array([0], dtype=np.int64), np.array([0.1], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.2], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.3], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.4], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.5], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.6], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.7], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.8], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.9], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.10], dtype=np.float32)),
    ]

    for uids, weights in weights_steps:
        # Customers run this before submitting weights
        weight_uids, weight_vals = convert_weights_and_uids_for_emit(
            uids=uids, weights=weights
        )

        # Set weights with CR enabled
        success, message = subtensor.set_weights(
            alice_wallet,
            netuid,
            uids=weight_uids,
            weights=weight_vals,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        assert success

        time.sleep(2)

    weight_commits = subtensor.query_module(
        module="SubtensorModule",
        name="WeightCommits",
        params=[netuid, alice_wallet.hotkey.ss58_address],
    )

    # Assert that the committed weights are set correctly
    assert weight_commits.value is not None, "Weight commit not found in storage"
    commit_hash, commit_block, reveal_block, expire_block = weight_commits.value[0]
    assert commit_block > 0, f"Invalid block number: {commit_block}"

    # Query the WeightCommitRevealInterval storage map
    reveal_periods = subtensor.query_module(
        module="SubtensorModule", name="RevealPeriodEpochs", params=[netuid]
    )
    periods = reveal_periods.value
    assert periods > 0, "Invalid RevealPeriodEpochs"

    # Wait until the reveal block range
    await wait_interval(
        subtensor.get_subnet_hyperparameters(netuid=netuid).tempo, subtensor
    )

    # allow one more block to pass
    time.sleep(12)

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


@pytest.mark.asyncio
@pytest.mark.timeout(120)  # 4 minute timeout
async def test_set_and_reveal_batch_weights_over_limit(local_chain):
    """
    Tests the commit/reveal batch weights mechanism with 11 commits, which should throw an exception

    Steps:
        1. Register a subnet through Alice
        2. Register Alice's neuron and add stake
        3. Enable commit-reveal mechanism on the subnet
        4. Lower the commit_reveal interval and rate limit
        5. Commit weights and verify
        6. Wait interval & see if subprocess did the reveal weights and verify
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    netuid = 1
    logging.info("Testing test_set_and_reveal_weights")
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

    subtensor = bittensor.Subtensor(
        network="ws://localhost:9945", subprocess_sleep_interval=0.25
    )  # Subprocess works with fast blocks
    assert subtensor.get_subnet_hyperparameters(
        netuid=netuid
    ).commit_reveal_weights_enabled, "Failed to enable commit/reveal"

    # Lower the commit_reveal interval
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_commit_reveal_weights_periods",
        call_params={"netuid": netuid, "periods": "1"},
        return_error_message=True,
    )

    assert (
            subtensor.get_subnet_hyperparameters(netuid=netuid).commit_reveal_periods == 1
    ), "Failed to set commit/reveal periods"

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
    assert (
            subtensor.get_subnet_hyperparameters(netuid=netuid).weights_rate_limit == 0
    ), "Failed to set weights_rate_limit"
    assert subtensor.weights_rate_limit(netuid=netuid) == 0

    # Commit-reveal values and weights for different steps
    weights_steps = [
        (np.array([0], dtype=np.int64), np.array([0.1], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.2], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.3], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.4], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.5], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.6], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.7], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.8], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.9], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([0.10], dtype=np.float32)),
    ]

    for uids, weights in weights_steps:
        # Customers run this before submitting weights
        weight_uids, weight_vals = convert_weights_and_uids_for_emit(
            uids=uids, weights=weights
        )

        # Set weights with CR enabled
        success, message = subtensor.set_weights(
            alice_wallet,
            netuid,
            uids=weight_uids,
            weights=weight_vals,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        assert success

        time.sleep(1)

    # 11th time (should throw error)
    # Commit-reveal values
    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.3], dtype=np.float32)
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )

    # Set weights with CR enabled
    success, message = subtensor.set_weights(
        alice_wallet,
        netuid,
        uids=weight_uids,
        weights=weight_vals,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert success is False
