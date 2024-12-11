import time

import numpy as np
import pytest

from bittensor.core.subtensor import Subtensor
from bittensor.utils.balance import Balance
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from tests.e2e_tests.utils.chain_interactions import (
    add_stake,
    register_subnet,
    sudo_set_hyperparameter_bool,
    sudo_set_hyperparameter_values,
    wait_interval,
)
from tests.e2e_tests.utils.e2e_test_utils import setup_wallet


@pytest.mark.asyncio
async def test_set_weights(local_chain):
    """
    Tests the set weights mechanism

    Steps:
        1. Register a subnet through Alice
        2. Register Alice's neuron and add stake
        3. Register Bob's neuron
        4. Disable commit_reveal on the subnet
        5. Set min stake low enough for us to set weights
        6. Set weights rate limit
        7. Set weights and verify
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    netuid = 1
    print("Testing test_set_weights")
    # Register root as Alice
    keypair, alice_wallet = setup_wallet("//Alice")
    assert register_subnet(local_chain, alice_wallet), "Unable to register the subnet"

    # Verify subnet 1 created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [netuid]
    ).serialize(), "Subnet wasn't created successfully"

    subtensor = Subtensor(network="ws://localhost:9945")

    # Register Alice to the subnet
    assert subtensor.burned_register(
        alice_wallet, netuid
    ), "Unable to register Alice as a neuron"

    # Register a second neuron
    keypair, bob_wallet = setup_wallet("//Bob")
    assert subtensor.burned_register(
        bob_wallet, netuid
    ), "Unable to register Bob as a neuron"

    # Stake to become to top neuron after the first epoch
    add_stake(local_chain, alice_wallet, Balance.from_tao(100_000))

    # Disable commit_reveal on the subnet
    assert sudo_set_hyperparameter_bool(
        local_chain,
        alice_wallet,
        "sudo_set_commit_reveal_weights_enabled",
        False,
        netuid,
    ), "Unable to disable commit reveal on the subnet"

    assert (
        subtensor.get_subnet_hyperparameters(
            netuid=netuid,
        ).commit_reveal_weights_enabled
        is False
    ), "Failed to disable commit/reveal"

    # Set min stake low enough for us to set weights
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_weights_min_stake",
        call_params={"min_stake": 100},
        return_error_message=True,
    )

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

    # Weight values
    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.1], dtype=np.float32)
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )

    # Set weights
    success, message = subtensor.set_weights(
        alice_wallet,
        netuid,
        uids=weight_uids,
        weights=weight_vals,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert success is True

    time.sleep(10)

    # Query the Weights storage map
    chain_weights = subtensor.query_module(
        module="SubtensorModule",
        name="Weights",
        params=[netuid, 0],  # netuid and uid
    )

    # Assert that the revealed weights are set correctly
    assert chain_weights.value is not None, "Weight set not found in storage"

    assert (
        weight_vals[0] == chain_weights.value[0][1]
    ), f"Incorrect weights. Expected: {weights[0]}, Actual: {chain_weights.value[0][1]}"
    print("✅ Passed test_set_weights")


@pytest.mark.asyncio
async def test_batch_set_weights(local_chain):
    """
    Tests the batch set weights mechanism

    Steps:
        1. Register multiple subnets through Alice
        2. Register Alice's neurons and add stake
        3. Register Bob's neurons
        4. Disable commit_reveal on the subnets
        5. Set min stake low enough for us to set weights
        6. Set weights rate limit
        7. Set weights and verify
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    netuid_1 = 1
    netuid_2 = 2
    print("Testing test_batch_set_weights")
    # Register root as Alice
    keypair, alice_wallet = setup_wallet("//Alice")
    assert register_subnet(local_chain, alice_wallet), "Unable to register the subnet"
    assert register_subnet(
        local_chain, alice_wallet
    ), "Unable to register the second subnet"

    # Verify subnet 1 created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [netuid_1]
    ).serialize(), "Subnet wasn't created successfully"

    # Verify subnet 2 created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [netuid_2]
    ).serialize(), "Subnet wasn't created successfully"

    subtensor = Subtensor(network="ws://localhost:9945")

    # Register Alice to the subnet
    assert subtensor.burned_register(
        alice_wallet, netuid_1
    ), "Unable to register Alice as a neuron"

    # Register Alice to the second ubnet
    assert subtensor.burned_register(
        alice_wallet, netuid_2
    ), "Unable to register Alice as a neuron to the second subnet"

    # Register a second neuron
    keypair, bob_wallet = setup_wallet("//Bob")
    assert subtensor.burned_register(
        bob_wallet, netuid_1
    ), "Unable to register Bob as a neuron"

    # Register a second neuron to the second subnet
    assert subtensor.burned_register(
        bob_wallet, netuid_2
    ), "Unable to register Bob as a neuron to the second subnet"

    # Stake to become to top neuron after the first epoch
    add_stake(local_chain, alice_wallet, Balance.from_tao(100_000))

    # Disable commit_reveal on both subnets
    assert sudo_set_hyperparameter_bool(
        local_chain,
        alice_wallet,
        "sudo_set_commit_reveal_weights_enabled",
        False,
        netuid_1,
    ), "Unable to disable commit reveal on the first subnet"

    assert sudo_set_hyperparameter_bool(
        local_chain,
        alice_wallet,
        "sudo_set_commit_reveal_weights_enabled",
        False,
        netuid_2,
    ), "Unable to disable commit reveal on the second subnet"

    assert (
        subtensor.get_subnet_hyperparameters(
            netuid=netuid_1,
        ).commit_reveal_weights_enabled
        is False
    ), "Failed to disable commit/reveal on the first subnet"

    assert (
        subtensor.get_subnet_hyperparameters(
            netuid=netuid_2,
        ).commit_reveal_weights_enabled
        is False
    ), "Failed to disable commit/reveal on the second subnet"

    # Set min stake low enough for us to set weights
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_weights_min_stake",
        call_params={"min_stake": 100},
        return_error_message=True,
    )

    assert (
        subtensor.weights_rate_limit(netuid=netuid_1) > 0
    ), "Weights rate limit is below 0"

    assert (
        subtensor.weights_rate_limit(netuid=netuid_2) > 0
    ), "Weights rate limit is below 0"

    # Lower the rate limit
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_weights_set_rate_limit",
        call_params={"netuid": netuid_1, "weights_set_rate_limit": "0"},
        return_error_message=True,
    )

    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_weights_set_rate_limit",
        call_params={"netuid": netuid_2, "weights_set_rate_limit": "0"},
        return_error_message=True,
    )

    assert (
        subtensor.get_subnet_hyperparameters(netuid=netuid_1).weights_rate_limit == 0
    ), "Failed to set weights_rate_limit on the first subnet"

    assert (
        subtensor.get_subnet_hyperparameters(netuid=netuid_2).weights_rate_limit == 0
    ), "Failed to set weights_rate_limit on the second subnet"

    assert subtensor.weights_rate_limit(netuid=netuid_1) == 0
    assert subtensor.weights_rate_limit(netuid=netuid_2) == 0

    # Weight values
    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.1], dtype=np.float32)
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )

    # Set weights in a batch
    success, message = subtensor.batch_set_weights(
        alice_wallet,
        netuids=[netuid_1, netuid_2],
        nested_uids=[weight_uids, weight_uids],
        nested_weights=[weight_vals, weight_vals],
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert success is True

    time.sleep(10)

    # Query the Weights storage map
    chain_weights_1 = subtensor.query_module(
        module="SubtensorModule",
        name="Weights",
        params=[netuid_1, 0],  # netuid and uid
    )

    chain_weights_2 = subtensor.query_module(
        module="SubtensorModule",
        name="Weights",
        params=[netuid_2, 0],  # netuid and uid
    )

    # Assert that the revealed weights are set correctly
    assert chain_weights_1.value is not None, "Weight set not found in storage"
    assert chain_weights_2.value is not None, "Weight set not found in storage"

    assert (
        weight_vals[0] == chain_weights_1.value[0][1]
    ), f"Incorrect weights. Expected: {weights[0]}, Actual: {chain_weights_1.value[0][1]}"

    assert (
        weight_vals[0] == chain_weights_2.value[0][1]
    ), f"Incorrect weights. Expected: {weights[0]}, Actual: {chain_weights_2.value[0][1]}"

    print("✅ Passed test_batch_set_weights")
