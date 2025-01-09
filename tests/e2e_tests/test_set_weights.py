import numpy as np
import pytest

import asyncio

from bittensor.core.subtensor import Subtensor
from bittensor.utils.balance import Balance
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from bittensor.core.extrinsics import utils
from tests.e2e_tests.utils.chain_interactions import (
    add_stake,
    register_subnet,
    sudo_set_hyperparameter_bool,
    sudo_set_hyperparameter_values,
    sudo_set_admin_utils,
)
from tests.e2e_tests.utils.e2e_test_utils import setup_wallet


@pytest.mark.asyncio
async def test_set_weights_uses_next_nonce(local_chain):
    """
    Tests that setting weights doesn't re-use a nonce in the transaction pool.

    Steps:
        1. Register three subnets through Alice
        2. Register Alice's neuron on each subnet and add stake
        3. Verify Alice has a vpermit on each subnet
        4. Lower the set weights rate limit on each subnet
        5. Set weights on each subnet
        6. Assert that all the set weights succeeded
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    netuids = [1, 2]
    utils.EXTRINSIC_SUBMISSION_TIMEOUT = 12  # handle fast blocks
    print("Testing test_set_weights_uses_next_nonce")
    # Register root as Alice
    keypair, alice_wallet = setup_wallet("//Alice")

    # Lower the network registration rate limit and cost
    sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_function="sudo_set_network_rate_limit",
        call_params={"rate_limit": "0"},  # No limit
        return_error_message=True,
    )
    # Set lock reduction interval
    sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_function="sudo_set_lock_reduction_interval",
        call_params={"interval": "1"},  # 1 block # reduce lock every block
        return_error_message=True,
    )
    # Try to register the subnets
    for _ in netuids:
        assert register_subnet(
            local_chain, alice_wallet
        ), "Unable to register the subnet"

    # Verify all subnets created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [3]
    ).serialize(), "Subnet wasn't created successfully"

    subtensor = Subtensor(network="ws://localhost:9945")

    for netuid in netuids:
        # Allow registration on the subnet
        assert sudo_set_hyperparameter_values(
            local_chain,
            alice_wallet,
            "sudo_set_network_registration_allowed",
            {"netuid": netuid, "registration_allowed": True},
            return_error_message=True,
        )

    # This should give a gap for the calls above to be included in the chain
    await asyncio.sleep(2)

    for netuid in netuids:
        # Register Alice to the subnet
        assert subtensor.burned_register(
            alice_wallet, netuid
        ), f"Unable to register Alice as a neuron on SN{netuid}"

    # Stake to become to top neuron after the first epoch
    add_stake(local_chain, alice_wallet, Balance.from_tao(100_000))

    # Set weight hyperparameters per subnet
    for netuid in netuids:
        assert sudo_set_hyperparameter_bool(
            local_chain,
            alice_wallet,
            "sudo_set_commit_reveal_weights_enabled",
            False,
            netuid,
        ), "Unable to enable commit reveal on the subnet"

        assert not subtensor.get_subnet_hyperparameters(
            netuid=netuid,
        ).commit_reveal_weights_enabled, "Failed to enable commit/reveal"

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

    # Weights values
    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.1], dtype=np.float32)
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )

    # Set weights for each subnet
    for netuid in netuids:
        success, message = subtensor.set_weights(
            alice_wallet,
            netuid,
            uids=weight_uids,
            weights=weight_vals,
            wait_for_inclusion=False,  # Don't wait for inclusion, we are testing the nonce when there is a tx in the pool
            wait_for_finalization=False,
        )

        assert success is True, f"Failed to set weights for subnet {netuid}"

    # Wait for the txs to be included in the chain
    await asyncio.sleep(4)

    for netuid in netuids:
        # Query the Weights storage map for all three subnets
        weights = subtensor.query_module(
            module="SubtensorModule",
            name="Weights",
            params=[netuid, 0],  # Alice should be the only UID
        )

        assert weights is not None, f"Weights not found for subnet {netuid}"
        assert weights == list(
            zip(weight_uids, weight_vals)
        ), f"Weights do not match for subnet {netuid}"
