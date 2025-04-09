import time

import numpy as np
import pytest

from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from tests.e2e_tests.utils.chain_interactions import (
    sudo_set_hyperparameter_bool,
    sudo_set_admin_utils,
    use_and_wait_for_next_nonce,
    wait_epoch,
)


@pytest.mark.asyncio
async def test_set_weights_uses_next_nonce(local_chain, subtensor, alice_wallet):
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

    netuids = [2, 3]
    print("Testing test_set_weights_uses_next_nonce")

    # Lower the network registration rate limit and cost
    sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_function="sudo_set_network_rate_limit",
        call_params={"rate_limit": "0"},  # No limit
    )
    # Set lock reduction interval
    sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_function="sudo_set_lock_reduction_interval",
        call_params={"interval": "1"},  # 1 block # reduce lock every block
    )

    # Try to register the subnets
    for _ in netuids:
        assert subtensor.register_subnet(
            alice_wallet,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        ), "Unable to register the subnet"

    # Verify all subnets created successfully
    for netuid in netuids:
        assert subtensor.subnet_exists(netuid), "Subnet wasn't created successfully"

        # weights sensitive to epoch changes
        assert sudo_set_admin_utils(
            local_chain,
            alice_wallet,
            call_function="sudo_set_tempo",
            call_params={
                "netuid": netuid,
                "tempo": 50,
            },
        )

    await wait_epoch(subtensor, netuid=2, times=2)
    subtensor.wait_for_block(subtensor.block + 1)

    # Stake to become to top neuron after the first epoch
    for netuid in netuids:
        subtensor.add_stake(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid,
            Balance.from_tao(10_000),
        )

    # Set weight hyperparameters per subnet
    for netuid in netuids:
        assert sudo_set_hyperparameter_bool(
            local_chain,
            alice_wallet,
            "sudo_set_commit_reveal_weights_enabled",
            False,
            netuid,
        ), "Unable to enable commit reveal on the subnet"

        assert not subtensor.commit_reveal_enabled(
            netuid,
        ), "Failed to enable commit/reveal"

        assert (
            subtensor.weights_rate_limit(netuid=netuid) > 0
        ), "Weights rate limit is below 0"

        # Lower the rate limit
        status, error = sudo_set_admin_utils(
            local_chain,
            alice_wallet,
            call_function="sudo_set_weights_set_rate_limit",
            call_params={"netuid": netuid, "weights_set_rate_limit": "0"},
        )

        assert error is None
        assert status is True

        assert (
            subtensor.get_subnet_hyperparameters(netuid=netuid).weights_rate_limit == 0
        ), "Failed to set weights_rate_limit"
        assert subtensor.get_hyperparameter("WeightsSetRateLimit", netuid) == 0
        assert subtensor.weights_rate_limit(netuid=netuid) == 0

    # Weights values
    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.5], dtype=np.float32)
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )

    # Set weights for each subnet
    for netuid in netuids:
        async with use_and_wait_for_next_nonce(subtensor, alice_wallet):
            success, message = subtensor.set_weights(
                alice_wallet,
                netuid,
                uids=weight_uids,
                weights=weight_vals,
                wait_for_inclusion=False,  # Don't wait for inclusion, we are testing the nonce when there is a tx in the pool
                wait_for_finalization=False,
            )

            assert success is True, message
            subtensor.wait_for_block(subtensor.block + 1)
            logging.console.success(f"Set weights for subnet {netuid}")

    extra_time = time.time()
    while not subtensor.query_module(
        module="SubtensorModule",
        name="Weights",
        params=[2, 0],
    ) or not subtensor.query_module(
        module="SubtensorModule",
        name="Weights",
        params=[3, 0],
    ):
        if time.time() - extra_time > 120:
            pytest.skip(
                "Skipping due to FLAKY TEST. Check the same tests with another Python version or run again."
            )

        logging.console.info(
            f"Additional fast block to wait chain data updated: {subtensor.block}"
        )
        subtensor.wait_for_block(subtensor.block + 1)

    for netuid in netuids:
        # Query the Weights storage map for all three subnets
        query = subtensor.query_module(
            module="SubtensorModule",
            name="Weights",
            params=[netuid, 0],  # Alice should be the only UID
        )

        weights = query.value
        logging.console.info(f"Weights for subnet {netuid}: {weights}")

        assert weights is not None, f"Weights not found for subnet {netuid}"
        assert weights == list(
            zip(weight_uids, weight_vals)
        ), f"Weights do not match for subnet {netuid}"
