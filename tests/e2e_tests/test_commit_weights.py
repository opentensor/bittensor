import asyncio

import numpy as np
import pytest

from bittensor.core.extrinsics.options import ExtrinsicEra
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from tests.e2e_tests.utils.chain_interactions import (
    sudo_set_admin_utils,
    sudo_set_hyperparameter_bool,
    use_and_wait_for_next_nonce,
    wait_epoch,
)


@pytest.mark.asyncio
async def test_commit_and_reveal_weights_legacy(local_chain, subtensor, alice_wallet):
    """
    Tests the commit/reveal weights mechanism with subprocess disabled (CR1.0)

    Steps:
        1. Register a subnet through Alice
        2. Enable commit-reveal mechanism on the subnet
        3. Lower the commit_reveal interval and rate limit
        4. Commit weights and verify
        5. Wait interval & reveal weights and verify
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    netuid = 2

    print("Testing test_commit_and_reveal_weights")

    # Register root as Alice
    assert subtensor.register_subnet(alice_wallet), "Unable to register the subnet"

    # Verify subnet 2 created successfully
    assert subtensor.subnet_exists(netuid), "Subnet wasn't created successfully"

    # Enable commit_reveal on the subnet
    assert sudo_set_hyperparameter_bool(
        local_chain,
        alice_wallet,
        "sudo_set_commit_reveal_weights_enabled",
        True,
        netuid,
    ), "Unable to enable commit reveal on the subnet"

    assert subtensor.commit_reveal_enabled(netuid), "Failed to enable commit/reveal"

    assert (
        subtensor.get_subnet_hyperparameters(netuid=netuid).commit_reveal_period == 1
    ), "Failed to set commit/reveal periods"

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
    assert subtensor.weights_rate_limit(netuid=netuid) == 0

    # Increase subnet tempo so we have enough time to commit and reveal weights
    sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_function="sudo_set_tempo",
        call_params={
            "netuid": netuid,
            "tempo": 100,
        },
    )

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

    assert success is True

    weight_commits = subtensor.query_module(
        module="SubtensorModule",
        name="WeightCommits",
        params=[netuid, alice_wallet.hotkey.ss58_address],
    )
    # Assert that the committed weights are set correctly
    assert weight_commits is not None, "Weight commit not found in storage"
    commit_hash, commit_block, reveal_block, expire_block = weight_commits[0]
    assert commit_block > 0, f"Invalid block number: {commit_block}"

    # Query the WeightCommitRevealInterval storage map
    assert (
        subtensor.get_subnet_reveal_period_epochs(netuid) > 0
    ), "Invalid RevealPeriodEpochs"

    # Wait until the reveal block range
    await wait_epoch(subtensor, netuid)

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

    assert success is True

    # Query the Weights storage map
    revealed_weights = subtensor.query_module(
        module="SubtensorModule",
        name="Weights",
        params=[netuid, 0],  # netuid and uid
    )

    # Assert that the revealed weights are set correctly
    assert revealed_weights is not None, "Weight reveal not found in storage"

    assert (
        weight_vals[0] == revealed_weights[0][1]
    ), f"Incorrect revealed weights. Expected: {weights[0]}, Actual: {revealed_weights[0][1]}"
    print("✅ Passed test_commit_and_reveal_weights")


@pytest.mark.asyncio
async def test_commit_weights_uses_next_nonce(local_chain, subtensor, alice_wallet):
    """
    Tests that commiting weights doesn't re-use a nonce in the transaction pool.

    Steps:
        1. Register a subnet through Alice
        2. Register Alice's neuron and add stake
        3. Enable commit-reveal mechanism on the subnet
        4. Lower the commit_reveal interval and rate limit
        5. Commit weights three times
        6. Assert that all commits succeeded
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    # Wait for 2 tempos to pass as CR3 only reveals weights after 2 tempos
    subtensor.wait_for_block(20)

    netuid = 2
    print("Testing test_commit_and_reveal_weights")
    # Register root as Alice
    assert subtensor.register_subnet(alice_wallet), "Unable to register the subnet"

    # Verify subnet 1 created successfully
    assert subtensor.subnet_exists(netuid), "Subnet wasn't created successfully"

    # Enable commit_reveal on the subnet
    assert sudo_set_hyperparameter_bool(
        local_chain,
        alice_wallet,
        "sudo_set_commit_reveal_weights_enabled",
        True,
        netuid,
    ), "Unable to enable commit reveal on the subnet"

    assert subtensor.commit_reveal_enabled(netuid), "Failed to enable commit/reveal"

    assert (
        subtensor.get_subnet_hyperparameters(netuid=netuid).commit_reveal_period == 1
    ), "Failed to set commit/reveal periods"

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
    assert subtensor.weights_rate_limit(netuid=netuid) == 0

    # Commit-reveal values
    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.1], dtype=np.float32)
    salt = [18, 179, 107, 0, 165, 211, 141, 197]
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )

    # Make a second salt
    salt2 = salt.copy()
    salt2[0] += 1  # Increment the first byte to produce a different commit hash

    # Make a third salt
    salt3 = salt.copy()
    salt3[0] += 2  # Increment the first byte to produce a different commit hash

    # Commit all three salts
    async with use_and_wait_for_next_nonce(subtensor, alice_wallet):
        success, message = subtensor.commit_weights(
            alice_wallet,
            netuid,
            salt=salt,
            uids=weight_uids,
            weights=weight_vals,
            wait_for_inclusion=False,  # Don't wait for inclusion, we are testing the nonce when there is a tx in the pool
            wait_for_finalization=False,
            era=ExtrinsicEra(period=144),
        )

        assert success is True

    async with use_and_wait_for_next_nonce(subtensor, alice_wallet):
        success, message = subtensor.commit_weights(
            alice_wallet,
            netuid,
            salt=salt2,
            uids=weight_uids,
            weights=weight_vals,
            wait_for_inclusion=False,
            wait_for_finalization=False,
            era=ExtrinsicEra(period=144),
        )

        assert success is True

    async with use_and_wait_for_next_nonce(subtensor, alice_wallet):
        success, message = subtensor.commit_weights(
            alice_wallet,
            netuid,
            salt=salt3,
            uids=weight_uids,
            weights=weight_vals,
            wait_for_inclusion=False,
            wait_for_finalization=False,
            era=ExtrinsicEra(period=144),
        )

        assert success is True

    # Wait for the txs to be included in the chain
    await asyncio.sleep(100)

    # Query the WeightCommits storage map for all three salts
    weight_commits = subtensor.query_module(
        module="SubtensorModule",
        name="WeightCommits",
        params=[netuid, alice_wallet.hotkey.ss58_address],
    )
    # Assert that the committed weights are set correctly
    assert weight_commits.value is not None, "Weight commit not found in storage"
    commit_hash, commit_block, reveal_block, expire_block = weight_commits.value[0]
    assert commit_block > 0, f"Invalid block number: {commit_block}"

    # Check for three commits in the WeightCommits storage map
    assert len(weight_commits.value) == 3, "Expected 3 weight commits"
