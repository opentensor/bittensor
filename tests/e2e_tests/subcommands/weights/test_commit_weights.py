import asyncio
import re

import numpy as np
import pytest

import bittensor
import bittensor.utils.weight_utils as weight_utils
from bittensor import logging
from bittensor.commands import (
    CommitWeightCommand,
    RegisterCommand,
    RegisterSubnetworkCommand,
    RevealWeightCommand,
    StakeCommand,
    SubnetSudoCommand,
)
from tests.e2e_tests.utils import setup_wallet, wait_interval

"""
Test the Commit/Reveal weights mechanism. 

Verify that:
* Weights are commited
* weights are hashed with salt 
--- after an epoch ---
* weights are un-hashed with salt
* weights are properly revealed

"""


@pytest.mark.asyncio
async def test_commit_and_reveal_weights(local_chain):
    logging.info("Testing test_commit_and_reveal_weights")
    # Register root as Alice
    keypair, exec_command, wallet = setup_wallet("//Alice")

    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # define values
    weights = 0.1
    uid = 0
    salt = "18, 179, 107, 0, 165, 211, 141, 197"

    # Verify subnet 1 created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [1]
    ).serialize(), "Subnet wasn't created successfully"

    # Register a neuron to the subnet
    exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
        ],
    )

    # Stake to become to top neuron after the first epoch
    exec_command(
        StakeCommand,
        [
            "stake",
            "add",
            "--amount",
            "100000",
        ],
    )

    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    # Enable Commit Reveal
    exec_command(
        SubnetSudoCommand,
        [
            "sudo",
            "set",
            "hyperparameters",
            "--netuid",
            "1",
            "--wallet.name",
            wallet.name,
            "--param",
            "commit_reveal_weights_enabled",
            "--value",
            "True",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    assert subtensor.get_subnet_hyperparameters(
        netuid=1
    ).commit_reveal_weights_enabled, "Failed to enable commit/reveal"

    # Lower the interval
    exec_command(
        SubnetSudoCommand,
        [
            "sudo",
            "set",
            "hyperparameters",
            "--netuid",
            "1",
            "--wallet.name",
            wallet.name,
            "--param",
            "commit_reveal_weights_interval",
            "--value",
            "370",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    assert (
        subtensor.get_subnet_hyperparameters(netuid=1).commit_reveal_weights_interval
        == 370
    ), "Failed to set commit/reveal interval"

    # Lower the rate limit
    exec_command(
        SubnetSudoCommand,
        [
            "sudo",
            "set",
            "hyperparameters",
            "--netuid",
            "1",
            "--wallet.name",
            wallet.name,
            "--param",
            "weights_rate_limit",
            "--value",
            "0",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    assert (
        subtensor.get_subnet_hyperparameters(netuid=1).weights_rate_limit == 0
    ), "Failed to set commit/reveal rate limit"

    # Configure the CLI arguments for the CommitWeightCommand
    exec_command(
        CommitWeightCommand,
        [
            "wt",
            "commit",
            "--no_prompt",
            "--netuid",
            "1",
            "--uids",
            str(uid),
            "--weights",
            str(weights),
            "--salt",
            str(salt),
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--wallet.path",
            "/tmp/btcli-wallet",
        ],
    )

    weight_commits = subtensor.query_module(
        module="SubtensorModule",
        name="WeightCommits",
        params=[1, wallet.hotkey.ss58_address],
    )

    # Assert that the committed weights are set correctly
    assert weight_commits.value is not None, "Weight commit not found in storage"
    commit_hash, commit_block = weight_commits.value
    assert commit_block > 0, f"Invalid block number: {commit_block}"

    # Query the WeightCommitRevealInterval storage map
    weight_commit_reveal_interval = subtensor.query_module(
        module="SubtensorModule", name="WeightCommitRevealInterval", params=[1]
    )
    interval = weight_commit_reveal_interval.value
    assert interval > 0, "Invalid WeightCommitRevealInterval"

    # Wait until the reveal block range
    await wait_interval(interval, subtensor)

    # Configure the CLI arguments for the RevealWeightCommand
    exec_command(
        RevealWeightCommand,
        [
            "wt",
            "reveal",
            "--no_prompt",
            "--netuid",
            "1",
            "--uids",
            str(uid),
            "--weights",
            str(weights),
            "--salt",
            str(salt),
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--wallet.path",
            "/tmp/btcli-wallet",
        ],
    )

    # Query the Weights storage map
    revealed_weights = subtensor.query_module(
        module="SubtensorModule",
        name="Weights",
        params=[1, uid],  # netuid and uid
    )

    # Assert that the revealed weights are set correctly
    assert revealed_weights.value is not None, "Weight reveal not found in storage"

    uid_list = [int(x) for x in re.split(r"[ ,]+", str(uid))]
    uids = np.array(uid_list, dtype=np.int64)
    weight_list = [float(x) for x in re.split(r"[ ,]+", str(weights))]
    weights_array = np.array(weight_list, dtype=np.float32)
    weight_uids, expected_weights = weight_utils.convert_weights_and_uids_for_emit(
        uids, weights_array
    )
    assert (
        expected_weights[0] == revealed_weights.value[0][1]
    ), f"Incorrect revealed weights. Expected: {expected_weights[0]}, Actual: {revealed_weights.value[0][1]}"
    logging.info("Passed test_commit_and_reveal_weights")
