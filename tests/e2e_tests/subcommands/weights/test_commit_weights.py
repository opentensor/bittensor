from bittensor.commands import (
    RegisterCommand,
    StakeCommand,
    RegisterSubnetworkCommand,
    CommitWeightCommand,
    RevealWeightCommand,
)
import bittensor
from tests.e2e_tests.utils import setup_wallet
import time
import bittensor.utils.weight_utils as weight_utils
import re
import numpy as np


def test_commit_and_reveal_weights(local_chain):
    # Register root as Alice
    (alice_keypair, exec_command) = setup_wallet("//Alice")
    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # define values
    weights = 0.1
    uid = 0

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    # Register a neuron to the subnet
    exec_command(
        RegisterCommand,
        ["s", "register", "--netuid", "1", "--wallet.path", "/tmp/btcli-wallet"],
    )

    # Create a test wallet and set the coldkey, coldkeypub, and hotkey
    wallet = bittensor.wallet(path="/tmp/btcli-wallet")
    wallet.set_coldkey(keypair=alice_keypair, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(keypair=alice_keypair, encrypt=False, overwrite=True)
    wallet.set_hotkey(keypair=alice_keypair, encrypt=False, overwrite=True)

    # Stake to become to top neuron after the first epoch
    exec_command(
        StakeCommand,
        [
            "stake",
            "add",
            "--wallet.path",
            "/tmp/btcli-wallet2",
            "--amount",
            "999998998",
        ],
    )

    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    # Enable Commit Reveal
    result = subtensor.set_hyperparameter(
        wallet=wallet,
        netuid=1,
        parameter="commit_reveal_weights_enabled",
        value=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        prompt=False,
    )
    assert result, "Failed to enable commit/reveal"

    # Lower the interval
    result = subtensor.set_hyperparameter(
        wallet=wallet,
        netuid=1,
        parameter="commit_reveal_weights_interval",
        value=370,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        prompt=False,
    )
    assert result, "Failed to set commit/reveal interval"

    # Lower the rate lmit
    result = subtensor.set_hyperparameter(
        wallet=wallet,
        netuid=1,
        parameter="weights_rate_limit",
        value=0,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        prompt=False,
    )
    assert result, "Failed to set weights rate limit"

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
    current_block = subtensor.get_current_block()
    reveal_block_start = (commit_block - (commit_block % interval)) + interval
    while current_block < reveal_block_start:
        time.sleep(1)  # Wait for 1 second before checking the block number again
        current_block = subtensor.get_current_block()
        if current_block % 10 == 0:
            print(f"Current Block: {current_block}  Revealing at: {reveal_block_start}")

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
        module="SubtensorModule", name="Weights", params=[1, uid]  # netuid and uid
    )

    # Assert that the revealed weights are set correctly
    assert revealed_weights.value is not None, "Weight reveal not found in storage"

    uid_list = list(map(int, re.split(r"[ ,]+", str(uid))))
    uids = np.array(uid_list, dtype=np.int64)
    weight_list = list(map(float, re.split(r"[ ,]+", str(weights))))
    weights_array = np.array(weight_list, dtype=np.float32)
    weight_uids, expected_weights = weight_utils.convert_weights_and_uids_for_emit(
        uids, weights_array
    )
    assert (
        expected_weights[0] == revealed_weights.value[0][1]
    ), f"Incorrect revealed weights. Expected: {expected_weights[0]}, Actual: {revealed_weights.value[0][1]}"
