from bittensor.commands.weights import CommitWeightCommand, RevealWeightCommand
from bittensor.commands.network import RegisterSubnetworkCommand
import bittensor
from tests.e2e_tests.utils import setup_wallet
import time


def test_commit_and_reveal_weights(local_chain):
    # Register root as Alice
    (alice_keypair, exec_command) = setup_wallet("//Alice")
    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    # Create a test wallet and set the coldkey, coldkeypub, and hotkey
    wallet = bittensor.wallet(path="/tmp/btcli-wallet")
    wallet.set_coldkey(keypair=alice_keypair, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(keypair=alice_keypair, encrypt=False, overwrite=True)
    wallet.set_hotkey(keypair=alice_keypair, encrypt=False, overwrite=True)

    subtensor = bittensor.subtensor(network="ws://localhost:9945")

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

    result = subtensor.set_hyperparameter(
        wallet=wallet,
        netuid=1,
        parameter="commit_reveal_weights_interval",
        value=7,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        prompt=False,
    )
    assert result, "Failed to set commit/reveal interval"

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
            "1",
            "--weights",
            "0.1",
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

    # Generate the expected commit hash
    uids = [1]
    weights = [0.1]
    version_key = bittensor.__version_as_int__
    expected_commit_hash = bittensor.utils.weight_utils.generate_weight_hash(
        who=wallet.hotkey.ss58_address,
        netuid=1,
        uids=uids,
        values=weights,
        version_key=version_key,
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
            "1",
            "--weights",
            "0.1",
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
        module="SubtensorModule", name="Weights", params=[1, 1]  # netuid and uid
    )

    # Assert that the revealed weights are set correctly
    assert revealed_weights.value is not None, "Weight reveal not found in storage"
    expected_weights = [(1, int(0.1 * 1e9))]  # Convert weights to fixed-point integers
    assert (
        revealed_weights.value == expected_weights
    ), f"Incorrect revealed weights. Expected: {expected_weights}, Actual: {revealed_weights.value}"
