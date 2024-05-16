from substrateinterface import Keypair
from bittensor.commands.weights import CommitWeightCommand, RevealWeightCommand
import bittensor
from bittensor.extrinsics.network import register_subnetwork_extrinsic
import time

def test_commit_and_reveal_weights(local_chain):
    # Create a keypair for Alice
    alice_keypair = Keypair.create_from_uri("//Alice")

    # Create a test wallet and set the coldkey, coldkeypub, and hotkey
    wallet = bittensor.wallet(path="/tmp/btcli-wallet")
    wallet.set_coldkey(keypair=alice_keypair, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(keypair=alice_keypair, encrypt=False, overwrite=True)
    wallet.set_hotkey(keypair=alice_keypair, encrypt=False, overwrite=True)

    # Create a subnet with Alice's key using the register_subnetwork_extrinsic function
    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    success = register_subnetwork_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        prompt=False,
    )
    assert success, "Failed to create subnet"

    # Create a CLI parser and configure the CLI arguments for the CommitWeightCommand
    parser = bittensor.cli.__create_parser__()
    config = bittensor.config(
        parser=parser,
        args=[
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

    # Create a CLI instance with the configured arguments
    cli_instance = bittensor.cli(config)
    
    # Run the CommitWeightCommand
    CommitWeightCommand.run(cli_instance)

    # Query the WeightCommits storage map
    weight_commits = local_chain.query(
        module='SubtensorModule',
        storage_function='WeightCommits',
        params=[1, alice_keypair.ss58_address]
    )

    # Generate the expected commit hash
    uids = [1]
    weights = [0.1]
    version_key = bittensor.__version_as_int__  
    print(f"Test - uids: {uids}, weights: {weights}, version_key: {version_key}")
    expected_commit_hash = bittensor.utils.weight_utils.generate_weight_hash(
        who=alice_keypair.ss58_address,
        netuid=1,
        uids=uids,
        values=weights,
        version_key=version_key
    )

    # Assert that the committed weights are set correctly
    assert weight_commits.value is not None, "Weight commit not found in storage"
    commit_hash, commit_block = weight_commits.value
    # assert commit_hash == expected_commit_hash, f"Incorrect commit hash. Expected: {expected_commit_hash}, Actual: {commit_hash}"
    assert commit_block > 0, f"Invalid block number: {commit_block}"

    # Query the WeightCommitRevealInterval storage map
    weight_commit_reveal_interval = local_chain.query(
        module='SubtensorModule',
        storage_function='WeightCommitRevealInterval',
        params=[]
    )
    interval = weight_commit_reveal_interval.value
    assert interval > 0, "Invalid WeightCommitRevealInterval"

    # Wait until the reveal block range
    current_block = local_chain.get_block()['header']['number']
    reveal_block_start = (commit_block - (commit_block % interval)) + interval
    while current_block < reveal_block_start:
        time.sleep(1)  # Wait for 1 second before checking the block number again
        current_block = local_chain.get_block()['header']['number']

    # Configure the CLI arguments for the RevealWeightCommand
    config = bittensor.config(
        parser=parser,
        args=[
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

    # Create a CLI instance with the configured arguments
    # configured arguments
    cli_instance = bittensor.cli(config)

    # Run the RevealWeightCommand
    RevealWeightCommand.run(cli_instance)

    # Query the Weights storage map
    revealed_weights = local_chain.query(
        module='SubtensorModule',
        storage_function='Weights',
        params=[1, 1]  # netuid and uid
    )

    # Assert that the revealed weights are set correctly
    assert revealed_weights.value is not None, "Weight reveal not found in storage"
    expected_weights = [(1, int(0.1 * 1e9))]  # Convert weights to fixed-point integers
    assert revealed_weights.value == expected_weights, f"Incorrect revealed weights. Expected: {expected_weights}, Actual: {revealed_weights.value}"