import asyncio
import sys

import pytest

from bittensor.core.subtensor import Subtensor
from tests.e2e_tests.utils.chain_interactions import (
    add_stake,
    register_subnet,
    wait_epoch,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    setup_wallet,
    template_path,
    templates_repo,
)
from bittensor.utils.balance import Balance
from bittensor.core.extrinsics.asyncex.weights import _do_set_weights
from bittensor.core.metagraph import Metagraph


FAST_BLOCKS_SPEEDUP_FACTOR = 5


@pytest.mark.asyncio
async def test_incentive(local_chain):
    """
    Test the incentive mechanism and interaction of miners/validators

    Steps:
        1. Register a subnet and register Alice & Bob
        2. Add Stake by Alice
        3. Run Alice as validator & Bob as miner. Wait Epoch
        4. Verify miner has correct: trust, rank, consensus, incentive
        5. Verify validator has correct: validator_permit, validator_trust, dividends, stake
    Raises:
        AssertionError: If any of the checks or verifications fail
    """

    print("Testing test_incentive")
    netuid = 1

    # Register root as Alice - the subnet owner and validator
    alice_keypair, alice_wallet = setup_wallet("//Alice")
    register_subnet(local_chain, alice_wallet)

    # Verify subnet <netuid> created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [netuid]
    ).serialize(), "Subnet wasn't created successfully"

    # Register Bob as miner
    bob_keypair, bob_wallet = setup_wallet("//Bob")

    subtensor = Subtensor(network="ws://localhost:9945")

    # Register Alice as a neuron on the subnet
    assert subtensor.burned_register(
        alice_wallet, netuid
    ), "Unable to register Alice as a neuron"

    # Register Bob as a neuron on the subnet
    assert subtensor.burned_register(
        bob_wallet, netuid
    ), "Unable to register Bob as a neuron"

    # Assert two neurons are in network
    assert (
        len(subtensor.neurons(netuid=netuid)) == 2
    ), "Alice & Bob not registered in the subnet"

    # Alice to stake to become to top neuron after the first epoch
    add_stake(local_chain, alice_wallet, Balance.from_tao(10_000))

    # Prepare to run Bob as miner
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/neurons/miner.py"',
            "--netuid",
            str(netuid),
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--wallet.path",
            bob_wallet.path,
            "--wallet.name",
            bob_wallet.name,
            "--wallet.hotkey",
            "default",
            "--logging.trace",
        ]
    )

    # Run Bob as miner in the background
    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    print("Neuron Bob is now mining")
    await asyncio.sleep(
        5
    )  # wait for 5 seconds for the metagraph to refresh with latest data

    # Prepare to run Alice as validator
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/neurons/validator.py"',
            "--netuid",
            str(netuid),
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--wallet.path",
            alice_wallet.path,
            "--wallet.name",
            alice_wallet.name,
            "--wallet.hotkey",
            "default",
            "--logging.trace",
        ]
    )

    # Run Alice as validator in the background
    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    print("Neuron Alice is now validating")
    await asyncio.sleep(
        5
    )  # wait for 5 seconds for the metagraph and subtensor to refresh with latest data

    # Get latest metagraph
    metagraph = Metagraph(netuid=netuid, network="ws://localhost:9945")

    # Get current miner/validator stats
    bob_neuron = metagraph.neurons[1]
    assert bob_neuron.incentive == 0
    assert bob_neuron.consensus == 0
    assert bob_neuron.rank == 0
    assert bob_neuron.trust == 0

    alice_neuron = metagraph.neurons[0]
    assert alice_neuron.validator_permit is False
    assert alice_neuron.dividends == 0
    assert alice_neuron.stake.tao == 10_000.0
    assert alice_neuron.validator_trust == 0

    # Wait until next epoch
    await wait_epoch(subtensor)

    # Set weights by Alice on the subnet
    await _do_set_weights(
        subtensor=subtensor.async_subtensor,
        wallet=alice_wallet,
        uids=[1],
        vals=[65535],
        netuid=netuid,
        version_key=0,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=5 * FAST_BLOCKS_SPEEDUP_FACTOR,
    )
    print("Alice neuron set weights successfully")

    await wait_epoch(subtensor)

    # Refresh metagraph
    metagraph = Metagraph(netuid=netuid, network="ws://localhost:9945")

    # Get current emissions and validate that Alice has gotten tao
    bob_neuron = metagraph.neurons[1]
    assert bob_neuron.incentive == 1
    assert bob_neuron.consensus == 1
    assert bob_neuron.rank == 1
    assert bob_neuron.trust == 1

    alice_neuron = metagraph.neurons[0]
    assert alice_neuron.validator_permit is True
    assert alice_neuron.dividends == 1
    assert alice_neuron.stake.tao == 10_000.0
    assert alice_neuron.validator_trust == 1

    print("✅ Passed test_incentive")
