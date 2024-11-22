import asyncio
import sys

import pytest

from bittensor.core.metagraph import Metagraph
from bittensor.core.subtensor import Subtensor
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.chain_interactions import (
    register_subnet,
    add_stake,
    wait_epoch,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    setup_wallet,
    template_path,
    templates_repo,
)


@pytest.mark.asyncio
async def test_dendrite(local_chain):
    """
    Test the Dendrite mechanism

    Steps:
        1. Register a subnet through Alice
        2. Register Bob as a validator
        3. Add stake to Bob and ensure neuron is not a validator yet
        4. Run Bob as a validator and wait epoch
        5. Ensure Bob's neuron has all correct attributes of a validator
    Raises:
        AssertionError: If any of the checks or verifications fail
    """

    logging.console.info("Testing test_dendrite")
    netuid = 1

    # Register root as Alice - the subnet owner
    alice_keypair, alice_wallet = setup_wallet("//Alice")

    # Register a subnet, netuid 1
    assert register_subnet(local_chain, alice_wallet), "Subnet wasn't created"

    # Verify subnet <netuid> created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [netuid]
    ).serialize(), "Subnet wasn't created successfully"

    # Register Bob
    bob_keypair, bob_wallet = setup_wallet("//Bob")

    subtensor = Subtensor(network="ws://localhost:9945")

    # Register Bob to the network
    assert subtensor.burned_register(
        bob_wallet, netuid
    ), "Unable to register Bob as a neuron"

    metagraph = Metagraph(netuid=netuid, network="ws://localhost:9945")

    # Assert one neuron is Bob
    assert len(subtensor.neurons(netuid=netuid)) == 1
    neuron = metagraph.neurons[0]
    assert neuron.hotkey == bob_keypair.ss58_address
    assert neuron.coldkey == bob_keypair.ss58_address

    # Assert stake is 0
    assert neuron.stake.tao == 0

    # Stake to become to top neuron after the first epoch
    assert add_stake(local_chain, bob_wallet, Balance.from_tao(10_000))

    # Refresh metagraph
    metagraph = Metagraph(netuid=netuid, network="ws://localhost:9945")
    old_neuron = metagraph.neurons[0]

    # Assert stake is 10000
    assert (
        old_neuron.stake.tao == 10_000.0
    ), f"Expected 10_000.0 staked TAO, but got {neuron.stake.tao}"

    # Assert neuron is not a validator yet
    assert old_neuron.active is True
    assert old_neuron.validator_permit is False
    assert old_neuron.validator_trust == 0.0
    assert old_neuron.pruning_score == 0

    # Prepare to run the validator
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
            bob_wallet.path,
            "--wallet.name",
            bob_wallet.name,
            "--wallet.hotkey",
            "default",
        ]
    )

    # Run the validator in the background
    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    logging.console.info("Neuron Alice is now validating")
    await asyncio.sleep(
        5
    )  # wait for 5 seconds for the metagraph and subtensor to refresh with latest data

    await wait_epoch(subtensor, netuid=netuid)

    # Refresh metagraph
    metagraph = Metagraph(netuid=netuid, network="ws://localhost:9945")

    # Refresh validator neuron
    updated_neuron = metagraph.neurons[0]

    assert len(metagraph.neurons) == 1
    assert updated_neuron.active is True
    assert updated_neuron.validator_permit is True
    assert updated_neuron.hotkey == bob_keypair.ss58_address
    assert updated_neuron.coldkey == bob_keypair.ss58_address
    assert updated_neuron.pruning_score != 0

    logging.console.info("✅ Passed test_dendrite")
