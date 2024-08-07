import asyncio
import sys

import pytest

import bittensor
from bittensor import logging
from bittensor.commands import (
    RegisterCommand,
    RegisterSubnetworkCommand,
    RootRegisterCommand,
    RootSetBoostCommand,
    StakeCommand,
)
from tests.e2e_tests.utils import (
    setup_wallet,
    template_path,
    templates_repo,
    wait_epoch,
)

"""
Test the dendrites mechanism. 

Verify that:
* dendrite is registered on network as a validator
* stake successfully 
* validator permit is set

"""


@pytest.mark.asyncio
async def test_dendrite(local_chain):
    logging.info("Testing test_dendrite")
    netuid = 1
    # Register root as Alice - the subnet owner
    alice_keypair, exec_command, wallet = setup_wallet("//Alice")
    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # Verify subnet <netuid> created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [netuid]
    ).serialize(), "Subnet wasn't created successfully"

    bob_keypair, exec_command, wallet_path = setup_wallet("//Bob")

    # Register a neuron to the subnet
    exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            str(netuid),
        ],
    )

    metagraph = bittensor.metagraph(netuid=netuid, network="ws://localhost:9945")
    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    # assert one neuron is Bob
    assert len(subtensor.neurons(netuid=netuid)) == 1
    neuron = metagraph.neurons[0]
    assert neuron.hotkey == bob_keypair.ss58_address
    assert neuron.coldkey == bob_keypair.ss58_address

    # assert stake is 0
    assert neuron.stake.tao == 0

    # Stake to become to top neuron after the first epoch
    exec_command(
        StakeCommand,
        [
            "stake",
            "add",
            "--amount",
            "10000",
        ],
    )

    # refresh metagraph
    metagraph = bittensor.metagraph(netuid=netuid, network="ws://localhost:9945")
    neuron = metagraph.neurons[0]
    # assert stake is 10000
    assert (
        neuron.stake.tao == 10_000.0
    ), f"Expected 10_000.0 staked TAO, but got {neuron.stake.tao}"

    # assert neuron is not validator
    assert neuron.active is True
    assert neuron.validator_permit is False
    assert neuron.validator_trust == 0.0
    assert neuron.pruning_score == 0

    # register validator from template
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/neurons/validator.py"',
            "--no_prompt",
            "--netuid",
            str(netuid),
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--wallet.path",
            wallet.path,
            "--wallet.name",
            wallet.name,
            "--wallet.hotkey",
            "default",
        ]
    )

    # run validator in the background
    dendrite_process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    logging.info("Neuron Alice is now validating")
    await asyncio.sleep(
        5
    )  # wait for 5 seconds for the metagraph and subtensor to refresh with latest data

    # register validator with root network
    exec_command(
        RootRegisterCommand,
        [
            "root",
            "register",
            "--netuid",
            str(netuid),
        ],
    )

    exec_command(
        RootSetBoostCommand,
        [
            "root",
            "boost",
            "--netuid",
            str(netuid),
            "--increase",
            "1",
        ],
    )
    # get current block, wait until next epoch
    await wait_epoch(subtensor, netuid=netuid)

    # refresh metagraph
    metagraph = bittensor.metagraph(netuid=netuid, network="ws://localhost:9945")

    # refresh validator neuron
    neuron = metagraph.neurons[0]

    assert len(metagraph.neurons) == 1
    assert neuron.active is True
    assert neuron.validator_permit is True
    assert neuron.hotkey == bob_keypair.ss58_address
    assert neuron.coldkey == bob_keypair.ss58_address
    logging.info("Passed test_dendrite")
