import asyncio
import logging
import sys
import time

import pytest

import bittensor
from bittensor.commands import (
    RegisterCommand,
    RegisterSubnetworkCommand,
    StakeCommand,
    RootRegisterCommand,
    RootSetBoostCommand,
)
from tests.e2e_tests.utils import (
    setup_wallet,
    template_path,
    repo_name,
)

logging.basicConfig(level=logging.INFO)


@pytest.mark.asyncio
async def test_dendrite(local_chain):
    # Register root as Alice - the subnet owner
    alice_keypair, exec_command, wallet_path = setup_wallet("//Alice")
    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    bob_keypair, exec_command, wallet_path = setup_wallet("//Bob")

    # Register a neuron to the subnet
    exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
            "--wallet.name",
            "default",
            "--wallet.hotkey",
            "default",
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--no_prompt",
        ],
    )

    metagraph = bittensor.metagraph(netuid=1, network="ws://localhost:9945")
    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    # assert one neuron is Bob
    assert len(subtensor.neurons(netuid=1)) == 1
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
    metagraph = bittensor.metagraph(netuid=1, network="ws://localhost:9945")
    neuron = metagraph.neurons[0]
    # assert stake is 10000
    assert neuron.stake.tao == 9999.999999

    # assert neuron is not validator
    assert neuron.active is True
    assert neuron.validator_permit is False
    assert neuron.validator_trust == 0.0
    assert neuron.pruning_score == 0

    # register validator from template
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{repo_name}/neurons/validator.py"',
            "--no_prompt",
            "--netuid",
            "1",
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--wallet.path",
            wallet_path,
            "--wallet.name",
            "default",
            "--wallet.hotkey",
            "default",
        ]
    )

    # run validator in the background
    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
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
            "1",
            "--wallet.name",
            "default",
            "--wallet.hotkey",
            "default",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
        ],
    )

    exec_command(
        RootSetBoostCommand,
        [
            "root",
            "boost",
            "--netuid",
            "1",
            "--increase",
            "1",
            "--wallet.name",
            "default",
            "--wallet.hotkey",
            "default",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
        ],
    )
    # get current block, wait until 360 blocks pass (subnet tempo)
    interval = 360
    current_block = subtensor.get_current_block()
    next_tempo_block_start = (current_block - (current_block % interval)) + interval
    while current_block < next_tempo_block_start:
        time.sleep(1)  # Wait for 1 second before checking the block number again
        current_block = subtensor.get_current_block()
        if current_block % 10 == 0:
            print(
                f"Current Block: {current_block}  Next tempo at: {next_tempo_block_start}"
            )
            logging.info(
                f"Current Block: {current_block}  Next tempo at: {next_tempo_block_start}"
            )

    # refresh metagraph
    metagraph = bittensor.metagraph(netuid=1, network="ws://localhost:9945")

    # refresh validator neuron
    neuron = metagraph.neurons[0]

    assert len(metagraph.neurons) == 1
    assert neuron.active is True
    assert neuron.validator_permit is True
    assert neuron.hotkey == bob_keypair.ss58_address
    assert neuron.coldkey == bob_keypair.ss58_address
