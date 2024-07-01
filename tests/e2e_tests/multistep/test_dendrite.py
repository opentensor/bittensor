import asyncio
import logging
import sys

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
    templates_repo,
    wait_interval,
    write_output_log_to_file,
)


logging.basicConfig(level=logging.INFO)

"""
Test the dendrites mechanism.

Verify that:
* dendrite is registered on network as a validator
* stake successfully
* validator permit is set

"""


@pytest.mark.asyncio
async def test_dendrite(local_chain):
    # Register root as Alice - the subnet owner
    alice_keypair, exec_command, wallet = await setup_wallet("//Alice")
    await exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    bob_keypair, exec_command, wallet_path = await setup_wallet("//Bob")

    # Register a neuron to the subnet
    await exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
        ],
    )
    metagraph = await bittensor.metagraph(netuid=1, network="ws://localhost:9945")
    neuron = metagraph.neurons[0]

    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    # assert one neuron is Bob
    assert len(await subtensor.neurons(netuid=1)) == 1

    assert neuron.hotkey == bob_keypair.ss58_address
    assert neuron.coldkey == bob_keypair.ss58_address

    # assert stake is 0
    assert neuron.stake.tao == 0

    # Stake to become to top neuron after the first epoch
    await exec_command(
        StakeCommand,
        [
            "stake",
            "add",
            "--amount",
            "10000",
        ],
    )

    # refresh metagraph
    metagraph = await bittensor.metagraph(
        netuid=1, network="ws://localhost:9945", sync=True
    )
    neuron = metagraph.neurons[0]
    # assert stake is 10000
    assert neuron.stake.tao == 10_000.0

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
            "1",
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

    # TODO: remove `write_output_log_to_file` logging after async migration done
    # record logs of process
    # Create tasks to read stdout and stderr concurrently
    # ignore, dont await coroutine, just write logs to file
    asyncio.create_task(
        write_output_log_to_file("dendrite_stdout", dendrite_process.stdout)
    )
    # ignore, dont await coroutine, just write logs to file
    asyncio.create_task(
        write_output_log_to_file("dendrite_stderr", dendrite_process.stderr)
    )

    await asyncio.sleep(
        5
    )  # wait for 5 seconds for the metagraph and subtensor to refresh with latest data

    # register validator with root network
    await exec_command(
        RootRegisterCommand,
        [
            "root",
            "register",
            "--netuid",
            "1",
        ],
    )

    await exec_command(
        RootSetBoostCommand,
        [
            "root",
            "boost",
            "--netuid",
            "1",
            "--increase",
            "1",
        ],
    )
    # get current block, wait until 360 blocks pass (subnet tempo)
    await wait_interval(360, subtensor)

    # refresh metagraph
    metagraph = await bittensor.metagraph(
        netuid=1, network="ws://localhost:9945", sync=True
    )
    # refresh validator neuron
    neuron = metagraph.neurons[0]
    assert len(metagraph.neurons) == 1
    assert neuron.active is True
    assert neuron.validator_permit is True
    assert neuron.hotkey == bob_keypair.ss58_address
    assert neuron.coldkey == bob_keypair.ss58_address
