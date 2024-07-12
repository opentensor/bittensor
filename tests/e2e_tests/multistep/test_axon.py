import asyncio
import sys

import pytest

import bittensor
from bittensor.utils import networking
from bittensor.commands import (
    RegisterCommand,
    RegisterSubnetworkCommand,
)
from tests.e2e_tests.utils import (
    setup_wallet,
    template_path,
    repo_name,
)

"""
Test the axon mechanism. 

Verify that:
* axon is registered on network as a miner
* ip
* type
* port

are set correctly, and that the miner is currently running

"""


@pytest.mark.asyncio
async def test_axon(local_chain):
    # Register root as Alice
    alice_keypair, exec_command, wallet = setup_wallet("//Alice")
    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

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

    metagraph = bittensor.metagraph(netuid=1, network="ws://localhost:9945")

    # validate one miner with ip of none
    old_axon = metagraph.axons[0]

    assert len(metagraph.axons) == 1
    assert old_axon.hotkey == alice_keypair.ss58_address
    assert old_axon.coldkey == alice_keypair.ss58_address
    assert old_axon.ip == "0.0.0.0"
    assert old_axon.port == 0
    assert old_axon.ip_type == 0

    # register miner
    # "python neurons/miner.py --netuid 1 --subtensor.chain_endpoint ws://localhost:9945 --wallet.name wallet.name --wallet.hotkey wallet.hotkey.ss58_address"
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{repo_name}/neurons/miner.py"',
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

    axon_process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    await asyncio.sleep(
        5
    )  # wait for 5 seconds for the metagraph to refresh with latest data

    # refresh metagraph
    metagraph = bittensor.metagraph(netuid=1, network="ws://localhost:9945")
    updated_axon = metagraph.axons[0]
    external_ip = networking.get_external_ip()

    assert len(metagraph.axons) == 1
    assert updated_axon.ip == external_ip
    assert updated_axon.ip_type == networking.ip_version(external_ip)
    assert updated_axon.port == 8091
    assert updated_axon.hotkey == alice_keypair.ss58_address
    assert updated_axon.coldkey == alice_keypair.ss58_address
