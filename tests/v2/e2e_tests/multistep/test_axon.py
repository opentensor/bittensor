import asyncio
import sys

import pytest
import bittensor.v2 as bittensor
from bittensor.v2.utils import networking
from bittensor.v2.commands import (
    RegisterCommand,
    RegisterSubnetworkCommand,
)
from tests.v2.e2e_tests.utils import (
    setup_wallet,
    template_path,
    templates_repo,
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
    alice_keypair, exec_command, wallet = await setup_wallet("//Alice")
    await exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

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
    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    metagraph = await bittensor.metagraph(
        netuid=1, network="ws://localhost:9945", subtensor=subtensor
    )

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
            f'"{template_path}{templates_repo}/neurons/miner.py"',
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

    # wait for 5 seconds for the metagraph to refresh with latest data
    await asyncio.sleep(5)

    # refresh metagraph
    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    metagraph = await bittensor.metagraph(
        netuid=1, network="ws://localhost:9945", sync=True, subtensor=subtensor
    )
    updated_axon = metagraph.axons[0]
    external_ip = networking.get_external_ip()

    await asyncio.sleep(5)

    assert len(metagraph.axons) == 1
    assert updated_axon.ip == external_ip
    assert updated_axon.ip_type == networking.ip_version(external_ip)
    assert updated_axon.port == 8091
    assert updated_axon.hotkey == alice_keypair.ss58_address
    assert updated_axon.coldkey == alice_keypair.ss58_address