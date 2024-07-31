import asyncio
import sys

import pytest

import bittensor
from bittensor import logging
from bittensor.utils import networking
from tests.e2e_tests.utils import (
    register_neuron,
    register_subnet,
    setup_wallet,
    template_path,
    templates_repo,
)


@pytest.mark.asyncio
async def test_axon(local_chain):
    """
    Test the Axon mechanism and successful registration.

    Steps:
        1. Register a subnet and register Alice
        2. Check if metagraph updated and check axon attributes
        3. Run Alice as a miner on the subnet
        4. Check the metagraph again after running the miner and verify all attreibutes
    Raises:
        AssertionError: If any of the checks or verifications fail
    """

    logging.info("Testing test_axon")

    netuid = 1
    # Register root as Alice
    alice_keypair, exec_command, wallet = setup_wallet("//Alice")

    assert register_subnet(local_chain, wallet), "Subnet wasn't created"
    # Verify subnet <netuid> created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [netuid]
    ).serialize(), "Subnet wasn't created successfully"

    assert register_neuron(
        local_chain, wallet, netuid
    ), f"Neuron wasn't registered to subnet {netuid}"

    metagraph = bittensor.metagraph(netuid=netuid, network="ws://localhost:9945")

    # validate one miner with ip of none
    old_axon = metagraph.axons[0]

    assert len(metagraph.axons) == 1, "Expected 1 axon, but got len(metagraph.axons)"
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

    axon_process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    logging.info("Neuron Alice is now mining")
    await asyncio.sleep(
        5
    )  # wait for 5 seconds for the metagraph to refresh with latest data

    # refresh metagraph
    metagraph = bittensor.metagraph(netuid=netuid, network="ws://localhost:9945")
    updated_axon = metagraph.axons[0]
    external_ip = networking.get_external_ip()

    assert len(metagraph.axons) == 1
    assert updated_axon.ip == external_ip
    assert updated_axon.ip_type == networking.ip_version(external_ip)
    assert updated_axon.port == 8091
    assert updated_axon.hotkey == alice_keypair.ss58_address
    assert updated_axon.coldkey == alice_keypair.ss58_address
    logging.info("Passed test_axon")
