import asyncio
import sys

import pytest

import bittensor
from bittensor import logging
from bittensor.utils import networking
from tests.e2e_tests.utils.chain_interactions import register_neuron, register_subnet
from tests.e2e_tests.utils.e2e_test_utils import (
    setup_wallet,
    template_path,
    templates_repo,
)


@pytest.mark.asyncio
async def test_axon(local_chain):
    """
    Test the Axon mechanism and successful registration on the network.

    Steps:
        1. Register a subnet and register Alice
        2. Check if metagraph.axon is updated and check axon attributes
        3. Run Alice as a miner on the subnet
        4. Check the metagraph again after running the miner and verify all attributes
    Raises:
        AssertionError: If any of the checks or verifications fail
    """

    logging.info("Testing test_axon")

    netuid = 1
    # Register root as Alice - the subnet owner
    alice_keypair, wallet = setup_wallet("//Alice")

    # Register a subnet, netuid 1
    assert register_subnet(local_chain, wallet), "Subnet wasn't created"

    # Verify subnet <netuid 1> created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [netuid]
    ).serialize(), "Subnet wasn't created successfully"

    # Register Alice to the network
    assert register_neuron(
        local_chain, wallet, netuid
    ), f"Neuron wasn't registered to subnet {netuid}"

    metagraph = bittensor.Metagraph(netuid=netuid, network="ws://localhost:9945")

    # Validate current metagraph stats
    old_axon = metagraph.axons[0]
    assert len(metagraph.axons) == 1, f"Expected 1 axon, but got {len(metagraph.axons)}"
    assert old_axon.hotkey == alice_keypair.ss58_address, "Hotkey mismatch for the axon"
    assert (
        old_axon.coldkey == alice_keypair.ss58_address
    ), "Coldkey mismatch for the axon"
    assert old_axon.ip == "0.0.0.0", f"Expected IP 0.0.0.0, but got {old_axon.ip}"
    assert old_axon.port == 0, f"Expected port 0, but got {old_axon.port}"
    assert old_axon.ip_type == 0, f"Expected IP type 0, but got {old_axon.ip_type}"

    # Prepare to run the miner
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

    # Run the miner in the background
    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    logging.info("Neuron Alice is now mining")

    # Waiting for 5 seconds for metagraph to be updated
    await asyncio.sleep(5)

    # Refresh the metagraph
    metagraph = bittensor.Metagraph(netuid=netuid, network="ws://localhost:9945")
    updated_axon = metagraph.axons[0]
    external_ip = networking.get_external_ip()

    # Assert updated attributes
    assert (
        len(metagraph.axons) == 1
    ), f"Expected 1 axon, but got {len(metagraph.axons)} after mining"

    assert (
        len(metagraph.neurons) == 1
    ), f"Expected 1 neuron, but got {len(metagraph.neurons)}"

    assert (
        updated_axon.ip == external_ip
    ), f"Expected IP {external_ip}, but got {updated_axon.ip}"

    assert (
        updated_axon.ip_type == networking.ip_version(external_ip)
    ), f"Expected IP type {networking.ip_version(external_ip)}, but got {updated_axon.ip_type}"

    assert updated_axon.port == 8091, f"Expected port 8091, but got {updated_axon.port}"

    assert (
        updated_axon.hotkey == alice_keypair.ss58_address
    ), "Hotkey mismatch after mining"

    assert (
        updated_axon.coldkey == alice_keypair.ss58_address
    ), "Coldkey mismatch after mining"

    logging.info("✅ Passed test_axon")
