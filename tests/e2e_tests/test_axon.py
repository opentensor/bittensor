import asyncio
import sys

import pytest

from bittensor.utils import networking
from tests.e2e_tests.utils.e2e_test_utils import (
    template_path,
    templates_repo,
)


@pytest.mark.asyncio
async def test_axon(subtensor, alice_wallet):
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

    print("Testing test_axon")

    netuid = 2

    # Register a subnet, netuid 2
    assert subtensor.register_subnet(alice_wallet), "Subnet wasn't created"

    # Verify subnet <netuid> created successfully
    assert subtensor.subnet_exists(netuid), "Subnet wasn't created successfully"

    metagraph = subtensor.metagraph(netuid)

    # Validate current metagraph stats
    old_axon = metagraph.axons[0]
    assert len(metagraph.axons) == 1, f"Expected 1 axon, but got {len(metagraph.axons)}"
    assert (
        old_axon.hotkey == alice_wallet.hotkey.ss58_address
    ), "Hotkey mismatch for the axon"
    assert (
        old_axon.coldkey == alice_wallet.coldkey.ss58_address
    ), "Coldkey mismatch for the axon"
    assert old_axon.ip == "0.0.0.0", f"Expected IP 0.0.0.0, but got {old_axon.ip}"
    assert old_axon.port == 0, f"Expected port 0, but got {old_axon.port}"
    assert old_axon.ip_type == 0, f"Expected IP type 0, but got {old_axon.ip_type}"

    # Prepare to run the miner
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/miner.py"',
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
        ]
    )

    # Run the miner in the background
    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    print("Neuron Alice is now mining")

    # Waiting for 5 seconds for metagraph to be updated
    await asyncio.sleep(5)

    # Refresh the metagraph
    metagraph = subtensor.metagraph(netuid)
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
        updated_axon.hotkey == alice_wallet.hotkey.ss58_address
    ), "Hotkey mismatch after mining"

    assert (
        updated_axon.coldkey == alice_wallet.coldkey.ss58_address
    ), "Coldkey mismatch after mining"

    print("✅ Passed test_axon")
