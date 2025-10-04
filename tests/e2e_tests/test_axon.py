import asyncio

import pytest

from bittensor.utils import networking
from tests.e2e_tests.utils import (
    TestSubnet,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
)


@pytest.mark.asyncio
async def test_axon(subtensor, templates, alice_wallet):
    """
    Test the Axon mechanism and successful registration on the network with sync Subtensor.

    Steps:
        1. Register a subnet and register Alice
        2. Check if metagraph.axon is updated and check axon attributes
        3. Run Alice as a miner on subnet
        4. Check the metagraph again after running the miner and verify all attributes
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    alice_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
    ]
    alice_sn.execute_steps(steps)

    metagraph = subtensor.metagraphs.metagraph(alice_sn.netuid)

    # Validate current metagraph stats
    old_axon = metagraph.axons[0]
    assert len(metagraph.axons) == 1, f"Expected 1 axon, but got {len(metagraph.axons)}"
    assert old_axon.hotkey == alice_wallet.hotkey.ss58_address, (
        "Hotkey mismatch for the axon"
    )
    assert old_axon.coldkey == alice_wallet.coldkey.ss58_address, (
        "Coldkey mismatch for the axon"
    )
    assert old_axon.ip == "0.0.0.0", f"Expected IP 0.0.0.0, but got {old_axon.ip}"
    assert old_axon.port == 0, f"Expected port 0, but got {old_axon.port}"
    assert old_axon.ip_type == 0, f"Expected IP type 0, but got {old_axon.ip_type}"

    async with templates.miner(alice_wallet, alice_sn.netuid):
        # Waiting for 5 seconds for metagraph to be updated
        await asyncio.sleep(5)

        # Refresh the metagraph
        metagraph = subtensor.metagraphs.metagraph(alice_sn.netuid)
        updated_axon = metagraph.axons[0]
        external_ip = networking.get_external_ip()

    # Assert updated attributes
    assert len(metagraph.axons) == 1, (
        f"Expected 1 axon, but got {len(metagraph.axons)} after mining"
    )

    assert len(metagraph.neurons) == 1, (
        f"Expected 1 neuron, but got {len(metagraph.neurons)}"
    )

    assert updated_axon.ip == external_ip, (
        f"Expected IP {external_ip}, but got {updated_axon.ip}"
    )

    assert updated_axon.ip_type == networking.ip_version(external_ip), (
        f"Expected IP type {networking.ip_version(external_ip)}, but got {updated_axon.ip_type}"
    )

    assert updated_axon.port == 8091, f"Expected port 8091, but got {updated_axon.port}"

    assert updated_axon.hotkey == alice_wallet.hotkey.ss58_address, (
        "Hotkey mismatch after mining"
    )

    assert updated_axon.coldkey == alice_wallet.coldkey.ss58_address, (
        "Coldkey mismatch after mining"
    )


@pytest.mark.asyncio
async def test_axon_async(async_subtensor, templates, alice_wallet):
    """
    Test the Axon mechanism and successful registration on the network with async Subtensor.

    Steps:
        1. Register a subnet and register Alice
        2. Check if metagraph.axon is updated and check axon attributes
        3. Run Alice as a miner on subnet
        4. Check the metagraph again after running the miner and verify all attributes
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    metagraph = await async_subtensor.metagraphs.metagraph(alice_sn.netuid)

    # Validate current metagraph stats
    old_axon = metagraph.axons[0]
    assert len(metagraph.axons) == 1, f"Expected 1 axon, but got {len(metagraph.axons)}"
    assert old_axon.hotkey == alice_wallet.hotkey.ss58_address, (
        "Hotkey mismatch for the axon"
    )
    assert old_axon.coldkey == alice_wallet.coldkey.ss58_address, (
        "Coldkey mismatch for the axon"
    )
    assert old_axon.ip == "0.0.0.0", f"Expected IP 0.0.0.0, but got {old_axon.ip}"
    assert old_axon.port == 0, f"Expected port 0, but got {old_axon.port}"
    assert old_axon.ip_type == 0, f"Expected IP type 0, but got {old_axon.ip_type}"

    async with templates.miner(alice_wallet, alice_sn.netuid):
        # Waiting for 5 seconds for metagraph to be updated
        await asyncio.sleep(5)

        # Refresh the metagraph
        metagraph = await async_subtensor.metagraphs.metagraph(alice_sn.netuid)
        updated_axon = metagraph.axons[0]
        external_ip = networking.get_external_ip()

    # Assert updated attributes
    assert len(metagraph.axons) == 1, (
        f"Expected 1 axon, but got {len(metagraph.axons)} after mining"
    )

    assert len(metagraph.neurons) == 1, (
        f"Expected 1 neuron, but got {len(metagraph.neurons)}"
    )

    assert updated_axon.ip == external_ip, (
        f"Expected IP {external_ip}, but got {updated_axon.ip}"
    )

    assert updated_axon.ip_type == networking.ip_version(external_ip), (
        f"Expected IP type {networking.ip_version(external_ip)}, but got {updated_axon.ip_type}"
    )

    assert updated_axon.port == 8091, f"Expected port 8091, but got {updated_axon.port}"

    assert updated_axon.hotkey == alice_wallet.hotkey.ss58_address, (
        "Hotkey mismatch after mining"
    )

    assert updated_axon.coldkey == alice_wallet.coldkey.ss58_address, (
        "Coldkey mismatch after mining"
    )
