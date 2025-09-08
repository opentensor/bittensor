import pytest
from bittensor.core.axon import Axon
from bittensor.utils.btlogging import logging


@pytest.mark.asyncio
async def test_neuron_certificate(subtensor, alice_wallet):
    """
    Tests the metagraph

    Steps:
        1. Register a subnet through Alice
        2. Serve Alice axon with neuron certificate
        3. Verify neuron certificate can be retrieved
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    logging.info("Testing [blue]neuron_certificate[/blue]")
    netuid = 2

    # Register root as Alice - the subnet owner and validator
    assert subtensor.subnets.register_subnet(alice_wallet)

    # Verify subnet <netuid> created successfully
    assert subtensor.subnets.subnet_exists(netuid), "Subnet wasn't created successfully"

    # Register Alice as a neuron on the subnet
    assert subtensor.subnets.burned_register(alice_wallet, netuid).success, (
        "Unable to register Alice as a neuron"
    )

    # Serve Alice's axon with a certificate
    axon = Axon(wallet=alice_wallet)
    encoded_certificate = "?FAKE_ALICE_CERT"
    subtensor.extrinsics.serve_axon(
        netuid,
        axon,
        certificate=encoded_certificate,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Verify we are getting the correct certificate
    assert (
        subtensor.neurons.get_neuron_certificate(
            netuid=netuid,
            hotkey=alice_wallet.hotkey.ss58_address,
        )
        == encoded_certificate
    )
    all_certs_query = subtensor.neurons.get_all_neuron_certificates(netuid=netuid)
    assert alice_wallet.hotkey.ss58_address in all_certs_query.keys()
    assert all_certs_query[alice_wallet.hotkey.ss58_address] == encoded_certificate

    logging.console.success("✅ Passed [blue]test_neuron_certificate[/blue]")


@pytest.mark.asyncio
async def test_neuron_certificate_async(async_subtensor, alice_wallet):
    """
    ASync tests the metagraph

    Steps:
        1. Register a subnet through Alice
        2. Serve Alice axon with neuron certificate
        3. Verify neuron certificate can be retrieved
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    logging.info("Testing [blue]neuron_certificate[/blue]")
    netuid = 2

    # Register root as Alice - the subnet owner and validator
    assert await async_subtensor.subnets.register_subnet(alice_wallet)

    # Verify subnet <netuid> created successfully
    assert await async_subtensor.subnets.subnet_exists(netuid), (
        "Subnet wasn't created successfully"
    )

    # Register Alice as a neuron on the subnet
    assert (
        await async_subtensor.subnets.burned_register(alice_wallet, netuid)
    ).success, "Unable to register Alice as a neuron"

    # Serve Alice's axon with a certificate
    axon = Axon(wallet=alice_wallet)
    encoded_certificate = "?FAKE_ALICE_CERT"
    await async_subtensor.extrinsics.serve_axon(
        netuid,
        axon,
        certificate=encoded_certificate,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Verify we are getting the correct certificate
    assert (
        await async_subtensor.neurons.get_neuron_certificate(
            netuid=netuid,
            hotkey=alice_wallet.hotkey.ss58_address,
        )
        == encoded_certificate
    )
    all_certs_query = await async_subtensor.neurons.get_all_neuron_certificates(
        netuid=netuid
    )
    assert alice_wallet.hotkey.ss58_address in all_certs_query.keys()
    assert all_certs_query[alice_wallet.hotkey.ss58_address] == encoded_certificate

    logging.console.success("✅ Passed [blue]test_neuron_certificate[/blue]")
