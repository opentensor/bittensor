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
    logging.info("Testing neuron_certificate")
    netuid = 2

    # Register root as Alice - the subnet owner and validator
    assert subtensor.register_subnet(alice_wallet)

    # Verify subnet <netuid> created successfully
    assert subtensor.subnet_exists(netuid), "Subnet wasn't created successfully"

    # Register Alice as a neuron on the subnet
    assert subtensor.burned_register(
        alice_wallet, netuid
    ), "Unable to register Alice as a neuron"

    # Serve Alice's axon with a certificate
    axon = Axon(wallet=alice_wallet)
    encoded_certificate = "?FAKE_ALICE_CERT"
    subtensor.serve_axon(
        netuid,
        axon,
        certificate=encoded_certificate,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Verify we are getting the correct certificate
    assert (
        subtensor.get_neuron_certificate(
            netuid=netuid,
            hotkey=alice_wallet.hotkey.ss58_address,
        )
        == encoded_certificate
    )

    logging.info("✅ Passed test_neuron_certificate")
