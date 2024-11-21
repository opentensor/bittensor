import pytest
from bittensor.core.subtensor import Subtensor
from bittensor.core.axon import Axon
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.chain_interactions import (
    wait_interval,
    register_subnet,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    setup_wallet,
)


@pytest.mark.asyncio
async def test_neuron_certificate(local_chain):
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
    netuid = 1

    # Register root as Alice - the subnet owner and validator
    alice_keypair, alice_wallet = setup_wallet("//Alice")
    register_subnet(local_chain, alice_wallet)

    # Verify subnet <netuid> created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [netuid]
    ).serialize(), "Subnet wasn't created successfully"

    subtensor = Subtensor(network="ws://localhost:9945")

    # Register Alice as a neuron on the subnet
    assert subtensor.burned_register(
        alice_wallet, netuid
    ), "Unable to register Alice as a neuron"

    # Serve Alice's axon with a certificate
    axon = Axon(wallet=alice_wallet)
    encoded_certificate = "?FAKE_ALICE_CERT"
    axon.serve(netuid=netuid, subtensor=subtensor, certificate=encoded_certificate)

    await wait_interval(tempo=1, subtensor=subtensor, netuid=netuid)

    # Verify we are getting the correct certificate
    assert (
        subtensor.get_neuron_certificate(
            netuid=netuid, hotkey=alice_keypair.ss58_address
        )
        == encoded_certificate
    )

    logging.info("✅ Passed test_neuron_certificate")
