import time

import bittensor
from bittensor import logging
from tests.e2e_tests.utils.chain_interactions import (
    add_stake,
    register_neuron,
    register_subnet,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    setup_wallet,
)


def neuron_to_dict(neuron):
    """
    Convert a neuron object to a dictionary, excluding private attributes, methods, and specific fields.
    Returns:
        dict: A dictionary of the neuron's public attributes.

    Note:
        Excludes 'weights' and 'bonds' fields. These are present in subtensor
        but not in metagraph
    """
    excluded_fields = {"weights", "bonds"}
    return {
        attr: getattr(neuron, attr)
        for attr in dir(neuron)
        if not attr.startswith("_")
        and not callable(getattr(neuron, attr))
        and attr not in excluded_fields
    }


def test_metagraph(local_chain):
    """
    Tests the metagraph

    Steps:
        1. Register a subnet through Alice
        2. Assert metagraph's initial state
        3. Register Bob and validate info in metagraph
        4. Fetch neuron info of Bob through subtensor & metagraph and verify
        5. Register Dave and validate info in metagraph
        6. Verify low balance stake fails & add stake thru Bob and verify
        7. Load pre_dave metagraph from latest save and verify both instances
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    logging.info("Testing test_metagraph_command")
    netuid = 2

    # Register Alice, Bob, and Dave
    alice_keypair, alice_wallet = setup_wallet("//Alice")
    bob_keypair, bob_wallet = setup_wallet("//Bob")
    dave_keypair, dave_wallet = setup_wallet("//Dave")

    # Register the subnet through Alice
    register_subnet(local_chain, alice_wallet), "Unable to register the subnet"

    # Verify subnet was created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [netuid]
    ).serialize(), "Subnet wasn't created successfully"

    # Initialize metagraph
    subtensor = bittensor.Subtensor(network="ws://localhost:9945")
    metagraph = subtensor.metagraph(netuid=netuid)

    # Assert metagraph is empty
    assert len(metagraph.uids) == 1, "Metagraph is not empty"

    # Register Bob to the subnet
    assert register_neuron(
        local_chain, bob_wallet, netuid
    ), "Unable to register Bob as a neuron"

    # Refresh the metagraph
    metagraph.sync(subtensor=subtensor)

    # Assert metagraph has Bob neuron
    assert len(metagraph.uids) == 2, "Metagraph doesn't have exactly 1 neuron"
    assert (
        metagraph.hotkeys[1] == bob_keypair.ss58_address
    ), "Bob's hotkey doesn't match in metagraph"
    assert len(metagraph.coldkeys) == 2, "Metagraph doesn't have exactly 1 coldkey"
    assert metagraph.n.max() == 2, "Metagraph's max n is not 1"
    assert metagraph.n.min() == 2, "Metagraph's min n is not 1"
    assert len(metagraph.addresses) == 2, "Metagraph doesn't have exactly 1 address"

    # Fetch UID of Bob
    uid = subtensor.get_uid_for_hotkey_on_subnet(
        bob_keypair.ss58_address, netuid=netuid
    )

    # Fetch neuron info of Bob through subtensor and metagraph
    neuron_info_bob = subtensor.neuron_for_uid(uid, netuid=netuid)
    metagraph_dict = neuron_to_dict(metagraph.neurons[uid])
    subtensor_dict = neuron_to_dict(neuron_info_bob)

    # Verify neuron info is the same in both objects
    assert (
        metagraph_dict == subtensor_dict
    ), "Neuron info of Bob doesn't match b/w metagraph & subtensor"

    # Create pre_dave metagraph for future verifications
    metagraph_pre_dave = subtensor.metagraph(netuid=netuid)

    # Register Dave as a neuron
    assert register_neuron(
        local_chain, dave_wallet, netuid
    ), "Unable to register Dave as a neuron"

    metagraph.sync(subtensor=subtensor)

    # Assert metagraph now includes Dave's neuron
    assert (
        len(metagraph.uids) == 3
    ), "Metagraph doesn't have exactly 2 neurons post Dave"
    assert (
        metagraph.hotkeys[2] == dave_keypair.ss58_address
    ), "Neuron's hotkey in metagraph doesn't match"
    assert (
        len(metagraph.coldkeys) == 3
    ), "Metagraph doesn't have exactly 2 coldkeys post Dave"
    assert metagraph.n.max() == 3, "Metagraph's max n is not 2 post Dave"
    assert metagraph.n.min() == 3, "Metagraph's min n is not 2 post Dave"
    assert len(metagraph.addresses) == 3, "Metagraph doesn't have 2 addresses post Dave"

    # Test staking with low balance
    assert not add_stake(
        local_chain, dave_wallet, netuid, bittensor.Balance.from_tao(10_000)
    ), "Low balance stake should fail"

    # Add stake by Bob
    assert add_stake(
        local_chain, bob_wallet, netuid, bittensor.Balance.from_tao(10_000)
    ), "Failed to add stake for Bob"

    # Assert stake is added after updating metagraph
    metagraph.sync(subtensor=subtensor)
    assert metagraph.neurons[1].stake == bittensor.Balance.from_tao(
        10_000
    ), "Bob's stake not updated in metagraph"

    # Test the save() and load() mechanism
    # We save the metagraph and pre_dave loads it
    metagraph.save()
    time.sleep(3)
    metagraph_pre_dave.load()

    # Ensure data is synced between two metagraphs
    assert len(metagraph.uids) == len(
        metagraph_pre_dave.uids
    ), "UID count mismatch after save and load"
    assert (
        metagraph.uids == metagraph_pre_dave.uids
    ).all(), "UIDs don't match after save and load"

    assert len(metagraph.axons) == len(
        metagraph_pre_dave.axons
    ), "Axon count mismatch after save and load"
    assert (
        metagraph.axons[2].hotkey == metagraph_pre_dave.axons[2].hotkey
    ), "Axon hotkey mismatch after save and load"
    assert (
        metagraph.axons == metagraph_pre_dave.axons
    ), "Axons don't match after save and load"

    assert len(metagraph.neurons) == len(
        metagraph_pre_dave.neurons
    ), "Neuron count mismatch after save and load"
    assert (
        metagraph.neurons == metagraph_pre_dave.neurons
    ), "Neurons don't match after save and load"

    logging.info("✅ Passed test_metagraph")
