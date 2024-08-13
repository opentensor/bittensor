import time

import bittensor
from bittensor import logging
from tests.e2e_tests.utils.chain_interactions import (
    add_stake,
    register_neuron,
    register_subnet,
)
from tests.e2e_tests.utils.test_utils import (
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
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    logging.info("Testing test_metagraph_command")
    netuid = 1

    # Register Alice, Bob, and Dave
    alice_keypair, alice_wallet = setup_wallet("//Alice")
    bob_keypair, bob_wallet = setup_wallet("//Bob")
    dave_keypair, dave_wallet = setup_wallet("//Dave")

    # Register the subnet through Alice
    register_subnet(local_chain, alice_wallet), "Unable to register the subnet"

    # Verify subnet was created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [1]
    ).serialize(), "Subnet wasn't created successfully"

    # Initialize metagraph
    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    metagraph = subtensor.metagraph(netuid=1)

    # Assert metagraph is empty
    assert len(metagraph.uids) == 0, "Metagraph is not empty"

    # Register Bob to the subnet
    assert register_neuron(
        local_chain, bob_wallet, netuid
    ), "Unable to register Bob as a neuron"

    # Refresh the metagraph
    metagraph.sync(subtensor=subtensor)

    # Assert metagraph has Bob neuron
    assert len(metagraph.uids) == 1, "Metagraph doesn't have exactly 1 neuron"
    assert metagraph.hotkeys[0] == bob_keypair.ss58_address
    assert len(metagraph.coldkeys) == 1
    assert metagraph.n.max() == 1
    assert metagraph.n.min() == 1
    assert len(metagraph.addresses) == 1

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
    metagraph_pre_dave = subtensor.metagraph(netuid=1)

    # Register Dave as a neuron
    assert register_neuron(
        local_chain, dave_wallet, netuid
    ), "Unable to register Dave as a neuron"

    metagraph.sync(subtensor=subtensor)

    # Assert metagraph now includes Dave's neuron
    assert len(metagraph.uids) == 2
    assert (
        metagraph.hotkeys[1] == dave_keypair.ss58_address
    ), "Neuron's hotkey in metagraph doesn't match"
    assert len(metagraph.coldkeys) == 2
    assert metagraph.n.max() == 2
    assert metagraph.n.min() == 2
    assert len(metagraph.addresses) == 2

    # Test staking with low balance
    assert not add_stake(local_chain, dave_wallet, bittensor.Balance.from_tao(10_000))

    # Add stake by Bob
    assert add_stake(local_chain, bob_wallet, bittensor.Balance.from_tao(10_000))

    # Assert stake is added after updating metagraph
    metagraph.sync(subtensor=subtensor)
    assert metagraph.neurons[0].stake == bittensor.Balance.from_tao(10_000)

    # Test the save() and load() mechanism
    # We save the metagraph and pre_dave loads it
    metagraph.save()
    time.sleep(3)
    metagraph_pre_dave.load()

    # Ensure data is synced between two metagraphs
    assert len(metagraph.uids) == len(metagraph_pre_dave.uids)
    assert (metagraph.uids == metagraph_pre_dave.uids).all()

    assert len(metagraph.axons) == len(metagraph_pre_dave.axons)
    assert metagraph.axons[1].hotkey == metagraph_pre_dave.axons[1].hotkey
    assert metagraph.axons == metagraph_pre_dave.axons

    assert len(metagraph.neurons) == len(metagraph_pre_dave.neurons)
    assert metagraph.neurons == metagraph_pre_dave.neurons

    logging.info("✅ Passed test_metagraph")
