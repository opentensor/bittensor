import os.path
import shutil
import time

from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging


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


def test_metagraph(subtensor, alice_wallet, bob_wallet, dave_wallet):
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
    logging.console.info("Testing test_metagraph_command")
    netuid = 2

    # Register the subnet through Alice
    assert subtensor.register_subnet(alice_wallet), "Unable to register the subnet"

    # Verify subnet was created successfully
    assert subtensor.subnet_exists(netuid), "Subnet wasn't created successfully"

    # Initialize metagraph
    metagraph = subtensor.metagraph(netuid=netuid)

    # Assert metagraph has only Alice (owner)
    assert len(metagraph.uids) == 1, "Metagraph doesn't have exactly 1 neuron"

    # Register Bob to the subnet
    assert subtensor.burned_register(
        bob_wallet, netuid
    ), "Unable to register Bob as a neuron"

    # Refresh the metagraph
    metagraph.sync(subtensor=subtensor)

    # Assert metagraph has Alice and Bob neurons
    assert len(metagraph.uids) == 2, "Metagraph doesn't have exactly 2 neurons"
    assert (
        metagraph.hotkeys[0] == alice_wallet.hotkey.ss58_address
    ), "Alice's hotkey doesn't match in metagraph"
    assert (
        metagraph.hotkeys[1] == bob_wallet.hotkey.ss58_address
    ), "Bob's hotkey doesn't match in metagraph"
    assert len(metagraph.coldkeys) == 2, "Metagraph doesn't have exactly 2 coldkey"
    assert metagraph.n.max() == 2, "Metagraph's max n is not 2"
    assert metagraph.n.min() == 2, "Metagraph's min n is not 2"
    assert len(metagraph.addresses) == 2, "Metagraph doesn't have exactly 2 address"

    # Fetch UID of Bob
    uid = subtensor.get_uid_for_hotkey_on_subnet(
        bob_wallet.hotkey.ss58_address, netuid=netuid
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
    assert subtensor.burned_register(
        dave_wallet, netuid
    ), "Unable to register Dave as a neuron"

    metagraph.sync(subtensor=subtensor)

    # Assert metagraph now includes Dave's neuron
    assert (
        len(metagraph.uids) == 3
    ), "Metagraph doesn't have exactly 3 neurons post Dave"
    assert (
        metagraph.hotkeys[2] == dave_wallet.hotkey.ss58_address
    ), "Neuron's hotkey in metagraph doesn't match"
    assert (
        len(metagraph.coldkeys) == 3
    ), "Metagraph doesn't have exactly 3 coldkeys post Dave"
    assert metagraph.n.max() == 3, "Metagraph's max n is not 3 post Dave"
    assert metagraph.n.min() == 3, "Metagraph's min n is not 3 post Dave"
    assert len(metagraph.addresses) == 3, "Metagraph doesn't have 3 addresses post Dave"

    # Add stake by Bob
    tao = Balance.from_tao(10_000)
    alpha, _ = subtensor.subnet(netuid).tao_to_alpha_with_slippage(tao)
    assert subtensor.add_stake(
        bob_wallet,
        netuid=netuid,
        amount=tao,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    ), "Failed to add stake for Bob"

    # Assert stake is added after updating metagraph
    metagraph.sync(subtensor=subtensor)
    assert (
        0.95 < metagraph.neurons[1].stake.rao / alpha.rao < 1.05
    ), "Bob's stake not updated in metagraph"

    # Test the save() and load() mechanism
    # We save the metagraph and pre_dave loads it
    # We do this in the /tmp dir to avoid interfering or interacting with user data
    metagraph_save_root_dir = ["/", "tmp", "bittensor-e2e", "metagraphs"]
    try:
        os.makedirs(os.path.join(*metagraph_save_root_dir), exist_ok=True)
        metagraph.save(root_dir=metagraph_save_root_dir)
        time.sleep(3)
        metagraph_pre_dave.load(root_dir=metagraph_save_root_dir)
    finally:
        shutil.rmtree(os.path.join(*metagraph_save_root_dir))

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
        metagraph.axons[1].hotkey == metagraph_pre_dave.axons[1].hotkey
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

    logging.console.info("✅ Passed test_metagraph")
