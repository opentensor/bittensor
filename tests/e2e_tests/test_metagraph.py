import os.path
import re
import shutil
import time

from bittensor.core.chain_data.metagraph_info import MetagraphInfo
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

NULL_KEY = tuple(bytearray(32))


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
    alice_subnet_netuid = 2

    # Register the subnet through Alice
    assert subtensor.register_subnet(alice_wallet), "Unable to register the subnet"

    # Verify subnet was created successfully
    assert subtensor.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    # make sure we passed start_call limit (10 blocks)
    subtensor.wait_for_block(subtensor.block + 10)
    assert subtensor.start_call(alice_wallet, alice_subnet_netuid, True, True)[0]

    # Initialize metagraph
    metagraph = subtensor.metagraph(netuid=alice_subnet_netuid)

    # Assert metagraph has only Alice (owner)
    assert len(metagraph.uids) == 1, "Metagraph doesn't have exactly 1 neuron"

    # Register Bob to the subnet
    assert subtensor.burned_register(bob_wallet, alice_subnet_netuid), (
        "Unable to register Bob as a neuron"
    )

    # Refresh the metagraph
    metagraph.sync(subtensor=subtensor)

    # wait for updated information to arrive (important for low resource docker)
    subtensor.wait_for_block(subtensor.block + 10)

    # Assert metagraph has Alice and Bob neurons
    assert len(metagraph.uids) == 2, "Metagraph doesn't have exactly 2 neurons"
    assert metagraph.hotkeys[0] == alice_wallet.hotkey.ss58_address, (
        "Alice's hotkey doesn't match in metagraph"
    )
    assert metagraph.hotkeys[1] == bob_wallet.hotkey.ss58_address, (
        "Bob's hotkey doesn't match in metagraph"
    )
    assert len(metagraph.coldkeys) == 2, "Metagraph doesn't have exactly 2 coldkey"
    assert metagraph.n.max() == 2, "Metagraph's max n is not 2"
    assert metagraph.n.min() == 2, "Metagraph's min n is not 2"
    assert len(metagraph.addresses) == 2, "Metagraph doesn't have exactly 2 address"

    # Fetch UID of Bob
    uid = subtensor.get_uid_for_hotkey_on_subnet(
        bob_wallet.hotkey.ss58_address, netuid=alice_subnet_netuid
    )

    # Fetch neuron info of Bob through subtensor and metagraph
    neuron_info_bob = subtensor.neuron_for_uid(uid, netuid=alice_subnet_netuid)
    metagraph_dict = neuron_to_dict(metagraph.neurons[uid])
    subtensor_dict = neuron_to_dict(neuron_info_bob)

    # Verify neuron info is the same in both objects
    assert metagraph_dict == subtensor_dict, (
        "Neuron info of Bob doesn't match b/w metagraph & subtensor"
    )

    # Create pre_dave metagraph for future verifications
    metagraph_pre_dave = subtensor.metagraph(netuid=alice_subnet_netuid)

    # Register Dave as a neuron
    assert subtensor.burned_register(dave_wallet, alice_subnet_netuid), (
        "Unable to register Dave as a neuron"
    )

    metagraph.sync(subtensor=subtensor)

    # Assert metagraph now includes Dave's neuron
    assert len(metagraph.uids) == 3, (
        "Metagraph doesn't have exactly 3 neurons post Dave"
    )
    assert metagraph.hotkeys[2] == dave_wallet.hotkey.ss58_address, (
        "Neuron's hotkey in metagraph doesn't match"
    )
    assert len(metagraph.coldkeys) == 3, (
        "Metagraph doesn't have exactly 3 coldkeys post Dave"
    )
    assert metagraph.n.max() == 3, "Metagraph's max n is not 3 post Dave"
    assert metagraph.n.min() == 3, "Metagraph's min n is not 3 post Dave"
    assert len(metagraph.addresses) == 3, "Metagraph doesn't have 3 addresses post Dave"

    # Add stake by Bob
    tao = Balance.from_tao(10_000)
    alpha, _ = subtensor.subnet(alice_subnet_netuid).tao_to_alpha_with_slippage(tao)
    assert subtensor.add_stake(
        bob_wallet,
        netuid=alice_subnet_netuid,
        amount=tao,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    ), "Failed to add stake for Bob"

    # Assert stake is added after updating metagraph
    metagraph.sync(subtensor=subtensor)
    assert 0.95 < metagraph.neurons[1].stake.rao / alpha.rao < 1.05, (
        "Bob's stake not updated in metagraph"
    )

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
    assert len(metagraph.uids) == len(metagraph_pre_dave.uids), (
        "UID count mismatch after save and load"
    )
    assert (metagraph.uids == metagraph_pre_dave.uids).all(), (
        "UIDs don't match after save and load"
    )

    assert len(metagraph.axons) == len(metagraph_pre_dave.axons), (
        "Axon count mismatch after save and load"
    )
    assert metagraph.axons[1].hotkey == metagraph_pre_dave.axons[1].hotkey, (
        "Axon hotkey mismatch after save and load"
    )
    assert metagraph.axons == metagraph_pre_dave.axons, (
        "Axons don't match after save and load"
    )

    assert len(metagraph.neurons) == len(metagraph_pre_dave.neurons), (
        "Neuron count mismatch after save and load"
    )
    assert metagraph.neurons == metagraph_pre_dave.neurons, (
        "Neurons don't match after save and load"
    )

    logging.console.info("✅ Passed test_metagraph")


def test_metagraph_info(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Check MetagraphInfo
    - Register Neuron
    - Register Subnet
    - Check MetagraphInfo is updated
    """

    alice_subnet_netuid = subtensor.get_total_subnets()  # 2
    subtensor.register_subnet(alice_wallet)

    metagraph_info = subtensor.get_metagraph_info(netuid=1, block=1)

    assert metagraph_info == MetagraphInfo(
        netuid=1,
        name="apex",
        symbol="α",
        identity=None,
        network_registered_at=0,
        owner_hotkey=(NULL_KEY,),
        owner_coldkey=(NULL_KEY,),
        block=1,
        tempo=100,
        last_step=0,
        blocks_since_last_step=1,
        subnet_emission=Balance(0),
        alpha_in=Balance.from_tao(10),
        alpha_out=Balance.from_tao(1),
        tao_in=Balance.from_tao(10),
        alpha_out_emission=Balance(0),
        alpha_in_emission=Balance(0),
        tao_in_emission=Balance(0),
        pending_alpha_emission=Balance(0),
        pending_root_emission=Balance(0),
        subnet_volume=Balance(0),
        moving_price=Balance(0),
        rho=10,
        kappa=32767,
        min_allowed_weights=0.0,
        max_weights_limit=1.0,
        weights_version=0,
        weights_rate_limit=100,
        activity_cutoff=5000,
        max_validators=64,
        num_uids=1,
        max_uids=256,
        burn=Balance.from_tao(1),
        difficulty=5.421010862427522e-13,
        registration_allowed=True,
        pow_registration_allowed=False,
        immunity_period=4096,
        min_difficulty=5.421010862427522e-13,
        max_difficulty=0.25,
        min_burn=Balance.from_tao(0.0005),
        max_burn=Balance.from_tao(100),
        adjustment_alpha=0.0,
        adjustment_interval=100,
        target_regs_per_interval=2,
        max_regs_per_block=1,
        serving_rate_limit=50,
        commit_reveal_weights_enabled=False,
        commit_reveal_period=1,
        liquid_alpha_enabled=False,
        alpha_high=0.9000076295109484,
        alpha_low=0.7000076295109483,
        bonds_moving_avg=4.87890977618477e-14,
        hotkeys=["5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"],
        coldkeys=["5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"],
        identities=[None],
        axons=(
            {
                "block": 0,
                "version": 0,
                "ip": 0,
                "port": 0,
                "ip_type": 0,
                "protocol": 0,
                "placeholder1": 0,
                "placeholder2": 0,
            },
        ),
        active=(True,),
        validator_permit=(False,),
        pruning_score=[0.0],
        last_update=(0,),
        emission=[Balance(0)],
        dividends=[0.0],
        incentives=[0.0],
        consensus=[0.0],
        trust=[0.0],
        rank=[0.0],
        block_at_registration=(0,),
        alpha_stake=[Balance.from_tao(1.0)],
        tao_stake=[Balance(0)],
        total_stake=[Balance.from_tao(1.0)],
        tao_dividends_per_hotkey=[
            ("5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM", Balance(0))
        ],
        alpha_dividends_per_hotkey=[
            ("5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM", Balance(0))
        ],
    )

    metagraph_infos = subtensor.get_all_metagraphs_info(block=1)

    assert metagraph_infos == [
        MetagraphInfo(
            netuid=0,
            name="root",
            symbol="Τ",
            identity=None,
            network_registered_at=0,
            owner_hotkey=(NULL_KEY,),
            owner_coldkey=(NULL_KEY,),
            block=1,
            tempo=100,
            last_step=0,
            blocks_since_last_step=1,
            subnet_emission=Balance(0),
            alpha_in=Balance(0),
            alpha_out=Balance(0),
            tao_in=Balance(0),
            alpha_out_emission=Balance(0),
            alpha_in_emission=Balance(0),
            tao_in_emission=Balance(0),
            pending_alpha_emission=Balance(0),
            pending_root_emission=Balance(0),
            subnet_volume=Balance(0),
            moving_price=Balance(0),
            rho=10,
            kappa=32767,
            min_allowed_weights=0.0,
            max_weights_limit=1.0,
            weights_version=0,
            weights_rate_limit=100,
            activity_cutoff=5000,
            max_validators=64,
            num_uids=0,
            max_uids=64,
            burn=Balance.from_tao(1),
            difficulty=5.421010862427522e-13,
            registration_allowed=True,
            pow_registration_allowed=False,
            immunity_period=4096,
            min_difficulty=5.421010862427522e-13,
            max_difficulty=0.25,
            min_burn=Balance.from_tao(0.0005),
            max_burn=Balance.from_tao(100),
            adjustment_alpha=0.0,
            adjustment_interval=100,
            target_regs_per_interval=1,
            max_regs_per_block=1,
            serving_rate_limit=50,
            commit_reveal_weights_enabled=False,
            commit_reveal_period=1,
            liquid_alpha_enabled=False,
            alpha_high=0.9000076295109484,
            alpha_low=0.7000076295109483,
            bonds_moving_avg=4.87890977618477e-14,
            hotkeys=[],
            coldkeys=[],
            identities={},
            axons=(),
            active=(),
            validator_permit=(),
            pruning_score=[],
            last_update=(),
            emission=[],
            dividends=[],
            incentives=[],
            consensus=[],
            trust=[],
            rank=[],
            block_at_registration=(),
            alpha_stake=[],
            tao_stake=[],
            total_stake=[],
            tao_dividends_per_hotkey=[],
            alpha_dividends_per_hotkey=[],
        ),
        metagraph_info,
    ]

    subtensor.wait_for_block(subtensor.block + 20)
    status, message = subtensor.start_call(
        alice_wallet, alice_subnet_netuid, True, True
    )
    assert status, message

    assert subtensor.burned_register(
        bob_wallet,
        netuid=alice_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    metagraph_info = subtensor.get_metagraph_info(netuid=alice_subnet_netuid)

    assert metagraph_info.num_uids == 2
    assert metagraph_info.hotkeys == [
        alice_wallet.hotkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
    ]
    assert metagraph_info.coldkeys == [
        alice_wallet.coldkey.ss58_address,
        bob_wallet.coldkey.ss58_address,
    ]
    assert metagraph_info.tao_dividends_per_hotkey == [
        (
            alice_wallet.hotkey.ss58_address,
            metagraph_info.tao_dividends_per_hotkey[0][1],
        ),
        (bob_wallet.hotkey.ss58_address, metagraph_info.tao_dividends_per_hotkey[1][1]),
    ]
    assert metagraph_info.alpha_dividends_per_hotkey == [
        (
            alice_wallet.hotkey.ss58_address,
            metagraph_info.alpha_dividends_per_hotkey[0][1],
        ),
        (
            bob_wallet.hotkey.ss58_address,
            metagraph_info.alpha_dividends_per_hotkey[1][1],
        ),
    ]

    alice_subnet_netuid = subtensor.get_total_subnets()  # 3
    assert subtensor.register_subnet(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    block = subtensor.get_current_block()
    metagraph_info = subtensor.get_metagraph_info(
        netuid=alice_subnet_netuid, block=block
    )

    assert metagraph_info.owner_coldkey == (tuple(alice_wallet.hotkey.public_key),)
    assert metagraph_info.owner_hotkey == (tuple(alice_wallet.coldkey.public_key),)

    metagraph_infos = subtensor.get_all_metagraphs_info(block)

    assert len(metagraph_infos) == 4
    assert metagraph_infos[-1] == metagraph_info

    # non-existed subnet
    metagraph_info = subtensor.get_metagraph_info(netuid=alice_subnet_netuid + 1)

    assert metagraph_info is None


def test_blocks(subtensor):
    """
    Tests:
    - Get current block
    - Get block hash
    - Wait for block
    """

    block = subtensor.get_current_block()

    assert block == subtensor.block

    block_hash = subtensor.get_block_hash(block)

    assert re.match("0x[a-z0-9]{64}", block_hash)

    subtensor.wait_for_block(block + 10)

    assert subtensor.get_current_block() == block + 10
