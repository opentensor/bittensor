import os.path
import shutil
import time
import numpy as np
import pytest

from bittensor.core.chain_data import SelectiveMetagraphIndex
from bittensor.core.chain_data.metagraph_info import MetagraphInfo
from bittensor.extras.dev_framework import (
    SUDO_SET_WEIGHTS_SET_RATE_LIMIT,
    SUDO_SET_TEMPO,
)
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from bittensor.utils.registration.pow import LazyLoadedTorch
from bittensor.utils.weight_utils import convert_and_normalize_weights_and_uids
from tests.e2e_tests.utils import (
    AdminUtils,
    NETUID,
    TestSubnet,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
    REGISTER_NEURON,
    SUDO_SET_ADMIN_FREEZE_WINDOW,
)

NULL_KEY = tuple(bytearray(32))


torch = LazyLoadedTorch()


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
    alice_sn = TestSubnet(subtensor)
    alice_sn.execute_steps(
        [
            REGISTER_SUBNET(alice_wallet),
            ACTIVATE_SUBNET(alice_wallet),
        ]
    )

    logging.console.info("Initialize metagraph")
    metagraph = subtensor.metagraphs.metagraph(netuid=alice_sn.netuid)

    logging.console.info("Assert metagraph has only Alice (owner)")
    assert len(metagraph.uids) == 1, "Metagraph doesn't have exactly 1 neuron"

    logging.console.info("Register Bob to the subnet")
    assert subtensor.subnets.burned_register(bob_wallet, alice_sn.netuid).success, (
        "Unable to register Bob as a neuron"
    )

    logging.console.info("Refresh the metagraph")
    metagraph.sync(subtensor=subtensor.inner_subtensor)

    logging.console.info("Assert metagraph has Alice and Bob neurons")
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

    logging.console.info("Fetch UID of Bob")
    uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(
        bob_wallet.hotkey.ss58_address, netuid=alice_sn.netuid
    )

    logging.console.info("Fetch neuron info of Bob through subtensor and metagraph")
    neuron_info_bob = subtensor.neurons.neuron_for_uid(uid, netuid=alice_sn.netuid)

    metagraph_dict = neuron_to_dict(metagraph.neurons[uid])
    subtensor_dict = neuron_to_dict(neuron_info_bob)

    logging.console.info("Verify neuron info is the same in both objects")
    assert metagraph_dict == subtensor_dict, (
        "Neuron info of Bob doesn't match b/w metagraph & subtensor"
    )

    logging.console.info("Create pre_dave metagraph for future verifications")
    metagraph_pre_dave = subtensor.metagraphs.metagraph(netuid=alice_sn.netuid)

    logging.console.info("Register Dave as a neuron")
    assert subtensor.subnets.burned_register(dave_wallet, alice_sn.netuid).success, (
        "Unable to register Dave as a neuron"
    )

    metagraph.sync(subtensor=subtensor.inner_subtensor)

    logging.console.info("Assert metagraph now includes Dave's neuron")
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

    logging.console.info("Add stake by Bob")
    tao = Balance.from_tao(10_000)
    alpha, _ = subtensor.subnets.subnet(alice_sn.netuid).tao_to_alpha_with_slippage(tao)
    assert subtensor.staking.add_stake(
        wallet=bob_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=tao,
    ).success, "Failed to add stake for Bob"

    logging.console.info("Assert stake is added after updating metagraph")
    metagraph.sync(subtensor=subtensor.inner_subtensor)
    assert 0.95 < metagraph.neurons[1].stake.rao / alpha.rao < 1.05, (
        "Bob's stake not updated in metagraph"
    )

    logging.console.info("Test the save() and load() mechanism")
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

    logging.console.info("Ensure data is synced between two metagraphs")
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


@pytest.mark.asyncio
async def test_metagraph_async(async_subtensor, alice_wallet, bob_wallet, dave_wallet):
    """
    Async tests the metagraph

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

    alice_sn = TestSubnet(async_subtensor)
    await alice_sn.async_execute_steps(
        [
            REGISTER_SUBNET(alice_wallet),
            ACTIVATE_SUBNET(alice_wallet),
        ]
    )

    logging.console.info("Initialize metagraph")
    metagraph = await async_subtensor.metagraphs.metagraph(netuid=alice_sn.netuid)

    logging.console.info("Assert metagraph has only Alice (owner)")
    assert len(metagraph.uids) == 1, "Metagraph doesn't have exactly 1 neuron"

    logging.console.info("Register Bob to the subnet")
    assert (
        await async_subtensor.subnets.burned_register(bob_wallet, alice_sn.netuid)
    ).success, "Unable to register Bob as a neuron"

    logging.console.info("Refresh the metagraph")
    await metagraph.sync(subtensor=async_subtensor.inner_subtensor)

    logging.console.info("Assert metagraph has Alice and Bob neurons")
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

    logging.console.info("Fetch UID of Bob")
    uid = await async_subtensor.subnets.get_uid_for_hotkey_on_subnet(
        bob_wallet.hotkey.ss58_address, netuid=alice_sn.netuid
    )

    logging.console.info("Fetch neuron info of Bob through subtensor and metagraph")
    neuron_info_bob = await async_subtensor.neurons.neuron_for_uid(
        uid, netuid=alice_sn.netuid
    )

    metagraph_dict = neuron_to_dict(metagraph.neurons[uid])
    subtensor_dict = neuron_to_dict(neuron_info_bob)

    logging.console.info("Verify neuron info is the same in both objects")
    assert metagraph_dict == subtensor_dict, (
        "Neuron info of Bob doesn't match b/w metagraph & subtensor"
    )

    logging.console.info("Create pre_dave metagraph for future verifications")
    metagraph_pre_dave = await async_subtensor.metagraphs.metagraph(
        netuid=alice_sn.netuid
    )

    logging.console.info("Register Dave as a neuron")
    assert (
        await async_subtensor.subnets.burned_register(dave_wallet, alice_sn.netuid)
    ).success, "Unable to register Dave as a neuron"

    await metagraph.sync(subtensor=async_subtensor.inner_subtensor)

    logging.console.info("Assert metagraph now includes Dave's neuron")
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

    logging.console.info("Add stake by Bob")
    tao = Balance.from_tao(10_000)
    alpha, _ = (
        await async_subtensor.subnets.subnet(alice_sn.netuid)
    ).tao_to_alpha_with_slippage(tao)
    assert (
        await async_subtensor.staking.add_stake(
            wallet=bob_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            amount=tao,
        )
    ).success, "Failed to add stake for Bob"

    logging.console.info("Assert stake is added after updating metagraph")
    await metagraph.sync(subtensor=async_subtensor.inner_subtensor)
    assert 0.95 < metagraph.neurons[1].stake.rao / alpha.rao < 1.05, (
        "Bob's stake not updated in metagraph"
    )

    logging.console.info("Test the save() and load() mechanism")
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

    logging.console.info("Ensure data is synced between two metagraphs")
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


def test_metagraph_weights_bonds(
    subtensor, alice_wallet, bob_wallet, charlie_wallet, dave_wallet
):
    """
    Tests that weights and bonds matrices are computed correctly when the metagraph is initialized with lite=False.

    Test:
    - Disable the admin freeze window (set to 0).
    - Register a new subnet owned by Bob.
    - Update subnet tempo to a custom value.
    - Activate the subnet.
    - Register Charlie and Dave as new neurons in the subnet.
    - Disable weights rate limit (set_weights_rate_limit = 0).
    - Set weights for Charlie and Dave (20% / 80%) using Bob.
    - Wait for commit-reveal completion and ensure version is 4.
    - Initialize the metagraph with lite=False.
    - Verify:
        - Shape and consistency of weights and bonds tensors.
        - Validator (Alice) has non-zero outgoing weights.
        - Miners have zero rows (no outgoing weights).
        - All bonds are non-negative.
        - Validator weight rows are normalized to 1.
    """
    logging.set_debug()
    TEMPO_TO_SET, BLOCK_TIME = (
        (100, 0.25) if subtensor.chain.is_fast_blocks() else (20, 12)
    )

    bob_sn = TestSubnet(subtensor)
    bob_sn.execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
            REGISTER_SUBNET(bob_wallet),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(bob_wallet),
            REGISTER_NEURON(charlie_wallet),
            REGISTER_NEURON(dave_wallet),
            SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, 0),
        ]
    )

    # wait before CRv4 works in new subnets
    bob_sn.wait_next_epoch()
    bob_sn.wait_next_epoch()

    cr_version = subtensor.substrate.query(
        module="SubtensorModule", storage_function="CommitRevealWeightsVersion"
    )
    assert cr_version == 4, "Commit reveal weights version is not 4"
    assert subtensor.subnets.weights_rate_limit(netuid=bob_sn.netuid) == 0
    tempo = subtensor.subnets.get_subnet_hyperparameters(netuid=bob_sn.netuid).tempo
    assert tempo == TEMPO_TO_SET, "SN tempos has not been changed."

    metagraph = subtensor.metagraphs.metagraph(netuid=bob_sn.netuid, lite=False)

    # Check that the metagraph is instantiated correctly.
    assert metagraph.weights.shape == (metagraph.n.item(), metagraph.n.item())
    assert metagraph.bonds.shape == (metagraph.n.item(), metagraph.n.item())

    uids = [1, 2]
    weights = [20, 80]

    response = subtensor.extrinsics.set_weights(
        wallet=bob_wallet,
        netuid=bob_sn.netuid,
        uids=uids,
        weights=weights,
        block_time=BLOCK_TIME,
    )
    logging.console.info(f"Response: {response}")

    assert response.success, response.message

    expected_reveal_round = response.data.get("reveal_round")
    last_drand_round = subtensor.chain.last_drand_round()

    while expected_reveal_round > last_drand_round + 24:  # drand offset for fast blocks
        last_drand_round = subtensor.chain.last_drand_round()
        subtensor.wait_for_block()
        logging.console.debug(
            f"expected_reveal_round: {expected_reveal_round}, last_drand_round: {last_drand_round}"
        )

    counter = TEMPO_TO_SET
    while True:
        weights = subtensor.subnets.weights(bob_sn.netuid)
        counter -= 1

        if weights or counter == 0:
            break

        subtensor.wait_for_block()
        logging.console.debug(f"Weights: {weights}, block: {subtensor.block}")

    metagraph.sync()

    # Ensure the validator has at least one non-zero weight
    assert metagraph.weights[0].sum().item() > 0.0, "Validator has no outgoing weights."

    # Ensure miner rows are all zeros (miners don't set weights)
    if metagraph.n.item() > 1:
        assert np.allclose(
            metagraph.weights[1],
            np.zeros_like(metagraph.weights[1]),
        ), "Miner row should be all zeros"

    # Ensure bond matrix contains no negative values
    assert (metagraph.bonds >= 0).all(), "Bond matrix contains negative values"

    # Ensure validator weight rows are normalized to 1
    row_sums = metagraph.weights.sum(axis=1)
    validator_mask = metagraph.validator_permit
    assert np.allclose(
        row_sums[validator_mask],
        np.ones_like(row_sums[validator_mask]),
        atol=1e-6,
    ), "Validator weight rows are not normalized to 1"


@pytest.mark.asyncio
async def test_metagraph_weights_bonds_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet, dave_wallet
):
    """
    Tests that weights and bonds matrices are computed correctly when the metagraph is initialized with lite=False.

    Test:
    - Disable the admin freeze window (set to 0).
    - Register a new subnet owned by Bob.
    - Update subnet tempo to a custom value.
    - Activate the subnet.
    - Register Charlie and Dave as new neurons in the subnet.
    - Disable weights rate limit (set_weights_rate_limit = 0).
    - Set weights for Charlie and Dave (20% / 80%) using Bob.
    - Wait for commit-reveal completion and ensure version is 4.
    - Initialize the metagraph with lite=False.
    - Verify:
        - Shape and consistency of weights and bonds tensors.
        - Validator (Alice) has non-zero outgoing weights.
        - Miners have zero rows (no outgoing weights).
        - All bonds are non-negative.
        - Validator weight rows are normalized to 1.
    """
    logging.set_debug()
    TEMPO_TO_SET, BLOCK_TIME = (
        (100, 0.25) if await async_subtensor.chain.is_fast_blocks() else (20, 12)
    )

    bob_sn = TestSubnet(async_subtensor)
    await bob_sn.async_execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
            REGISTER_SUBNET(bob_wallet),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(bob_wallet),
            REGISTER_NEURON(charlie_wallet),
            REGISTER_NEURON(dave_wallet),
            SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, 0),
        ]
    )

    # wait before CRv4 works in new subnets
    await bob_sn.async_wait_next_epoch()
    await bob_sn.async_wait_next_epoch()

    cr_version = await async_subtensor.substrate.query(
        module="SubtensorModule", storage_function="CommitRevealWeightsVersion"
    )
    assert cr_version == 4, "Commit reveal weights version is not 4"
    assert await async_subtensor.subnets.weights_rate_limit(netuid=bob_sn.netuid) == 0
    tempo = (
        await async_subtensor.subnets.get_subnet_hyperparameters(netuid=bob_sn.netuid)
    ).tempo
    assert tempo == TEMPO_TO_SET, "SN tempos has not been changed."

    metagraph = await async_subtensor.metagraphs.metagraph(
        netuid=bob_sn.netuid, lite=False
    )

    # Check that the metagraph is instantiated correctly.
    assert metagraph.weights.shape == (metagraph.n.item(), metagraph.n.item())
    assert metagraph.bonds.shape == (metagraph.n.item(), metagraph.n.item())

    uids = [1, 2]
    weights = [20, 80]

    response = await async_subtensor.extrinsics.set_weights(
        wallet=bob_wallet,
        netuid=bob_sn.netuid,
        uids=uids,
        weights=weights,
        block_time=BLOCK_TIME,
        period=TEMPO_TO_SET,
        wait_for_finalization=False,
    )
    logging.console.info(f"Response: {response}")

    assert response.success, response.message

    expected_reveal_round = response.data.get("reveal_round")
    last_drand_round = await async_subtensor.chain.last_drand_round()

    while expected_reveal_round > last_drand_round + 24:  # drand offset for fast blocks
        last_drand_round = await async_subtensor.chain.last_drand_round()
        await async_subtensor.wait_for_block()
        logging.console.debug(
            f"expected_reveal_round: {expected_reveal_round}, last_drand_round: {last_drand_round}"
        )

    counter = TEMPO_TO_SET
    while True:
        weights = await async_subtensor.subnets.weights(bob_sn.netuid)
        counter -= 1

        if weights or counter == 0:
            break

        await async_subtensor.wait_for_block()
        logging.console.debug(
            f"Weights: {weights}, block: {await async_subtensor.block}"
        )

    await metagraph.sync()

    # Ensure the validator has at least one non-zero weight
    assert metagraph.weights[0].sum().item() > 0.0, "Validator has no outgoing weights."

    # Ensure miner rows are all zeros (miners don't set weights)
    if metagraph.n.item() > 1:
        assert np.allclose(
            metagraph.weights[1],
            np.zeros_like(metagraph.weights[1]),
        ), "Miner row should be all zeros"

    # Ensure bond matrix contains no negative values
    assert (metagraph.bonds >= 0).all(), "Bond matrix contains negative values"

    # Ensure validator weight rows are normalized to 1
    row_sums = metagraph.weights.sum(axis=1)
    validator_mask = metagraph.validator_permit
    assert np.allclose(
        row_sums[validator_mask],
        np.ones_like(row_sums[validator_mask]),
        atol=1e-6,
    ), "Validator weight rows are not normalized to 1"


def test_metagraph_info(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Check MetagraphInfo
    - Register Neuron
    - Register Subnet
    - Check MetagraphInfo is updated
    """
    alice_sn = TestSubnet(subtensor)
    alice_sn.execute_one(REGISTER_SUBNET(alice_wallet))

    metagraph_info = subtensor.metagraphs.get_metagraph_info(netuid=1, block=1)

    expected_metagraph_info = MetagraphInfo(
        netuid=1,
        mechid=0,
        name="apex",
        symbol="α",
        identity=None,
        network_registered_at=0,
        owner_hotkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
        owner_coldkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
        block=1,
        tempo=100,
        last_step=0,
        blocks_since_last_step=1,
        subnet_emission=Balance(0),
        alpha_in=Balance.from_tao(10).set_unit(1),
        alpha_out=Balance.from_tao(1).set_unit(1),
        tao_in=Balance.from_tao(10),
        alpha_out_emission=Balance(0).set_unit(1),
        alpha_in_emission=Balance(0).set_unit(1),
        tao_in_emission=Balance(0),
        pending_alpha_emission=Balance(0).set_unit(1),
        pending_root_emission=Balance(0),
        subnet_volume=Balance(0).set_unit(1),
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
        burn=Balance.from_tao(0.1),
        difficulty=5.421010862427522e-13,
        registration_allowed=True,
        pow_registration_allowed=True,
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
        commit_reveal_weights_enabled=True,
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
        emission=[Balance(0).set_unit(1)],
        dividends=[0.0],
        incentives=[0.0],
        consensus=[0.0],
        trust=[0.0],
        rank=[0.0],
        block_at_registration=(0,),
        alpha_stake=[Balance.from_tao(1.0).set_unit(1)],
        tao_stake=[Balance(0)],
        total_stake=[Balance.from_tao(1.0).set_unit(1)],
        tao_dividends_per_hotkey=[
            ("5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM", Balance(0))
        ],
        alpha_dividends_per_hotkey=[
            ("5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM", Balance(0).set_unit(1))
        ],
        validators=None,
        commitments=None,
    )

    assert metagraph_info == expected_metagraph_info

    metagraph_infos = subtensor.metagraphs.get_all_metagraphs_info(block=1)

    expected_metagraph_infos = [
        MetagraphInfo(
            netuid=0,
            mechid=0,
            name="root",
            symbol="Τ",
            identity=None,
            network_registered_at=0,
            owner_hotkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            owner_coldkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
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
            burn=Balance.from_tao(0.1),
            difficulty=5.421010862427522e-13,
            registration_allowed=True,
            pow_registration_allowed=True,
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
            commit_reveal_weights_enabled=True,
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
            validators=None,
            commitments=None,
        ),
        metagraph_info,
    ]

    assert metagraph_infos == expected_metagraph_infos

    alice_sn.execute_steps([ACTIVATE_SUBNET(alice_wallet), REGISTER_NEURON(bob_wallet)])

    metagraph_info = subtensor.metagraphs.get_metagraph_info(netuid=alice_sn.netuid)

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

    bob_sn = TestSubnet(subtensor)
    bob_sn.execute_one(REGISTER_SUBNET(bob_wallet))

    block = subtensor.chain.get_current_block()
    metagraph_info = subtensor.metagraphs.get_metagraph_info(
        netuid=bob_sn.netuid, block=block
    )

    assert metagraph_info.owner_coldkey == bob_wallet.hotkey.ss58_address
    assert metagraph_info.owner_hotkey == bob_wallet.coldkey.ss58_address

    metagraph_infos = subtensor.metagraphs.get_all_metagraphs_info(block)

    assert len(metagraph_infos) == 4
    assert metagraph_infos[-1] == metagraph_info

    # non-existed subnet
    metagraph_info = subtensor.metagraphs.get_metagraph_info(netuid=bob_sn.netuid + 1)

    assert metagraph_info is None


@pytest.mark.asyncio
async def test_metagraph_info_async(async_subtensor, alice_wallet, bob_wallet):
    """
    Async tests:
    - Check MetagraphInfo
    - Register Neuron
    - Register Subnet
    - Check MetagraphInfo is updated
    """
    alice_sn = TestSubnet(async_subtensor)
    await alice_sn.async_execute_one(REGISTER_SUBNET(alice_wallet))

    metagraph_info = await async_subtensor.metagraphs.get_metagraph_info(
        netuid=1, block=1
    )

    expected_metagraph_info = MetagraphInfo(
        netuid=1,
        mechid=0,
        name="apex",
        symbol="α",
        identity=None,
        network_registered_at=0,
        owner_hotkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
        owner_coldkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
        block=1,
        tempo=100,
        last_step=0,
        blocks_since_last_step=1,
        subnet_emission=Balance(0),
        alpha_in=Balance.from_tao(10).set_unit(1),
        alpha_out=Balance.from_tao(1).set_unit(1),
        tao_in=Balance.from_tao(10),
        alpha_out_emission=Balance(0).set_unit(1),
        alpha_in_emission=Balance(0).set_unit(1),
        tao_in_emission=Balance(0),
        pending_alpha_emission=Balance(0).set_unit(1),
        pending_root_emission=Balance(0),
        subnet_volume=Balance(0).set_unit(1),
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
        burn=Balance.from_tao(0.1),
        difficulty=5.421010862427522e-13,
        registration_allowed=True,
        pow_registration_allowed=True,
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
        commit_reveal_weights_enabled=True,
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
        emission=[Balance(0).set_unit(1)],
        dividends=[0.0],
        incentives=[0.0],
        consensus=[0.0],
        trust=[0.0],
        rank=[0.0],
        block_at_registration=(0,),
        alpha_stake=[Balance.from_tao(1.0).set_unit(1)],
        tao_stake=[Balance(0)],
        total_stake=[Balance.from_tao(1.0).set_unit(1)],
        tao_dividends_per_hotkey=[
            ("5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM", Balance(0))
        ],
        alpha_dividends_per_hotkey=[
            ("5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM", Balance(0).set_unit(1))
        ],
        validators=None,
        commitments=None,
    )

    assert metagraph_info == expected_metagraph_info

    metagraph_infos = await async_subtensor.metagraphs.get_all_metagraphs_info(block=1)

    expected_metagraph_infos = [
        MetagraphInfo(
            netuid=0,
            mechid=0,
            name="root",
            symbol="Τ",
            identity=None,
            network_registered_at=0,
            owner_hotkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            owner_coldkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
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
            burn=Balance.from_tao(0.1),
            difficulty=5.421010862427522e-13,
            registration_allowed=True,
            pow_registration_allowed=True,
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
            commit_reveal_weights_enabled=True,
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
            validators=None,
            commitments=None,
        ),
        metagraph_info,
    ]

    assert metagraph_infos == expected_metagraph_infos

    await alice_sn.async_execute_steps(
        [ACTIVATE_SUBNET(alice_wallet), REGISTER_NEURON(bob_wallet)]
    )

    metagraph_info = await async_subtensor.metagraphs.get_metagraph_info(
        netuid=alice_sn.netuid
    )

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

    bob_sn = TestSubnet(async_subtensor)
    await bob_sn.async_execute_one(REGISTER_SUBNET(bob_wallet))

    block = await async_subtensor.chain.get_current_block()
    metagraph_info = await async_subtensor.metagraphs.get_metagraph_info(
        netuid=bob_sn.netuid, block=block
    )

    assert metagraph_info.owner_coldkey == bob_wallet.hotkey.ss58_address
    assert metagraph_info.owner_hotkey == bob_wallet.coldkey.ss58_address

    metagraph_infos = await async_subtensor.metagraphs.get_all_metagraphs_info(block)

    assert len(metagraph_infos) == 4
    assert metagraph_infos[-1] == metagraph_info

    # non-existed subnet
    metagraph_info = await async_subtensor.metagraphs.get_metagraph_info(
        netuid=bob_sn.netuid + 1
    )

    assert metagraph_info is None


def test_metagraph_info_with_indexes(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Check MetagraphInfo
    - Register Neuron
    - Register Subnet
    - Check MetagraphInfo is updated
    """
    alice_sn = TestSubnet(subtensor)
    alice_sn.execute_one(REGISTER_SUBNET(alice_wallet))

    selected_indices = [
        SelectiveMetagraphIndex.Name,
        SelectiveMetagraphIndex.Active,
        SelectiveMetagraphIndex.OwnerHotkey,
        SelectiveMetagraphIndex.OwnerColdkey,
        SelectiveMetagraphIndex.Axons,
    ]

    metagraph_info = subtensor.metagraphs.get_metagraph_info(
        netuid=alice_sn.netuid, selected_indices=selected_indices
    )

    assert metagraph_info == MetagraphInfo(
        netuid=alice_sn.netuid,
        mechid=0,
        name="omron",
        owner_hotkey=alice_wallet.hotkey.ss58_address,
        owner_coldkey=alice_wallet.coldkey.ss58_address,
        active=(True,),
        axons=(
            {
                "block": 0,
                "ip": 0,
                "ip_type": 0,
                "placeholder1": 0,
                "placeholder2": 0,
                "port": 0,
                "protocol": 0,
                "version": 0,
            },
        ),
        symbol=None,
        identity=None,
        network_registered_at=None,
        block=None,
        tempo=None,
        last_step=None,
        blocks_since_last_step=None,
        subnet_emission=None,
        alpha_in=None,
        alpha_out=None,
        tao_in=None,
        alpha_out_emission=None,
        alpha_in_emission=None,
        tao_in_emission=None,
        pending_alpha_emission=None,
        pending_root_emission=None,
        subnet_volume=None,
        moving_price=None,
        rho=None,
        kappa=None,
        min_allowed_weights=None,
        max_weights_limit=None,
        weights_version=None,
        weights_rate_limit=None,
        activity_cutoff=None,
        max_validators=None,
        num_uids=None,
        max_uids=None,
        burn=None,
        difficulty=None,
        registration_allowed=None,
        pow_registration_allowed=None,
        immunity_period=None,
        min_difficulty=None,
        max_difficulty=None,
        min_burn=None,
        max_burn=None,
        adjustment_alpha=None,
        adjustment_interval=None,
        target_regs_per_interval=None,
        max_regs_per_block=None,
        serving_rate_limit=None,
        commit_reveal_weights_enabled=None,
        commit_reveal_period=None,
        liquid_alpha_enabled=None,
        alpha_high=None,
        alpha_low=None,
        bonds_moving_avg=None,
        hotkeys=None,
        coldkeys=None,
        identities=None,
        validator_permit=None,
        pruning_score=None,
        last_update=None,
        emission=None,
        dividends=None,
        incentives=None,
        consensus=None,
        trust=None,
        rank=None,
        block_at_registration=None,
        alpha_stake=None,
        tao_stake=None,
        total_stake=None,
        tao_dividends_per_hotkey=None,
        alpha_dividends_per_hotkey=None,
        validators=None,
        commitments=None,
    )

    alice_sn.execute_steps([ACTIVATE_SUBNET(alice_wallet), REGISTER_NEURON(bob_wallet)])

    fields = [
        SelectiveMetagraphIndex.Name,
        SelectiveMetagraphIndex.Active,
        SelectiveMetagraphIndex.OwnerHotkey,
        SelectiveMetagraphIndex.OwnerColdkey,
        SelectiveMetagraphIndex.Axons,
    ]

    metagraph_info = subtensor.metagraphs.get_metagraph_info(
        netuid=alice_sn.netuid, selected_indices=fields
    )

    assert metagraph_info == MetagraphInfo(
        netuid=alice_sn.netuid,
        mechid=0,
        name="omron",
        owner_hotkey=alice_wallet.hotkey.ss58_address,
        owner_coldkey=alice_wallet.coldkey.ss58_address,
        active=(True, True),
        axons=(
            {
                "block": 0,
                "ip": 0,
                "ip_type": 0,
                "placeholder1": 0,
                "placeholder2": 0,
                "port": 0,
                "protocol": 0,
                "version": 0,
            },
            {
                "block": 0,
                "ip": 0,
                "ip_type": 0,
                "placeholder1": 0,
                "placeholder2": 0,
                "port": 0,
                "protocol": 0,
                "version": 0,
            },
        ),
        symbol=None,
        identity=None,
        network_registered_at=None,
        block=None,
        tempo=None,
        last_step=None,
        blocks_since_last_step=None,
        subnet_emission=None,
        alpha_in=None,
        alpha_out=None,
        tao_in=None,
        alpha_out_emission=None,
        alpha_in_emission=None,
        tao_in_emission=None,
        pending_alpha_emission=None,
        pending_root_emission=None,
        subnet_volume=None,
        moving_price=None,
        rho=None,
        kappa=None,
        min_allowed_weights=None,
        max_weights_limit=None,
        weights_version=None,
        weights_rate_limit=None,
        activity_cutoff=None,
        max_validators=None,
        num_uids=None,
        max_uids=None,
        burn=None,
        difficulty=None,
        registration_allowed=None,
        pow_registration_allowed=None,
        immunity_period=None,
        min_difficulty=None,
        max_difficulty=None,
        min_burn=None,
        max_burn=None,
        adjustment_alpha=None,
        adjustment_interval=None,
        target_regs_per_interval=None,
        max_regs_per_block=None,
        serving_rate_limit=None,
        commit_reveal_weights_enabled=None,
        commit_reveal_period=None,
        liquid_alpha_enabled=None,
        alpha_high=None,
        alpha_low=None,
        bonds_moving_avg=None,
        hotkeys=None,
        coldkeys=None,
        identities=None,
        validator_permit=None,
        pruning_score=None,
        last_update=None,
        emission=None,
        dividends=None,
        incentives=None,
        consensus=None,
        trust=None,
        rank=None,
        block_at_registration=None,
        alpha_stake=None,
        tao_stake=None,
        total_stake=None,
        tao_dividends_per_hotkey=None,
        alpha_dividends_per_hotkey=None,
        validators=None,
        commitments=None,
    )


@pytest.mark.asyncio
async def test_metagraph_info_with_indexes_async(
    async_subtensor, alice_wallet, bob_wallet
):
    """
    Async tests:
    - Check MetagraphInfo
    - Register Neuron
    - Register Subnet
    - Check MetagraphInfo is updated
    """
    alice_sn = TestSubnet(async_subtensor)
    await alice_sn.async_execute_one(REGISTER_SUBNET(alice_wallet))

    selected_indices = [
        SelectiveMetagraphIndex.Name,
        SelectiveMetagraphIndex.Active,
        SelectiveMetagraphIndex.OwnerHotkey,
        SelectiveMetagraphIndex.OwnerColdkey,
        SelectiveMetagraphIndex.Axons,
    ]

    metagraph_info = await async_subtensor.metagraphs.get_metagraph_info(
        netuid=alice_sn.netuid, selected_indices=selected_indices
    )

    assert metagraph_info == MetagraphInfo(
        netuid=alice_sn.netuid,
        mechid=0,
        name="omron",
        owner_hotkey=alice_wallet.hotkey.ss58_address,
        owner_coldkey=alice_wallet.coldkey.ss58_address,
        active=(True,),
        axons=(
            {
                "block": 0,
                "ip": 0,
                "ip_type": 0,
                "placeholder1": 0,
                "placeholder2": 0,
                "port": 0,
                "protocol": 0,
                "version": 0,
            },
        ),
        symbol=None,
        identity=None,
        network_registered_at=None,
        block=None,
        tempo=None,
        last_step=None,
        blocks_since_last_step=None,
        subnet_emission=None,
        alpha_in=None,
        alpha_out=None,
        tao_in=None,
        alpha_out_emission=None,
        alpha_in_emission=None,
        tao_in_emission=None,
        pending_alpha_emission=None,
        pending_root_emission=None,
        subnet_volume=None,
        moving_price=None,
        rho=None,
        kappa=None,
        min_allowed_weights=None,
        max_weights_limit=None,
        weights_version=None,
        weights_rate_limit=None,
        activity_cutoff=None,
        max_validators=None,
        num_uids=None,
        max_uids=None,
        burn=None,
        difficulty=None,
        registration_allowed=None,
        pow_registration_allowed=None,
        immunity_period=None,
        min_difficulty=None,
        max_difficulty=None,
        min_burn=None,
        max_burn=None,
        adjustment_alpha=None,
        adjustment_interval=None,
        target_regs_per_interval=None,
        max_regs_per_block=None,
        serving_rate_limit=None,
        commit_reveal_weights_enabled=None,
        commit_reveal_period=None,
        liquid_alpha_enabled=None,
        alpha_high=None,
        alpha_low=None,
        bonds_moving_avg=None,
        hotkeys=None,
        coldkeys=None,
        identities=None,
        validator_permit=None,
        pruning_score=None,
        last_update=None,
        emission=None,
        dividends=None,
        incentives=None,
        consensus=None,
        trust=None,
        rank=None,
        block_at_registration=None,
        alpha_stake=None,
        tao_stake=None,
        total_stake=None,
        tao_dividends_per_hotkey=None,
        alpha_dividends_per_hotkey=None,
        validators=None,
        commitments=None,
    )

    await alice_sn.async_execute_steps(
        [ACTIVATE_SUBNET(alice_wallet), REGISTER_NEURON(bob_wallet)]
    )

    fields = [
        SelectiveMetagraphIndex.Name,
        SelectiveMetagraphIndex.Active,
        SelectiveMetagraphIndex.OwnerHotkey,
        SelectiveMetagraphIndex.OwnerColdkey,
        SelectiveMetagraphIndex.Axons,
    ]

    metagraph_info = await async_subtensor.metagraphs.get_metagraph_info(
        netuid=alice_sn.netuid, selected_indices=fields
    )

    assert metagraph_info == MetagraphInfo(
        netuid=alice_sn.netuid,
        mechid=0,
        name="omron",
        owner_hotkey=alice_wallet.hotkey.ss58_address,
        owner_coldkey=alice_wallet.coldkey.ss58_address,
        active=(True, True),
        axons=(
            {
                "block": 0,
                "ip": 0,
                "ip_type": 0,
                "placeholder1": 0,
                "placeholder2": 0,
                "port": 0,
                "protocol": 0,
                "version": 0,
            },
            {
                "block": 0,
                "ip": 0,
                "ip_type": 0,
                "placeholder1": 0,
                "placeholder2": 0,
                "port": 0,
                "protocol": 0,
                "version": 0,
            },
        ),
        symbol=None,
        identity=None,
        network_registered_at=None,
        block=None,
        tempo=None,
        last_step=None,
        blocks_since_last_step=None,
        subnet_emission=None,
        alpha_in=None,
        alpha_out=None,
        tao_in=None,
        alpha_out_emission=None,
        alpha_in_emission=None,
        tao_in_emission=None,
        pending_alpha_emission=None,
        pending_root_emission=None,
        subnet_volume=None,
        moving_price=None,
        rho=None,
        kappa=None,
        min_allowed_weights=None,
        max_weights_limit=None,
        weights_version=None,
        weights_rate_limit=None,
        activity_cutoff=None,
        max_validators=None,
        num_uids=None,
        max_uids=None,
        burn=None,
        difficulty=None,
        registration_allowed=None,
        pow_registration_allowed=None,
        immunity_period=None,
        min_difficulty=None,
        max_difficulty=None,
        min_burn=None,
        max_burn=None,
        adjustment_alpha=None,
        adjustment_interval=None,
        target_regs_per_interval=None,
        max_regs_per_block=None,
        serving_rate_limit=None,
        commit_reveal_weights_enabled=None,
        commit_reveal_period=None,
        liquid_alpha_enabled=None,
        alpha_high=None,
        alpha_low=None,
        bonds_moving_avg=None,
        hotkeys=None,
        coldkeys=None,
        identities=None,
        validator_permit=None,
        pruning_score=None,
        last_update=None,
        emission=None,
        dividends=None,
        incentives=None,
        consensus=None,
        trust=None,
        rank=None,
        block_at_registration=None,
        alpha_stake=None,
        tao_stake=None,
        total_stake=None,
        tao_dividends_per_hotkey=None,
        alpha_dividends_per_hotkey=None,
        validators=None,
        commitments=None,
    )
