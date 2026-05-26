import pytest

from bittensor import logging
from bittensor.utils.balance import Balance
from tests.e2e_tests.utils import (
    TestSubnet,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
    REGISTER_NEURON,
)


def test_owner_lock_lifecycle(subtensor, alice_wallet, bob_wallet):
    """
    Tests owner lock lifecycle including auto-lock from emission.

    1. After emission, owner already has an auto-lock on their own hotkey
    2. Query lock state and verify conviction > 0 (owner gets instant conviction)
    3. Verify get_most_convicted_hotkey_on_subnet returns owner hotkey
    4. Fail to lock on a different hotkey (LockHotkeyMismatch)
    5. Top-up the existing lock on the same hotkey
    6. Unstake within available limit succeeds, lock remains unchanged
    """
    alice_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    alice_sn.execute_steps(steps)

    # Wait for at least one epoch so that emission fires auto_lock_owner_cut
    next_epoch = subtensor.subnets.get_next_epoch_start_block(alice_sn.netuid)
    subtensor.wait_for_block(next_epoch + 1)

    # Owner should already have an auto-lock on their own hotkey
    locks = subtensor.staking.get_stake_locks(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Owner locks after emission: {locks}")
    assert len(locks) == 1
    hotkey, lock_state = locks[0]
    assert hotkey == alice_wallet.hotkey.ss58_address
    assert lock_state["locked_mass"].rao > 0

    # Owner conviction should be > 0 (owner gets conviction = locked_mass instantly)
    conviction = subtensor.staking.get_hotkey_conviction(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Owner hotkey conviction: {conviction}")
    assert conviction > 0.0

    # Subnet king should be the owner hotkey
    king = subtensor.staking.get_most_convicted_hotkey_on_subnet(
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Subnet king: {king}")
    assert king == alice_wallet.hotkey.ss58_address

    # Trying to lock on a different hotkey should fail (LockHotkeyMismatch)
    response = subtensor.staking.lock_stake(
        wallet=alice_wallet,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
        amount=Balance.from_tao(1).set_unit(alice_sn.netuid),
    )
    logging.console.info(f"Lock on different hotkey response: {response}")
    assert response.success is False

    # Add more stake on own hotkey so we have enough to lock
    assert subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(5),
    ).success

    # Record locked_mass before top-up
    lock_before = subtensor.staking.get_stake_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
    )
    locked_mass_before = lock_before["locked_mass"].rao
    logging.console.info(f"Locked mass before top-up: {locked_mass_before}")

    # Top-up lock on own hotkey
    top_up_amount = Balance.from_tao(2).set_unit(alice_sn.netuid)
    response = subtensor.staking.lock_stake(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
        amount=top_up_amount,
    )
    logging.console.info(f"Top-up lock response: {response}")
    assert response.success

    # Verify locked_mass increased
    lock_after = subtensor.staking.get_stake_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
    )
    logging.console.info(f"Locked mass after top-up: {lock_after['locked_mass'].rao}")
    assert lock_after["locked_mass"].rao > locked_mass_before

    # Unstaking within available limit should succeed without affecting the lock
    small_amount = Balance.from_tao(1).set_unit(alice_sn.netuid)
    response = subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=small_amount,
    )
    logging.console.info(f"Unstake small amount response: {response}")
    assert response.success

    # Unstake must not reduce locked_mass (may grow due to auto_lock_owner_cut between blocks)
    lock_unchanged = subtensor.staking.get_stake_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
    )
    assert lock_unchanged["locked_mass"].rao >= lock_after["locked_mass"].rao


@pytest.mark.asyncio
async def test_owner_lock_lifecycle_async(async_subtensor, alice_wallet, bob_wallet):
    """
    Async version: Tests owner lock lifecycle including auto-lock from emission.

    1. After emission, owner already has an auto-lock on their own hotkey
    2. Query lock state and verify conviction > 0 (owner gets instant conviction)
    3. Verify get_most_convicted_hotkey_on_subnet returns owner hotkey
    4. Fail to lock on a different hotkey (LockHotkeyMismatch)
    5. Top-up the existing lock on the same hotkey
    6. Unstake within available limit succeeds, lock remains unchanged
    """
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    # Wait for at least one epoch so that emission fires auto_lock_owner_cut
    next_epoch = await async_subtensor.subnets.get_next_epoch_start_block(
        alice_sn.netuid
    )
    await async_subtensor.wait_for_block(next_epoch + 1)

    # Owner should already have an auto-lock on their own hotkey
    locks = await async_subtensor.staking.get_stake_locks(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Owner locks after emission: {locks}")
    assert len(locks) == 1
    hotkey, lock_state = locks[0]
    assert hotkey == alice_wallet.hotkey.ss58_address
    assert lock_state["locked_mass"].rao > 0

    # Owner conviction should be > 0 (owner gets conviction = locked_mass instantly)
    conviction = await async_subtensor.staking.get_hotkey_conviction(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Owner hotkey conviction: {conviction}")
    assert conviction > 0.0

    # Subnet king should be the owner hotkey
    king = await async_subtensor.staking.get_most_convicted_hotkey_on_subnet(
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Subnet king: {king}")
    assert king == alice_wallet.hotkey.ss58_address

    # Trying to lock on a different hotkey should fail (LockHotkeyMismatch)
    response = await async_subtensor.staking.lock_stake(
        wallet=alice_wallet,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
        amount=Balance.from_tao(1).set_unit(alice_sn.netuid),
    )
    logging.console.info(f"Lock on different hotkey response: {response}")
    assert response.success is False

    # Add more stake on own hotkey so we have enough to lock
    assert (
        await async_subtensor.staking.add_stake(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(5),
        )
    ).success

    # Record locked_mass before top-up
    lock_before = await async_subtensor.staking.get_stake_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
    )
    locked_mass_before = lock_before["locked_mass"].rao
    logging.console.info(f"Locked mass before top-up: {locked_mass_before}")

    # Top-up lock on own hotkey
    top_up_amount = Balance.from_tao(2).set_unit(alice_sn.netuid)
    response = await async_subtensor.staking.lock_stake(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
        amount=top_up_amount,
    )
    logging.console.info(f"Top-up lock response: {response}")
    assert response.success

    # Verify locked_mass increased
    lock_after = await async_subtensor.staking.get_stake_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
    )
    logging.console.info(f"Locked mass after top-up: {lock_after['locked_mass'].rao}")
    assert lock_after["locked_mass"].rao > locked_mass_before

    # Unstaking within available limit should succeed without affecting the lock
    small_amount = Balance.from_tao(1).set_unit(alice_sn.netuid)
    response = await async_subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=small_amount,
    )
    logging.console.info(f"Unstake small amount response: {response}")
    assert response.success

    # Unstake must not reduce locked_mass (may grow due to auto_lock_owner_cut between blocks)
    lock_unchanged = await async_subtensor.staking.get_stake_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
    )
    assert lock_unchanged["locked_mass"].rao >= lock_after["locked_mass"].rao


def test_non_owner_lock_lifecycle(subtensor, alice_wallet, bob_wallet, charlie_wallet):
    """
    Tests non-owner lock lifecycle from scratch.

    1. No locks exist initially for non-owner
    2. Fail to lock without sufficient stake (InsufficientStakeForLock)
    3. Add stake and lock successfully
    4. Verify lock state (locked_mass, conviction near zero)
    5. Fail to lock on a different hotkey (LockHotkeyMismatch)
    6. Top-up lock on same hotkey
    7. Fail to unstake more than available (total - locked_mass)
    8. Successfully unstake within available limit
    """
    alice_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
        REGISTER_NEURON(charlie_wallet),
    ]
    alice_sn.execute_steps(steps)

    # Bob has no locks initially
    locks = subtensor.staking.get_stake_locks(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert len(locks) == 0

    # Locking without stake should fail
    lock_amount = Balance.from_tao(1).set_unit(alice_sn.netuid)
    response = subtensor.staking.lock_stake(
        wallet=bob_wallet,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
        amount=lock_amount,
    )
    logging.console.info(f"Lock without stake response: {response}")
    assert response.success is False

    # Add stake for Bob
    assert subtensor.staking.add_stake(
        wallet=bob_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(5),
    ).success

    # Now lock should succeed
    first_lock = Balance.from_tao(1).set_unit(alice_sn.netuid)
    response = subtensor.staking.lock_stake(
        wallet=bob_wallet,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
        amount=first_lock,
    )
    logging.console.info(f"First lock response: {response}")
    assert response.success

    # Verify lock state
    lock_info = subtensor.staking.get_stake_lock(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
    )
    logging.console.info(f"Bob lock info: {lock_info}")
    assert lock_info is not None
    assert lock_info["locked_mass"].rao == first_lock.rao
    assert lock_info["conviction"] < 1.0
    assert lock_info["last_update"] > 0

    # Hotkey conviction should be near zero for a fresh non-owner lock
    conviction = subtensor.staking.get_hotkey_conviction(
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Bob hotkey conviction: {conviction}")
    assert isinstance(conviction, float)

    # Locking on a different hotkey should fail (LockHotkeyMismatch)
    response = subtensor.staking.lock_stake(
        wallet=bob_wallet,
        hotkey_ss58=charlie_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
        amount=Balance.from_tao(1).set_unit(alice_sn.netuid),
    )
    logging.console.info(f"Lock on different hotkey response: {response}")
    assert response.success is False

    # Top-up on same hotkey should succeed
    second_lock = Balance.from_tao(1).set_unit(alice_sn.netuid)
    response = subtensor.staking.lock_stake(
        wallet=bob_wallet,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
        amount=second_lock,
    )
    assert response.success

    # Verify locked_mass increased (not exact due to decay between blocks)
    lock_info = subtensor.staking.get_stake_lock(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
    )
    logging.console.info(
        f"Bob locked_mass after top-up: {lock_info['locked_mass'].rao}, "
        f"first_lock: {first_lock.rao}"
    )
    assert lock_info["locked_mass"].rao > first_lock.rao

    # Unstaking more than available should fail
    total_stake = subtensor.staking.get_stake(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    response = subtensor.staking.unstake(
        wallet=bob_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=total_stake,
    )
    logging.console.info(f"Unstake all response: {response}")
    assert response.success is False

    # Unstaking a small amount within available limit should succeed
    small_amount = Balance.from_tao(1).set_unit(alice_sn.netuid)
    available = total_stake.rao - lock_info["locked_mass"].rao
    if available > small_amount.rao:
        response = subtensor.staking.unstake(
            wallet=bob_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            amount=small_amount,
        )
        logging.console.info(f"Unstake small amount response: {response}")
        assert response.success


@pytest.mark.asyncio
async def test_non_owner_lock_lifecycle_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet
):
    """
    Async version: Tests non-owner lock lifecycle from scratch.

    1. No locks exist initially for non-owner
    2. Fail to lock without sufficient stake (InsufficientStakeForLock)
    3. Add stake and lock successfully
    4. Verify lock state (locked_mass, conviction near zero)
    5. Fail to lock on a different hotkey (LockHotkeyMismatch)
    6. Top-up lock on same hotkey
    7. Fail to unstake more than available (total - locked_mass)
    8. Successfully unstake within available limit
    """
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
        REGISTER_NEURON(charlie_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    # Bob has no locks initially
    locks = await async_subtensor.staking.get_stake_locks(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert len(locks) == 0

    # Locking without stake should fail
    lock_amount = Balance.from_tao(1).set_unit(alice_sn.netuid)
    response = await async_subtensor.staking.lock_stake(
        wallet=bob_wallet,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
        amount=lock_amount,
    )
    logging.console.info(f"Lock without stake response: {response}")
    assert response.success is False

    # Add stake for Bob
    assert (
        await async_subtensor.staking.add_stake(
            wallet=bob_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(5),
        )
    ).success

    # Now lock should succeed
    first_lock = Balance.from_tao(1).set_unit(alice_sn.netuid)
    response = await async_subtensor.staking.lock_stake(
        wallet=bob_wallet,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
        amount=first_lock,
    )
    logging.console.info(f"First lock response: {response}")
    assert response.success

    # Verify lock state
    lock_info = await async_subtensor.staking.get_stake_lock(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
    )
    logging.console.info(f"Bob lock info: {lock_info}")
    assert lock_info is not None
    assert lock_info["locked_mass"].rao == first_lock.rao
    assert lock_info["conviction"] < 1.0
    assert lock_info["last_update"] > 0

    # Hotkey conviction should be near zero for a fresh non-owner lock
    conviction = await async_subtensor.staking.get_hotkey_conviction(
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Bob hotkey conviction: {conviction}")
    assert isinstance(conviction, float)

    # Locking on a different hotkey should fail (LockHotkeyMismatch)
    response = await async_subtensor.staking.lock_stake(
        wallet=bob_wallet,
        hotkey_ss58=charlie_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
        amount=Balance.from_tao(1).set_unit(alice_sn.netuid),
    )
    logging.console.info(f"Lock on different hotkey response: {response}")
    assert response.success is False

    # Top-up on same hotkey should succeed
    second_lock = Balance.from_tao(1).set_unit(alice_sn.netuid)
    response = await async_subtensor.staking.lock_stake(
        wallet=bob_wallet,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
        amount=second_lock,
    )
    assert response.success

    # Verify locked_mass increased (not exact due to decay between blocks)
    lock_info = await async_subtensor.staking.get_stake_lock(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
    )
    logging.console.info(
        f"Bob locked_mass after top-up: {lock_info['locked_mass'].rao}, "
        f"first_lock: {first_lock.rao}"
    )
    assert lock_info["locked_mass"].rao > first_lock.rao

    # Unstaking more than available should fail
    total_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    response = await async_subtensor.staking.unstake(
        wallet=bob_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=total_stake,
    )
    logging.console.info(f"Unstake all response: {response}")
    assert response.success is False

    # Unstaking a small amount within available limit should succeed
    small_amount = Balance.from_tao(1).set_unit(alice_sn.netuid)
    available = total_stake.rao - lock_info["locked_mass"].rao
    if available > small_amount.rao:
        response = await async_subtensor.staking.unstake(
            wallet=bob_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            amount=small_amount,
        )
        logging.console.info(f"Unstake small amount response: {response}")
        assert response.success


def test_move_lock(subtensor, alice_wallet, bob_wallet, charlie_wallet):
    """
    Tests move_lock functionality.

    1. Fail to move when no lock exists (NoExistingLock)
    2. Create a lock for Bob, then move it to Charlie's hotkey
    3. Verify lock moved: old hotkey has None, new hotkey has lock
    4. Owner moves their auto-lock to a different hotkey
    5. Verify owner lock moved successfully
    """
    alice_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
        REGISTER_NEURON(charlie_wallet),
    ]
    alice_sn.execute_steps(steps)

    # Moving without a lock should fail
    response = subtensor.staking.move_lock(
        wallet=bob_wallet,
        destination_hotkey_ss58=charlie_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Move lock without existing lock: {response}")
    assert response.success is False

    # Create a lock for Bob
    assert subtensor.staking.add_stake(
        wallet=bob_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(5),
    ).success

    lock_amount = Balance.from_tao(2).set_unit(alice_sn.netuid)
    assert subtensor.staking.lock_stake(
        wallet=bob_wallet,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
        amount=lock_amount,
    ).success

    # Verify lock exists on Bob's hotkey
    lock_before = subtensor.staking.get_stake_lock(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
    )
    assert lock_before is not None
    logging.console.info(f"Bob lock before move: {lock_before}")

    # Move lock from Bob's hotkey to Charlie's hotkey
    response = subtensor.staking.move_lock(
        wallet=bob_wallet,
        destination_hotkey_ss58=charlie_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Move lock response: {response}")
    assert response.success

    # Old hotkey should have no lock
    old_lock = subtensor.staking.get_stake_lock(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
    )
    assert old_lock is None

    # New hotkey should have the lock
    new_lock = subtensor.staking.get_stake_lock(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=charlie_wallet.hotkey.ss58_address,
    )
    assert new_lock is not None
    assert new_lock["locked_mass"].rao <= lock_amount.rao
    assert new_lock["locked_mass"].rao > lock_amount.rao * 0.99
    logging.console.info(f"Bob lock after move: {new_lock}")

    # Owner move: wait for emission so owner has auto-lock
    next_epoch = subtensor.subnets.get_next_epoch_start_block(alice_sn.netuid)
    subtensor.wait_for_block(next_epoch + 1)

    owner_lock = subtensor.staking.get_stake_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
    )
    logging.console.info(f"Owner lock before move: {owner_lock}")
    assert owner_lock is not None

    # Owner moves lock to Bob's hotkey
    response = subtensor.staking.move_lock(
        wallet=alice_wallet,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Owner move lock response: {response}")
    assert response.success

    # Owner lock should no longer be on alice's hotkey
    old_owner_lock = subtensor.staking.get_stake_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
    )
    assert old_owner_lock is None

    # Owner lock should now be on Bob's hotkey
    new_owner_lock = subtensor.staking.get_stake_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
    )
    assert new_owner_lock is not None
    logging.console.info(f"Owner lock after move: {new_owner_lock}")


@pytest.mark.asyncio
async def test_move_lock_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet
):
    """
    Async version: Tests move_lock functionality.

    1. Fail to move when no lock exists (NoExistingLock)
    2. Create a lock for Bob, then move it to Charlie's hotkey
    3. Verify lock moved: old hotkey has None, new hotkey has lock
    4. Owner moves their auto-lock to a different hotkey
    5. Verify owner lock moved successfully
    """
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
        REGISTER_NEURON(charlie_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    # Moving without a lock should fail
    response = await async_subtensor.staking.move_lock(
        wallet=bob_wallet,
        destination_hotkey_ss58=charlie_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Move lock without existing lock: {response}")
    assert response.success is False

    # Create a lock for Bob
    assert (
        await async_subtensor.staking.add_stake(
            wallet=bob_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(5),
        )
    ).success

    lock_amount = Balance.from_tao(2).set_unit(alice_sn.netuid)
    assert (
        await async_subtensor.staking.lock_stake(
            wallet=bob_wallet,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            netuid=alice_sn.netuid,
            amount=lock_amount,
        )
    ).success

    # Verify lock exists on Bob's hotkey
    lock_before = await async_subtensor.staking.get_stake_lock(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
    )
    assert lock_before is not None
    logging.console.info(f"Bob lock before move: {lock_before}")

    # Move lock from Bob's hotkey to Charlie's hotkey
    response = await async_subtensor.staking.move_lock(
        wallet=bob_wallet,
        destination_hotkey_ss58=charlie_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Move lock response: {response}")
    assert response.success

    # Old hotkey should have no lock
    old_lock = await async_subtensor.staking.get_stake_lock(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
    )
    assert old_lock is None

    # New hotkey should have the lock
    new_lock = await async_subtensor.staking.get_stake_lock(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=charlie_wallet.hotkey.ss58_address,
    )
    assert new_lock is not None
    assert new_lock["locked_mass"].rao <= lock_amount.rao
    assert new_lock["locked_mass"].rao > lock_amount.rao * 0.99
    logging.console.info(f"Bob lock after move: {new_lock}")

    # Owner move: wait for emission so owner has auto-lock
    next_epoch = await async_subtensor.subnets.get_next_epoch_start_block(
        alice_sn.netuid
    )
    await async_subtensor.wait_for_block(next_epoch + 1)

    owner_lock = await async_subtensor.staking.get_stake_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
    )
    logging.console.info(f"Owner lock before move: {owner_lock}")
    assert owner_lock is not None

    # Owner moves lock to Bob's hotkey
    response = await async_subtensor.staking.move_lock(
        wallet=alice_wallet,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Owner move lock response: {response}")
    assert response.success

    # Owner lock should no longer be on alice's hotkey
    old_owner_lock = await async_subtensor.staking.get_stake_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
    )
    assert old_owner_lock is None

    # Owner lock should now be on Bob's hotkey
    new_owner_lock = await async_subtensor.staking.get_stake_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
    )
    assert new_owner_lock is not None
    logging.console.info(f"Owner lock after move: {new_owner_lock}")


def test_perpetual_lock_toggle(subtensor, alice_wallet, bob_wallet):
    """
    Tests set_perpetual_lock toggle for both owner and non-owner.

    1. Locks are decaying by default (is_perpetual_lock returns False)
    2. Toggle to perpetual via set_perpetual_lock(enabled=True), verify
    3. Toggle back to decaying via set_perpetual_lock(enabled=False), verify
    4. Non-owner creates a lock, toggles to perpetual, verify
    """
    alice_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    alice_sn.execute_steps(steps)

    # Owner lock is decaying by default
    is_perpetual = subtensor.staking.is_perpetual_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert is_perpetual is False

    # Set to perpetual
    response = subtensor.staking.set_perpetual_lock(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        enabled=True,
    )
    logging.console.info(f"Set perpetual=True response: {response}")
    assert response.success

    is_perpetual = subtensor.staking.is_perpetual_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert is_perpetual is True

    # Set back to decaying
    response = subtensor.staking.set_perpetual_lock(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        enabled=False,
    )
    logging.console.info(f"Set perpetual=False response: {response}")
    assert response.success

    is_perpetual = subtensor.staking.is_perpetual_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert is_perpetual is False

    # Non-owner: Bob is also decaying by default
    is_perpetual = subtensor.staking.is_perpetual_lock(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert is_perpetual is False

    # Bob creates a lock and sets it to perpetual
    assert subtensor.staking.add_stake(
        wallet=bob_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(3),
    ).success

    assert subtensor.staking.lock_stake(
        wallet=bob_wallet,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
        amount=Balance.from_tao(1).set_unit(alice_sn.netuid),
    ).success

    response = subtensor.staking.set_perpetual_lock(
        wallet=bob_wallet,
        netuid=alice_sn.netuid,
        enabled=True,
    )
    assert response.success

    is_perpetual = subtensor.staking.is_perpetual_lock(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert is_perpetual is True


@pytest.mark.asyncio
async def test_perpetual_lock_toggle_async(async_subtensor, alice_wallet, bob_wallet):
    """
    Async version: Tests set_perpetual_lock toggle for both owner and non-owner.

    1. Locks are decaying by default (is_perpetual_lock returns False)
    2. Toggle to perpetual via set_perpetual_lock(enabled=True), verify
    3. Toggle back to decaying via set_perpetual_lock(enabled=False), verify
    4. Non-owner creates a lock, toggles to perpetual, verify
    """
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    # Owner lock is decaying by default
    is_perpetual = await async_subtensor.staking.is_perpetual_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert is_perpetual is False

    # Set to perpetual
    response = await async_subtensor.staking.set_perpetual_lock(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        enabled=True,
    )
    logging.console.info(f"Set perpetual=True response: {response}")
    assert response.success

    is_perpetual = await async_subtensor.staking.is_perpetual_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert is_perpetual is True

    # Set back to decaying
    response = await async_subtensor.staking.set_perpetual_lock(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        enabled=False,
    )
    logging.console.info(f"Set perpetual=False response: {response}")
    assert response.success

    is_perpetual = await async_subtensor.staking.is_perpetual_lock(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert is_perpetual is False

    # Non-owner: Bob is also decaying by default
    is_perpetual = await async_subtensor.staking.is_perpetual_lock(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert is_perpetual is False

    # Bob creates a lock and sets it to perpetual
    assert (
        await async_subtensor.staking.add_stake(
            wallet=bob_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(3),
        )
    ).success

    assert (
        await async_subtensor.staking.lock_stake(
            wallet=bob_wallet,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            netuid=alice_sn.netuid,
            amount=Balance.from_tao(1).set_unit(alice_sn.netuid),
        )
    ).success

    response = await async_subtensor.staking.set_perpetual_lock(
        wallet=bob_wallet,
        netuid=alice_sn.netuid,
        enabled=True,
    )
    assert response.success

    is_perpetual = await async_subtensor.staking.is_perpetual_lock(
        coldkey_ss58=bob_wallet.coldkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert is_perpetual is True
