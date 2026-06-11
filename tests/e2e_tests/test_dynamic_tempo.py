"""E2E tests for configurable tempo and owner-triggered epochs (Subtensor PR #2638)."""

import time

import numpy as np
import pytest

from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from tests.e2e_tests.utils import (
    AdminUtils,
    TestSubnet,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
    SUDO_SET_ADMIN_FREEZE_WINDOW,
    SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED,
    SUDO_SET_LOCK_REDUCTION_INTERVAL,
    SUDO_SET_NETWORK_RATE_LIMIT,
    SUDO_SET_TEMPO,
    SUDO_SET_WEIGHTS_SET_RATE_LIMIT,
    NETUID,
)


def _setup_subnet(subtensor, wallet, tempo=None, admin_freeze_window=0):
    """Register and activate a subnet with the given tempo and freeze window."""
    if tempo is None:
        tempo = subtensor.chain.get_min_tempo()
    sn = TestSubnet(subtensor)
    sn.execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(wallet, AdminUtils, True, admin_freeze_window),
            SUDO_SET_NETWORK_RATE_LIMIT(wallet, AdminUtils, True, 0),
            SUDO_SET_LOCK_REDUCTION_INTERVAL(wallet, AdminUtils, True, 1),
        ]
    )
    sn.execute_steps(
        [
            REGISTER_SUBNET(wallet),
            SUDO_SET_TEMPO(wallet, AdminUtils, True, NETUID, tempo),
            ACTIVATE_SUBNET(wallet),
        ]
    )
    return sn


async def _setup_subnet_async(
    async_subtensor, wallet, tempo=None, admin_freeze_window=0
):
    """Register and activate a subnet with the given tempo and freeze window."""
    if tempo is None:
        tempo = await async_subtensor.chain.get_min_tempo()
    sn = TestSubnet(async_subtensor)
    await sn.async_execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(wallet, AdminUtils, True, admin_freeze_window),
            SUDO_SET_NETWORK_RATE_LIMIT(wallet, AdminUtils, True, 0),
            SUDO_SET_LOCK_REDUCTION_INTERVAL(wallet, AdminUtils, True, 1),
        ]
    )
    await sn.async_execute_steps(
        [
            REGISTER_SUBNET(wallet),
            SUDO_SET_TEMPO(wallet, AdminUtils, True, NETUID, tempo),
            ACTIVATE_SUBNET(wallet),
        ]
    )
    return sn


def test_set_tempo(subtensor, alice_wallet):
    """
    Verify owner set_tempo resets LastEpochBlock and matches get_next_epoch_start_block.

    Steps:
        1. Register and activate a subnet.
        2. Call owner set_tempo.
        3. Verify runtime epoch schedule and hyperparameters tempo.
    """
    sn = _setup_subnet(subtensor, alice_wallet)
    netuid = sn.netuid

    new_tempo = subtensor.chain.get_min_tempo() + 10
    result = subtensor.extrinsics.set_tempo(
        wallet=alice_wallet,
        netuid=netuid,
        tempo=new_tempo,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success, result.message

    block_at_set = subtensor.subnets.get_last_epoch_block(netuid)
    assert (
        subtensor.subnets.get_next_epoch_start_block(netuid) == block_at_set + new_tempo
    )

    hp = subtensor.subnets.get_subnet_hyperparameters(netuid)
    assert hp.tempo == new_tempo

    epoch_index = subtensor.subnets.get_subnet_epoch_index(netuid)
    assert isinstance(epoch_index, int) and epoch_index >= 0

    rate_limit = subtensor.chain.get_owner_hyperparam_rate_limit()
    assert isinstance(rate_limit, int) and rate_limit > 0


@pytest.mark.asyncio
async def test_set_tempo_async(async_subtensor, alice_wallet):
    """
    Verify async owner set_tempo resets LastEpochBlock and matches get_next_epoch_start_block.

    Steps:
        1. Register and activate a subnet.
        2. Call owner set_tempo.
        3. Verify runtime epoch schedule and hyperparameters tempo.
    """
    sn = await _setup_subnet_async(async_subtensor, alice_wallet)
    netuid = sn.netuid

    new_tempo = await async_subtensor.chain.get_min_tempo() + 10
    result = await async_subtensor.extrinsics.set_tempo(
        wallet=alice_wallet,
        netuid=netuid,
        tempo=new_tempo,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success, result.message

    block_at_set = await async_subtensor.subnets.get_last_epoch_block(netuid)
    next_start = await async_subtensor.subnets.get_next_epoch_start_block(netuid)
    assert next_start == block_at_set + new_tempo

    hp = await async_subtensor.subnets.get_subnet_hyperparameters(netuid)
    assert hp.tempo == new_tempo

    epoch_index = await async_subtensor.subnets.get_subnet_epoch_index(netuid)
    assert isinstance(epoch_index, int) and epoch_index >= 0

    rate_limit = await async_subtensor.chain.get_owner_hyperparam_rate_limit()
    assert isinstance(rate_limit, int) and rate_limit > 0


def test_trigger_epoch(subtensor, alice_wallet):
    """
    Verify owner trigger_epoch is blocked by commit-reveal and succeeds after disabling it.

    Steps:
        1. Register and activate a subnet with a long tempo (CR enabled by default).
        2. Set admin freeze window.
        3. Attempt trigger_epoch — expect DynamicTempoBlockedByCommitReveal rejection.
        4. Disable commit-reveal.
        5. Call trigger_epoch again — expect success.
        6. Verify pending epoch and freeze window state.
    """
    max_tempo = subtensor.chain.get_max_tempo()
    sn = _setup_subnet(subtensor, alice_wallet, tempo=max_tempo, admin_freeze_window=0)
    netuid = sn.netuid

    sn.execute_steps([SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 10)])
    freeze_window = subtensor.chain.get_admin_freeze_window()

    result = subtensor.extrinsics.trigger_epoch(
        wallet=alice_wallet,
        netuid=netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success is False

    sn.execute_steps(
        [
            SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
                alice_wallet, AdminUtils, True, NETUID, False
            )
        ]
    )

    result = subtensor.extrinsics.trigger_epoch(
        wallet=alice_wallet,
        netuid=netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success, result.message

    pending = subtensor.subnets.get_pending_epoch_at(netuid)
    assert pending > 0
    trigger_block = pending - freeze_window
    assert pending == trigger_block + freeze_window
    assert pending > subtensor.block
    assert subtensor.chain.is_in_admin_freeze_window(netuid) is True


@pytest.mark.asyncio
async def test_trigger_epoch_async(async_subtensor, alice_wallet):
    """
    Verify async owner trigger_epoch is blocked by commit-reveal and succeeds after disabling it.

    Steps:
        1. Register and activate a subnet with a long tempo (CR enabled by default).
        2. Set admin freeze window.
        3. Attempt trigger_epoch — expect DynamicTempoBlockedByCommitReveal rejection.
        4. Disable commit-reveal.
        5. Call trigger_epoch again — expect success.
        6. Verify pending epoch and freeze window state.
    """
    max_tempo = await async_subtensor.chain.get_max_tempo()
    sn = await _setup_subnet_async(
        async_subtensor, alice_wallet, tempo=max_tempo, admin_freeze_window=0
    )
    netuid = sn.netuid

    await sn.async_execute_steps(
        [SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 10)]
    )
    freeze_window = await async_subtensor.chain.get_admin_freeze_window()

    result = await async_subtensor.extrinsics.trigger_epoch(
        wallet=alice_wallet,
        netuid=netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success is False

    await sn.async_execute_steps(
        [
            SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
                alice_wallet, AdminUtils, True, NETUID, False
            )
        ]
    )

    result = await async_subtensor.extrinsics.trigger_epoch(
        wallet=alice_wallet,
        netuid=netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success, result.message

    pending = await async_subtensor.subnets.get_pending_epoch_at(netuid)
    assert pending > 0
    trigger_block = pending - freeze_window
    assert pending == trigger_block + freeze_window
    assert pending > await async_subtensor.block
    assert await async_subtensor.chain.is_in_admin_freeze_window(netuid) is True


def test_set_activity_cutoff_factor(subtensor, alice_wallet):
    """
    Verify owner set_activity_cutoff_factor updates hyperparameters.

    Steps:
        1. Register and activate a subnet.
        2. Call owner set_activity_cutoff_factor.
        3. Verify activity_cutoff_factor in hyperparameters.
    """
    sn = _setup_subnet(subtensor, alice_wallet)
    netuid = sn.netuid

    new_factor = 5_000
    result = subtensor.extrinsics.set_activity_cutoff_factor(
        wallet=alice_wallet,
        netuid=netuid,
        factor_milli=new_factor,
    )
    assert result.success, result.message

    hp = subtensor.subnets.get_subnet_hyperparameters(netuid)
    assert hp.activity_cutoff_factor == new_factor

    factor = subtensor.subnets.get_activity_cutoff_factor_milli(netuid)
    assert factor == new_factor


@pytest.mark.asyncio
async def test_set_activity_cutoff_factor_async(async_subtensor, alice_wallet):
    """
    Verify async owner set_activity_cutoff_factor updates hyperparameters.

    Steps:
        1. Register and activate a subnet.
        2. Call owner set_activity_cutoff_factor.
        3. Verify activity_cutoff_factor in hyperparameters.
    """
    sn = await _setup_subnet_async(async_subtensor, alice_wallet)
    netuid = sn.netuid

    new_factor = 5_000
    result = await async_subtensor.extrinsics.set_activity_cutoff_factor(
        wallet=alice_wallet,
        netuid=netuid,
        factor_milli=new_factor,
    )
    assert result.success, result.message

    hp = await async_subtensor.subnets.get_subnet_hyperparameters(netuid)
    assert hp.activity_cutoff_factor == new_factor

    factor = await async_subtensor.subnets.get_activity_cutoff_factor_milli(netuid)
    assert factor == new_factor


def test_commit_reveal_after_owner_set_tempo(subtensor, alice_wallet):
    """
    Verify commit-reveal weights after owner set_tempo (not sudo tempo).

    Steps:
        1. Register and activate a subnet with commit-reveal enabled.
        2. Call owner set_tempo.
        3. Commit and reveal weights using CRv4 schedule.
    """
    BLOCK_TIME = 0.25 if subtensor.chain.is_fast_blocks() else 12.0
    logging.console.info(f"Using block time: {BLOCK_TIME}")

    max_tempo = subtensor.chain.get_max_tempo()
    min_tempo = subtensor.chain.get_min_tempo()

    sn = TestSubnet(subtensor)
    sn.execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
            SUDO_SET_NETWORK_RATE_LIMIT(alice_wallet, AdminUtils, True, 0),
            REGISTER_SUBNET(alice_wallet),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, max_tempo),
            ACTIVATE_SUBNET(alice_wallet),
            SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
                alice_wallet, AdminUtils, True, NETUID, True
            ),
            SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, 0),
        ]
    )
    netuid = sn.netuid

    owner_tempo = min_tempo
    tempo_result = subtensor.extrinsics.set_tempo(
        wallet=alice_wallet,
        netuid=netuid,
        tempo=owner_tempo,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert tempo_result.success, tempo_result.message
    assert subtensor.subnets.get_subnet_hyperparameters(netuid).tempo == owner_tempo

    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.1], dtype=np.float32)
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )

    current_block = subtensor.chain.get_current_block()
    upcoming_tempo = subtensor.subnets.get_next_epoch_start_block(netuid)
    if upcoming_tempo - current_block < 6:
        sn.wait_next_epoch()
    current_block = subtensor.chain.get_current_block()
    upcoming_tempo = subtensor.subnets.get_next_epoch_start_block(netuid)
    logging.console.info(
        f"Current block: {current_block}, next epoch: {upcoming_tempo}"
    )

    expected_commit_block = subtensor.block + 1
    response = subtensor.extrinsics.set_weights(
        wallet=alice_wallet,
        netuid=netuid,
        mechid=0,
        uids=weight_uids,
        weights=weight_vals,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        block_time=BLOCK_TIME,
        period=16,
        raise_error=True,
    )
    assert response.success is True, response.message
    expected_reveal_round = response.data.get("reveal_round")
    assert expected_reveal_round is not None

    subtensor.wait_for_block(subtensor.block + 1)

    commits_on_chain = subtensor.commitments.get_timelocked_weight_commits(
        netuid=netuid, mechid=0
    )
    address, commit_block, _commit, reveal_round = commits_on_chain[0]
    assert expected_reveal_round == reveal_round
    assert address == alice_wallet.hotkey.ss58_address
    assert expected_commit_block in [
        commit_block - 1,
        commit_block,
        commit_block + 1,
    ]
    assert subtensor.subnets.weights(netuid=netuid, mechid=0) == []

    expected_reveal_block = subtensor.subnets.get_next_epoch_start_block(netuid) + 5
    subtensor.wait_for_block(expected_reveal_block)

    latest_drand_round = 0
    while latest_drand_round <= expected_reveal_round:
        latest_drand_round = subtensor.chain.last_drand_round()
        time.sleep(3)

    subnet_weights = subtensor.subnets.weights(netuid=netuid, mechid=0)
    assert subnet_weights != []
    revealed_weights = subnet_weights[0][1]
    assert weight_uids[0] == revealed_weights[0][0]
    assert weight_vals[0] == revealed_weights[0][1]
    assert (
        subtensor.commitments.get_timelocked_weight_commits(netuid=netuid, mechid=0)
        == []
    )


@pytest.mark.asyncio
async def test_commit_reveal_after_owner_set_tempo_async(async_subtensor, alice_wallet):
    """
    Verify async commit-reveal weights after owner set_tempo (not sudo tempo).

    Steps:
        1. Register and activate a subnet with commit-reveal enabled.
        2. Call owner set_tempo.
        3. Commit and reveal weights using CRv4 schedule.
    """
    BLOCK_TIME = 0.25 if await async_subtensor.chain.is_fast_blocks() else 12.0
    logging.console.info(f"Using block time: {BLOCK_TIME}")

    max_tempo = await async_subtensor.chain.get_max_tempo()
    min_tempo = await async_subtensor.chain.get_min_tempo()

    sn = TestSubnet(async_subtensor)
    await sn.async_execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
            SUDO_SET_NETWORK_RATE_LIMIT(alice_wallet, AdminUtils, True, 0),
            REGISTER_SUBNET(alice_wallet),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, max_tempo),
            ACTIVATE_SUBNET(alice_wallet),
            SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
                alice_wallet, AdminUtils, True, NETUID, True
            ),
            SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, 0),
        ]
    )
    netuid = sn.netuid

    owner_tempo = min_tempo
    tempo_result = await async_subtensor.extrinsics.set_tempo(
        wallet=alice_wallet,
        netuid=netuid,
        tempo=owner_tempo,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert tempo_result.success, tempo_result.message
    hp = await async_subtensor.subnets.get_subnet_hyperparameters(netuid)
    assert hp.tempo == owner_tempo

    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.1], dtype=np.float32)
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )

    current_block = await async_subtensor.chain.get_current_block()
    upcoming_tempo = await async_subtensor.subnets.get_next_epoch_start_block(netuid)
    if upcoming_tempo - current_block < 6:
        await sn.wait_next_epoch()
    current_block = await async_subtensor.chain.get_current_block()
    upcoming_tempo = await async_subtensor.subnets.get_next_epoch_start_block(netuid)
    logging.console.info(
        f"Current block: {current_block}, next epoch: {upcoming_tempo}"
    )

    expected_commit_block = await async_subtensor.block + 1
    response = await async_subtensor.extrinsics.set_weights(
        wallet=alice_wallet,
        netuid=netuid,
        mechid=0,
        uids=weight_uids,
        weights=weight_vals,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        block_time=BLOCK_TIME,
        period=16,
        raise_error=True,
    )
    assert response.success is True, response.message
    expected_reveal_round = response.data.get("reveal_round")
    assert expected_reveal_round is not None

    await async_subtensor.wait_for_block(await async_subtensor.block + 1)

    commits_on_chain = await async_subtensor.commitments.get_timelocked_weight_commits(
        netuid=netuid, mechid=0
    )
    address, commit_block, _commit, reveal_round = commits_on_chain[0]
    assert expected_reveal_round == reveal_round
    assert address == alice_wallet.hotkey.ss58_address
    assert expected_commit_block in [
        commit_block - 1,
        commit_block,
        commit_block + 1,
    ]
    assert await async_subtensor.subnets.weights(netuid=netuid, mechid=0) == []

    expected_reveal_block = (
        await async_subtensor.subnets.get_next_epoch_start_block(netuid) + 5
    )
    await async_subtensor.wait_for_block(expected_reveal_block)

    latest_drand_round = 0
    while latest_drand_round <= expected_reveal_round:
        latest_drand_round = await async_subtensor.chain.last_drand_round()
        time.sleep(3)

    subnet_weights = await async_subtensor.subnets.weights(netuid=netuid, mechid=0)
    assert subnet_weights != []
    revealed_weights = subnet_weights[0][1]
    assert weight_uids[0] == revealed_weights[0][0]
    assert weight_vals[0] == revealed_weights[0][1]
    assert (
        await async_subtensor.commitments.get_timelocked_weight_commits(
            netuid=netuid, mechid=0
        )
        == []
    )


def test_root_set_activity_cutoff_factor(subtensor, alice_wallet):
    """
    Verify sudo root_set_activity_cutoff_factor overrides the owner-level value.

    Steps:
        1. Register and activate a subnet.
        2. Call root_set_activity_cutoff_factor via sudo.
        3. Verify the factor is updated in hyperparameters and direct query.
    """
    sn = _setup_subnet(subtensor, alice_wallet)
    netuid = sn.netuid

    new_factor = subtensor.chain.get_min_activity_cutoff_factor_milli() + 500
    result = subtensor.extrinsics.root_set_activity_cutoff_factor(
        wallet=alice_wallet,
        netuid=netuid,
        factor_milli=new_factor,
    )
    assert result.success, result.message

    hp = subtensor.subnets.get_subnet_hyperparameters(netuid)
    assert hp.activity_cutoff_factor == new_factor

    factor = subtensor.subnets.get_activity_cutoff_factor_milli(netuid)
    assert factor == new_factor


@pytest.mark.asyncio
async def test_root_set_activity_cutoff_factor_async(async_subtensor, alice_wallet):
    """
    Verify async sudo root_set_activity_cutoff_factor overrides the owner-level value.

    Steps:
        1. Register and activate a subnet.
        2. Call root_set_activity_cutoff_factor via sudo.
        3. Verify the factor is updated in hyperparameters and direct query.
    """
    sn = await _setup_subnet_async(async_subtensor, alice_wallet)
    netuid = sn.netuid

    new_factor = (
        await async_subtensor.chain.get_min_activity_cutoff_factor_milli() + 500
    )
    result = await async_subtensor.extrinsics.root_set_activity_cutoff_factor(
        wallet=alice_wallet,
        netuid=netuid,
        factor_milli=new_factor,
    )
    assert result.success, result.message

    hp = await async_subtensor.subnets.get_subnet_hyperparameters(netuid)
    assert hp.activity_cutoff_factor == new_factor

    factor = await async_subtensor.subnets.get_activity_cutoff_factor_milli(netuid)
    assert factor == new_factor


def test_tempo_control_negative_cases(subtensor, alice_wallet, bob_wallet):
    """
    Verify negative scenarios for tempo control extrinsics.

    Steps:
        1. Register and activate a subnet (owner = alice).
        2. Non-owner (bob) attempts set_tempo — expect failure.
        3. Owner sets tempo below chain minimum — expect failure.
        4. Owner sets tempo above chain maximum — expect failure.
        5. Owner sets activity cutoff factor above chain maximum — expect failure.
        6. Owner sets activity cutoff factor below chain minimum — expect failure.
    """
    sn = _setup_subnet(subtensor, alice_wallet)
    netuid = sn.netuid

    min_tempo = subtensor.chain.get_min_tempo()
    max_tempo = subtensor.chain.get_max_tempo()
    min_cutoff = subtensor.chain.get_min_activity_cutoff_factor_milli()
    max_cutoff = subtensor.chain.get_max_activity_cutoff_factor_milli()

    result = subtensor.extrinsics.set_tempo(
        wallet=bob_wallet,
        netuid=netuid,
        tempo=min_tempo + 5,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success is False

    result = subtensor.extrinsics.set_tempo(
        wallet=alice_wallet,
        netuid=netuid,
        tempo=min_tempo - 1,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success is False

    result = subtensor.extrinsics.set_tempo(
        wallet=alice_wallet,
        netuid=netuid,
        tempo=max_tempo + 1,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success is False

    result = subtensor.extrinsics.set_activity_cutoff_factor(
        wallet=alice_wallet,
        netuid=netuid,
        factor_milli=max_cutoff + 1,
    )
    assert result.success is False

    result = subtensor.extrinsics.set_activity_cutoff_factor(
        wallet=alice_wallet,
        netuid=netuid,
        factor_milli=min_cutoff - 1,
    )
    assert result.success is False


@pytest.mark.asyncio
async def test_tempo_control_negative_cases_async(
    async_subtensor, alice_wallet, bob_wallet
):
    """
    Verify async negative scenarios for tempo control extrinsics.

    Steps:
        1. Register and activate a subnet (owner = alice).
        2. Non-owner (bob) attempts set_tempo — expect failure.
        3. Owner sets tempo below chain minimum — expect failure.
        4. Owner sets tempo above chain maximum — expect failure.
        5. Owner sets activity cutoff factor above chain maximum — expect failure.
        6. Owner sets activity cutoff factor below chain minimum — expect failure.
    """
    sn = await _setup_subnet_async(async_subtensor, alice_wallet)
    netuid = sn.netuid

    min_tempo = await async_subtensor.chain.get_min_tempo()
    max_tempo = await async_subtensor.chain.get_max_tempo()
    min_cutoff = await async_subtensor.chain.get_min_activity_cutoff_factor_milli()
    max_cutoff = await async_subtensor.chain.get_max_activity_cutoff_factor_milli()

    result = await async_subtensor.extrinsics.set_tempo(
        wallet=bob_wallet,
        netuid=netuid,
        tempo=min_tempo + 5,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success is False

    result = await async_subtensor.extrinsics.set_tempo(
        wallet=alice_wallet,
        netuid=netuid,
        tempo=min_tempo - 1,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success is False

    result = await async_subtensor.extrinsics.set_tempo(
        wallet=alice_wallet,
        netuid=netuid,
        tempo=max_tempo + 1,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success is False

    result = await async_subtensor.extrinsics.set_activity_cutoff_factor(
        wallet=alice_wallet,
        netuid=netuid,
        factor_milli=max_cutoff + 1,
    )
    assert result.success is False

    result = await async_subtensor.extrinsics.set_activity_cutoff_factor(
        wallet=alice_wallet,
        netuid=netuid,
        factor_milli=min_cutoff - 1,
    )
    assert result.success is False


def test_set_tempo_rejected_in_freeze_window(subtensor, alice_wallet):
    """
    Verify set_tempo is rejected when the subnet is in admin freeze window.

    Steps:
        1. Register and activate a subnet with long tempo and no freeze window.
        2. Set admin freeze window, trigger epoch to enter freeze state.
        3. Attempt set_tempo while frozen — expect failure.
    """
    max_tempo = subtensor.chain.get_max_tempo()
    min_tempo = subtensor.chain.get_min_tempo()
    sn = _setup_subnet(subtensor, alice_wallet, tempo=max_tempo, admin_freeze_window=0)
    netuid = sn.netuid

    sn.execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 10),
            SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
                alice_wallet, AdminUtils, True, NETUID, False
            ),
        ]
    )

    result = subtensor.extrinsics.trigger_epoch(
        wallet=alice_wallet,
        netuid=netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success, result.message
    assert subtensor.chain.is_in_admin_freeze_window(netuid) is True

    result = subtensor.extrinsics.set_tempo(
        wallet=alice_wallet,
        netuid=netuid,
        tempo=min_tempo,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success is False


@pytest.mark.asyncio
async def test_set_tempo_rejected_in_freeze_window_async(async_subtensor, alice_wallet):
    """
    Verify async set_tempo is rejected when the subnet is in admin freeze window.

    Steps:
        1. Register and activate a subnet with long tempo and no freeze window.
        2. Set admin freeze window, trigger epoch to enter freeze state.
        3. Attempt set_tempo while frozen — expect failure.
    """
    max_tempo = await async_subtensor.chain.get_max_tempo()
    min_tempo = await async_subtensor.chain.get_min_tempo()
    sn = await _setup_subnet_async(
        async_subtensor, alice_wallet, tempo=max_tempo, admin_freeze_window=0
    )
    netuid = sn.netuid

    await sn.async_execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 10),
            SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
                alice_wallet, AdminUtils, True, NETUID, False
            ),
        ]
    )

    result = await async_subtensor.extrinsics.trigger_epoch(
        wallet=alice_wallet,
        netuid=netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success, result.message
    assert await async_subtensor.chain.is_in_admin_freeze_window(netuid) is True

    result = await async_subtensor.extrinsics.set_tempo(
        wallet=alice_wallet,
        netuid=netuid,
        tempo=min_tempo,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result.success is False
