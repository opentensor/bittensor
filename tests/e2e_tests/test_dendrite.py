import asyncio

import pytest

from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils import (
    TestSubnet,
    AdminUtils,
    NETUID,
    ACTIVATE_SUBNET,
    REGISTER_NEURON,
    REGISTER_SUBNET,
    SUDO_SET_ADMIN_FREEZE_WINDOW,
    SUDO_SET_TEMPO,
    SUDO_SET_MAX_ALLOWED_VALIDATORS,
    SUDO_SET_WEIGHTS_SET_RATE_LIMIT,
)

FAST_RUNTIME_TEMPO = 100
NON_FAST_RUNTIME_TEMPO = 10


@pytest.mark.asyncio
async def test_dendrite(subtensor, templates, alice_wallet, bob_wallet):
    """
    Test the Dendrite mechanism

    Steps:
        1. Register a subnet through Alice
        2. Register Bob as a validator
        3. Add stake to Bob and ensure neuron is not a validator yet
        4. Run Bob as a validator and wait epoch
        5. Ensure Bob's neuron has all correct attributes of a validator
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    TEMPO_TO_SET = (
        FAST_RUNTIME_TEMPO
        if subtensor.chain.is_fast_blocks()
        else NON_FAST_RUNTIME_TEMPO
    )
    alice_sn = TestSubnet(subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(alice_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
        ACTIVATE_SUBNET(alice_wallet),
        SUDO_SET_MAX_ALLOWED_VALIDATORS(alice_wallet, AdminUtils, True, NETUID, 1),
        SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, 10),
        REGISTER_NEURON(bob_wallet),
    ]
    alice_sn.execute_steps(steps)

    if not subtensor.chain.is_fast_blocks():
        # Make sure Alice is Top Validator (for non-fast-runtime only)
        assert subtensor.staking.add_stake(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(1),
        ).success

    metagraph = subtensor.metagraphs.metagraph(alice_sn.netuid)

    # Assert neurons are Alice and Bob
    assert len(metagraph.neurons) == 2

    alice_neuron = next(
        n for n in metagraph.neurons if n.hotkey == alice_wallet.hotkey.ss58_address
    )
    assert alice_neuron.hotkey == alice_wallet.hotkey.ss58_address
    assert alice_neuron.coldkey == alice_wallet.coldkey.ss58_address

    bob_neuron = next(
        n for n in metagraph.neurons if n.hotkey == bob_wallet.hotkey.ss58_address
    )
    assert bob_neuron.hotkey == bob_wallet.hotkey.ss58_address
    assert bob_neuron.coldkey == bob_wallet.coldkey.ss58_address

    # Assert stake is 0
    assert bob_neuron.stake.tao == 0

    # Stake to become to top neuron after the first epoch
    tao = Balance.from_tao(10_000)
    alpha, _ = subtensor.subnets.subnet(alice_sn.netuid).tao_to_alpha_with_slippage(tao)

    assert subtensor.staking.add_stake(
        wallet=bob_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=tao,
    ).success, "Unable to stake to Bob."

    # Waiting to give the chain a chance to update its state
    subtensor.wait_for_block()

    # Refresh metagraph
    metagraph = subtensor.metagraphs.metagraph(alice_sn.netuid)
    bob_neuron = next(
        n for n in metagraph.neurons if n.hotkey == bob_wallet.hotkey.ss58_address
    )

    logging.console.info(
        f"block: {subtensor.block}, bob_neuron.stake.rao: {bob_neuron.stake.rao}, "
        f"alpha.rao: {alpha.rao}, division: {bob_neuron.stake.rao / alpha.rao}"
    )
    # Assert alpha is close to stake equivalent
    assert 0.95 < bob_neuron.stake.rao / alpha.rao < 1.05

    # Assert neuron is not a validator yet
    assert bob_neuron.active is True
    assert bob_neuron.validator_permit is False
    assert bob_neuron.validator_trust == 0.0
    assert bob_neuron.pruning_score == 0

    async with templates.validator(bob_wallet, alice_sn.netuid):
        await asyncio.sleep(5)  # wait for 5 seconds for the Validator to process

        subtensor.wait_for_block(
            subtensor.subnets.get_next_epoch_start_block(alice_sn.netuid) + 1
        )

        # Refresh metagraph
        metagraph = subtensor.metagraphs.metagraph(alice_sn.netuid)

    # Refresh validator neuron
    updated_neuron = next(
        n for n in metagraph.neurons if n.hotkey == bob_wallet.hotkey.ss58_address
    )

    assert len(metagraph.neurons) == 2
    assert updated_neuron.active is True
    assert updated_neuron.validator_permit is True
    assert updated_neuron.hotkey == bob_wallet.hotkey.ss58_address
    assert updated_neuron.coldkey == bob_wallet.coldkey.ss58_address
    assert updated_neuron.pruning_score != 0


@pytest.mark.asyncio
async def test_dendrite_async(async_subtensor, templates, alice_wallet, bob_wallet):
    """
    Test the Dendrite mechanism

    Steps:
        1. Register a subnet through Alice
        2. Register Bob as a validator
        3. Add stake to Bob and ensure neuron is not a validator yet
        4. Run Bob as a validator and wait epoch
        5. Ensure Bob's neuron has all correct attributes of a validator
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    TEMPO_TO_SET = (
        FAST_RUNTIME_TEMPO
        if await async_subtensor.chain.is_fast_blocks()
        else NON_FAST_RUNTIME_TEMPO
    )
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(alice_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
        ACTIVATE_SUBNET(alice_wallet),
        SUDO_SET_MAX_ALLOWED_VALIDATORS(alice_wallet, AdminUtils, True, NETUID, 1),
        SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, 10),
        REGISTER_NEURON(bob_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    metagraph = await async_subtensor.metagraphs.metagraph(alice_sn.netuid)

    # Assert neurons are Alice and Bob
    assert len(metagraph.neurons) == 2

    alice_neuron = next(
        n for n in metagraph.neurons if n.hotkey == alice_wallet.hotkey.ss58_address
    )
    assert alice_neuron.hotkey == alice_wallet.hotkey.ss58_address
    assert alice_neuron.coldkey == alice_wallet.coldkey.ss58_address

    bob_neuron = next(
        n for n in metagraph.neurons if n.hotkey == bob_wallet.hotkey.ss58_address
    )
    assert bob_neuron.hotkey == bob_wallet.hotkey.ss58_address
    assert bob_neuron.coldkey == bob_wallet.coldkey.ss58_address

    # Assert stake is 0
    assert bob_neuron.stake.tao == 0

    # Stake to become to top neuron after the first epoch
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
            wait_for_inclusion=False,
            wait_for_finalization=False,
        )
    ).success, "Unable to stake to Bob."

    # Waiting to give the chain a chance to update its state
    await async_subtensor.wait_for_block()

    # Refresh metagraph
    metagraph = await async_subtensor.metagraphs.metagraph(alice_sn.netuid)
    bob_neuron = next(
        n for n in metagraph.neurons if n.hotkey == bob_wallet.hotkey.ss58_address
    )

    logging.console.info(
        f"block: {await async_subtensor.block}, bob_neuron.stake.rao: {bob_neuron.stake.rao}, alpha.rao: {alpha.rao}, division: {bob_neuron.stake.rao / alpha.rao}"
    )

    # Assert alpha is close to stake equivalent
    assert 0.95 < bob_neuron.stake.rao / alpha.rao < 1.05

    # Assert neuron is not a validator yet
    assert bob_neuron.active is True
    assert bob_neuron.validator_permit is False
    assert bob_neuron.validator_trust == 0.0
    assert bob_neuron.pruning_score == 0

    async with templates.validator(bob_wallet, alice_sn.netuid):
        await asyncio.sleep(5)  # wait for 5 seconds for the Validator to process

        await async_subtensor.wait_for_block(
            await async_subtensor.subnets.get_next_epoch_start_block(alice_sn.netuid)
            + 1
        )

        # Refresh metagraph
        metagraph = await async_subtensor.metagraphs.metagraph(alice_sn.netuid)

    # Refresh validator neuron
    updated_neuron = next(
        n for n in metagraph.neurons if n.hotkey == bob_wallet.hotkey.ss58_address
    )

    assert len(metagraph.neurons) == 2
    assert updated_neuron.active is True
    assert updated_neuron.validator_permit is True
    assert updated_neuron.hotkey == bob_wallet.hotkey.ss58_address
    assert updated_neuron.coldkey == bob_wallet.coldkey.ss58_address
    assert updated_neuron.pruning_score != 0
