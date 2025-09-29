import asyncio

import pytest

from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.chain_interactions import (
    # async_sudo_set_admin_utils,
    async_wait_epoch,
    # sudo_set_admin_utils,
    wait_epoch,
)
from bittensor.core.extrinsics.utils import sudo_call_extrinsic
from bittensor.core.extrinsics import sudo
from bittensor.core.extrinsics.asyncex import sudo as async_sudo
from bittensor.core.extrinsics.asyncex.utils import (
    sudo_call_extrinsic as async_sudo_call_extrinsic,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    async_wait_to_start_call,
    wait_to_start_call,
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
    SET_TEMPO = (
        FAST_RUNTIME_TEMPO
        if subtensor.chain.is_fast_blocks()
        else NON_FAST_RUNTIME_TEMPO
    )

    assert sudo.sudo_set_admin_freeze_window_extrinsic(
        subtensor, alice_wallet, 0
    ).success

    alice_subnet_netuid = subtensor.subnets.get_total_subnets()  # 2

    # Register a subnet, netuid 2
    assert subtensor.subnets.register_subnet(alice_wallet).success, (
        "Subnet wasn't created."
    )

    # Verify subnet <netuid> created successfully
    assert subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully."
    )

    assert sudo_call_extrinsic(
        subtensor=subtensor,
        wallet=alice_wallet,
        call_function="sudo_set_tempo",
        call_params={
            "netuid": alice_subnet_netuid,
            "tempo": SET_TEMPO,
        },
    ).success, "Unable to set tempo."

    assert wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid), (
        "Subnet wasn't started."
    )
    assert subtensor.subnets.is_subnet_active(alice_subnet_netuid), (
        "Subnet is not active."
    )

    if not subtensor.chain.is_fast_blocks():
        # Make sure Alice is Top Validator (for non-fast-runtime only)
        assert subtensor.staking.add_stake(
            wallet=alice_wallet,
            netuid=alice_subnet_netuid,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(1),
        ).success

    # update max_allowed_validators so only one neuron can get validator_permit
    assert sudo_call_extrinsic(
        subtensor=subtensor,
        wallet=alice_wallet,
        call_function="sudo_set_max_allowed_validators",
        call_params={
            "netuid": alice_subnet_netuid,
            "max_allowed_validators": 1,
        },
    ).success, "Unable to set max_allowed_validators."

    # update weights_set_rate_limit for fast-blocks
    assert sudo_call_extrinsic(
        subtensor=subtensor,
        wallet=alice_wallet,
        call_function="sudo_set_weights_set_rate_limit",
        call_params={
            "netuid": alice_subnet_netuid,
            "weights_set_rate_limit": 10,
        },
    ).success, "Unable to set weights_set_rate_limit."

    # Register Bob to the network
    assert subtensor.subnets.burned_register(bob_wallet, alice_subnet_netuid).success, (
        "Unable to register Bob as a neuron."
    )

    metagraph = subtensor.metagraphs.metagraph(alice_subnet_netuid)

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
    alpha, _ = subtensor.subnets.subnet(alice_subnet_netuid).tao_to_alpha_with_slippage(
        tao
    )

    assert subtensor.staking.add_stake(
        wallet=bob_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=tao,
    ).success, "Unable to stake to Bob."

    # Waiting to give the chain a chance to update its state
    subtensor.wait_for_block()

    # Refresh metagraph
    metagraph = subtensor.metagraphs.metagraph(alice_subnet_netuid)
    bob_neuron = next(
        n for n in metagraph.neurons if n.hotkey == bob_wallet.hotkey.ss58_address
    )

    logging.console.info(
        f"block: {subtensor.block}, bob_neuron.stake.rao: {bob_neuron.stake.rao}, alpha.rao: {alpha.rao}, division: {bob_neuron.stake.rao / alpha.rao}"
    )
    # Assert alpha is close to stake equivalent
    assert 0.95 < bob_neuron.stake.rao / alpha.rao < 1.05

    # Assert neuron is not a validator yet
    assert bob_neuron.active is True
    assert bob_neuron.validator_permit is False
    assert bob_neuron.validator_trust == 0.0
    assert bob_neuron.pruning_score == 0

    async with templates.validator(bob_wallet, alice_subnet_netuid):
        await asyncio.sleep(5)  # wait for 5 seconds for the Validator to process

        subtensor.wait_for_block(
            subtensor.subnets.get_next_epoch_start_block(alice_subnet_netuid) + 1
        )

        # Refresh metagraph
        metagraph = subtensor.metagraphs.metagraph(alice_subnet_netuid)

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
    SET_TEMPO = (
        FAST_RUNTIME_TEMPO
        if await async_subtensor.chain.is_fast_blocks()
        else NON_FAST_RUNTIME_TEMPO
    )

    assert (
        await async_sudo.sudo_set_admin_freeze_window_extrinsic(
            async_subtensor, alice_wallet, 0
        )
    ).success

    alice_subnet_netuid = await async_subtensor.subnets.get_total_subnets()  # 2

    # Register a subnet, netuid 2
    assert (await async_subtensor.subnets.register_subnet(alice_wallet)).success, (
        "Subnet wasn't created."
    )

    # Verify subnet <netuid> created successfully
    assert await async_subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully."
    )

    assert (
        await async_sudo_call_extrinsic(
            subtensor=async_subtensor,
            wallet=alice_wallet,
            call_function="sudo_set_tempo",
            call_params={
                "netuid": alice_subnet_netuid,
                "tempo": SET_TEMPO,
            },
        )
    ).success, "Unable to set tempo."

    assert await async_wait_to_start_call(
        async_subtensor, alice_wallet, alice_subnet_netuid
    ), "Subnet wasn't started."

    assert await async_subtensor.subnets.is_subnet_active(alice_subnet_netuid), (
        "Subnet is not active."
    )

    if not await async_subtensor.chain.is_fast_blocks():
        # Make sure Alice is Top Validator (for non-fast-runtime only)
        assert (
            await async_subtensor.staking.add_stake(
                wallet=alice_wallet,
                netuid=alice_subnet_netuid,
                hotkey_ss58=alice_wallet.hotkey.ss58_address,
                amount=Balance.from_tao(5),
                wait_for_inclusion=False,
                wait_for_finalization=False,
            )
        ).success

    # update max_allowed_validators so only one neuron can get validator_permit
    assert (
        await async_sudo_call_extrinsic(
            subtensor=async_subtensor,
            wallet=alice_wallet,
            call_function="sudo_set_max_allowed_validators",
            call_params={
                "netuid": alice_subnet_netuid,
                "max_allowed_validators": 1,
            },
        )
    ).success, "Unable to set max_allowed_validators."

    # update weights_set_rate_limit for fast-blocks
    assert (
        await async_sudo_call_extrinsic(
            subtensor=async_subtensor,
            wallet=alice_wallet,
            call_function="sudo_set_weights_set_rate_limit",
            call_params={
                "netuid": alice_subnet_netuid,
                "weights_set_rate_limit": 10,
            },
        )
    ).success, "Unable to set weights_set_rate_limit."

    # Register Bob to the network
    assert (
        await async_subtensor.subnets.burned_register(bob_wallet, alice_subnet_netuid)
    ).success, "Unable to register Bob as a neuron."

    metagraph = await async_subtensor.metagraphs.metagraph(alice_subnet_netuid)

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
        await async_subtensor.subnets.subnet(alice_subnet_netuid)
    ).tao_to_alpha_with_slippage(tao)

    assert (
        await async_subtensor.staking.add_stake(
            wallet=bob_wallet,
            netuid=alice_subnet_netuid,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            amount=tao,
            wait_for_inclusion=False,
            wait_for_finalization=False,
        )
    ).success, "Unable to stake to Bob."

    # Waiting to give the chain a chance to update its state
    await async_subtensor.wait_for_block()

    # Refresh metagraph
    metagraph = await async_subtensor.metagraphs.metagraph(alice_subnet_netuid)
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

    async with templates.validator(bob_wallet, alice_subnet_netuid):
        await asyncio.sleep(5)  # wait for 5 seconds for the Validator to process

        await async_subtensor.wait_for_block(
            await async_subtensor.subnets.get_next_epoch_start_block(
                alice_subnet_netuid
            )
            + 1
        )

        # Refresh metagraph
        metagraph = await async_subtensor.metagraphs.metagraph(alice_subnet_netuid)

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
