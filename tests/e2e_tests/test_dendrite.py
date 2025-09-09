import asyncio

import pytest

from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.chain_interactions import (
    async_sudo_set_admin_utils,
    async_wait_epoch,
    sudo_set_admin_utils,
    wait_epoch,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    async_wait_to_start_call,
    wait_to_start_call,
)

logging.on()
logging.set_debug()

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
    logging.console.info("Testing `test_dendrite`.")

    alice_subnet_netuid = subtensor.subnets.get_total_subnets()  # 2

    # Register a subnet, netuid 2
    assert subtensor.subnets.register_subnet(alice_wallet), "Subnet wasn't created."

    # Verify subnet <netuid> created successfully
    assert subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully."
    )

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
        # set tempo to 10 block for non-fast-runtime
        assert sudo_set_admin_utils(
            substrate=subtensor.substrate,
            wallet=alice_wallet,
            call_function="sudo_set_tempo",
            call_params={
                "netuid": alice_subnet_netuid,
                "tempo": NON_FAST_RUNTIME_TEMPO,
            },
        )

    # update max_allowed_validators so only one neuron can get validator_permit
    assert sudo_set_admin_utils(
        substrate=subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_max_allowed_validators",
        call_params={
            "netuid": alice_subnet_netuid,
            "max_allowed_validators": 1,
        },
    )

    # update weights_set_rate_limit for fast-blocks
    status, error = sudo_set_admin_utils(
        substrate=subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_weights_set_rate_limit",
        call_params={
            "netuid": alice_subnet_netuid,
            "weights_set_rate_limit": 10,
        },
    )
    assert error is None
    assert status is True

    # Register Bob to the network
    assert subtensor.subnets.burned_register(bob_wallet, alice_subnet_netuid).success, (
        "Unable to register Bob as a neuron"
    )

    metagraph = subtensor.metagraphs.metagraph(alice_subnet_netuid)

    # Assert neurons are Alice and Bob
    assert len(metagraph.neurons) == 2

    alice_neuron = metagraph.neurons[0]
    assert alice_neuron.hotkey == alice_wallet.hotkey.ss58_address
    assert alice_neuron.coldkey == alice_wallet.coldkey.ss58_address

    bob_neuron = metagraph.neurons[1]
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
    ).success

    # Refresh metagraph
    metagraph = subtensor.metagraphs.metagraph(alice_subnet_netuid)
    bob_neuron = metagraph.neurons[1]

    # Assert alpha is close to stake equivalent
    assert 0.95 < bob_neuron.stake.rao / alpha.rao < 1.05

    # Assert neuron is not a validator yet
    assert bob_neuron.active is True
    assert bob_neuron.validator_permit is False
    assert bob_neuron.validator_trust == 0.0
    assert bob_neuron.pruning_score == 0

    async with templates.validator(bob_wallet, alice_subnet_netuid):
        await asyncio.sleep(5)  # wait for 5 seconds for the Validator to process

        await wait_epoch(subtensor, netuid=alice_subnet_netuid)

        # Refresh metagraph
        metagraph = subtensor.metagraphs.metagraph(alice_subnet_netuid)

    # Refresh validator neuron
    updated_neuron = metagraph.neurons[1]

    assert len(metagraph.neurons) == 2
    assert updated_neuron.active is True
    assert updated_neuron.validator_permit is True
    assert updated_neuron.hotkey == bob_wallet.hotkey.ss58_address
    assert updated_neuron.coldkey == bob_wallet.coldkey.ss58_address
    assert updated_neuron.pruning_score != 0

    logging.console.info("✅ Passed `test_dendrite`")


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
    logging.console.info("Testing `test_dendrite_async`.")

    alice_subnet_netuid = await async_subtensor.subnets.get_total_subnets()  # 2

    # Register a subnet, netuid 2
    assert await async_subtensor.subnets.register_subnet(alice_wallet), (
        "Subnet wasn't created"
    )

    # Verify subnet <netuid> created successfully
    assert await async_subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

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
        # set tempo to 10 block for non-fast-runtime
        assert await async_sudo_set_admin_utils(
            substrate=async_subtensor.substrate,
            wallet=alice_wallet,
            call_function="sudo_set_tempo",
            call_params={
                "netuid": alice_subnet_netuid,
                "tempo": NON_FAST_RUNTIME_TEMPO,
            },
        )

    # update max_allowed_validators so only one neuron can get validator_permit
    assert await async_sudo_set_admin_utils(
        substrate=async_subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_max_allowed_validators",
        call_params={
            "netuid": alice_subnet_netuid,
            "max_allowed_validators": 1,
        },
    )

    # update weights_set_rate_limit for fast-blocks
    status, error = await async_sudo_set_admin_utils(
        substrate=async_subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_weights_set_rate_limit",
        call_params={
            "netuid": alice_subnet_netuid,
            "weights_set_rate_limit": 10,
        },
    )
    assert error is None
    assert status is True

    # Register Bob to the network
    assert (
        await async_subtensor.subnets.burned_register(bob_wallet, alice_subnet_netuid)
    ).success, "Unable to register Bob as a neuron"

    metagraph = await async_subtensor.metagraphs.metagraph(alice_subnet_netuid)

    # Assert neurons are Alice and Bob
    assert len(metagraph.neurons) == 2

    alice_neuron = metagraph.neurons[0]
    assert alice_neuron.hotkey == alice_wallet.hotkey.ss58_address
    assert alice_neuron.coldkey == alice_wallet.coldkey.ss58_address

    bob_neuron = metagraph.neurons[1]
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
    ).success

    # Refresh metagraph
    metagraph = await async_subtensor.metagraphs.metagraph(alice_subnet_netuid)
    bob_neuron = metagraph.neurons[1]

    # Assert alpha is close to stake equivalent
    assert 0.95 < bob_neuron.stake.rao / alpha.rao < 1.05

    # Assert neuron is not a validator yet
    assert bob_neuron.active is True
    assert bob_neuron.validator_permit is False
    assert bob_neuron.validator_trust == 0.0
    assert bob_neuron.pruning_score == 0

    async with templates.validator(bob_wallet, alice_subnet_netuid):
        await asyncio.sleep(5)  # wait for 5 seconds for the Validator to process

        await async_wait_epoch(async_subtensor, netuid=alice_subnet_netuid)

        # Refresh metagraph
        metagraph = await async_subtensor.metagraphs.metagraph(alice_subnet_netuid)

    # Refresh validator neuron
    updated_neuron = metagraph.neurons[1]

    assert len(metagraph.neurons) == 2
    assert updated_neuron.active is True
    assert updated_neuron.validator_permit is True
    assert updated_neuron.hotkey == bob_wallet.hotkey.ss58_address
    assert updated_neuron.coldkey == bob_wallet.coldkey.ss58_address
    assert updated_neuron.pruning_score != 0

    logging.console.info("✅ Passed `test_dendrite_async`")
