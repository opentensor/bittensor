import asyncio

import pytest

from bittensor.utils.balance import Balance
from tests.e2e_tests.utils import (
    TestSubnet,
    AdminUtils,
    NETUID,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
    SUDO_SET_DIFFICULTY,
    SUDO_SET_NETWORK_POW_REGISTRATION_ALLOWED,
)

FAST_BLOCKS_SPEEDUP_FACTOR = 5

"""
Verifies:

* root_register()
* neurons()
* register_subnet()
* burned_register()
* immunity_period()
* tempo()
* get_uid_for_hotkey_on_subnet()
* blocks_since_last_update()
* subnetwork_n()
* min_allowed_weights()
* max_weight_limit()
* weights_rate_limit()
* root_set_weights()
* neurons_lite() 
"""


@pytest.mark.asyncio
async def test_root_reg_hyperparams(subtensor, templates, alice_wallet, bob_wallet):
    """
    Test root weights and hyperparameters in the Subtensor network.

    Steps:
        1. Register Alice in the root network (netuid=0).
        2. Create a new subnet (netuid=1) and register Alice on this subnet using burned registration.
        3. Verify that the subnet's `immunity_period` and `tempo` match the default values.
        4. Run Alice as a validator in the background.
        5. Fetch Alice's UID on the subnet and record the blocks since her last update.
        6. Verify that the subnet was created successfully by checking `subnetwork_n`.
        7. Verify hyperparameters related to weights: `min_allowed_weights`, `max_weight_limit`, and `weights_rate_limit`.
        8. Wait until the next epoch and set root weights for netuids 0 and 1.
        9. Verify that the weights are correctly set on the chain.
        10. Adjust hyperparameters to allow proof-of-work (PoW) registration.
        11. Verify that the `blocks_since_last_update` has incremented.
        12. Fetch neurons using `neurons_lite` for the subnet and verify Alice's participation.

    Raises:
        AssertionError: If any of the checks or verifications fail.
    """
    # Default immunity period and tempo set through the subtensor side
    default_immunity_period = 5000
    default_tempo = 10 if subtensor.chain.is_fast_blocks() else 360

    # Register Alice in root network (0)
    assert subtensor.extrinsics.root_register(alice_wallet).success

    # Assert Alice is successfully registered to root
    alice_root_neuron = subtensor.neurons.neurons(netuid=0)[0]
    assert alice_root_neuron.coldkey == alice_wallet.coldkeypub.ss58_address
    assert alice_root_neuron.hotkey == alice_wallet.hotkey.ss58_address

    # Create netuid = 2
    alice_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
    ]
    alice_sn.execute_steps(steps)

    # Ensure correct immunity period and tempo is being fetched
    assert (
        subtensor.subnets.immunity_period(netuid=alice_sn.netuid)
        == default_immunity_period
    )
    assert subtensor.subnets.tempo(netuid=alice_sn.netuid) == default_tempo

    assert subtensor.staking.add_stake(
        wallet=bob_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(1),
        period=16,
    ).success, "Unable to stake from Bob to Alice"

    async with templates.validator(alice_wallet, alice_sn.netuid):
        await asyncio.sleep(5)  # Wait a bit for chain to process data

        # Fetch uid against Alice's hotkey on sn 2 (it will be 0 as she is the only registered neuron)
        alice_uid_sn_2 = subtensor.subnets.get_uid_for_hotkey_on_subnet(
            alice_wallet.hotkey.ss58_address, alice_sn.netuid
        )

        # Fetch the block since last update for the neuron
        block_since_update = subtensor.subnets.blocks_since_last_update(
            netuid=alice_sn.netuid, uid=alice_uid_sn_2
        )
        assert block_since_update is not None

    # Use subnetwork_n hyperparam to check sn creation
    assert subtensor.subnets.subnetwork_n(alice_sn.netuid) == 1  # TODO?
    assert subtensor.subnets.subnetwork_n(alice_sn.netuid + 1) is None

    # Ensure correct hyperparams are being fetched regarding weights
    assert subtensor.subnets.min_allowed_weights(alice_sn.netuid) is not None
    assert subtensor.subnets.max_weight_limit(alice_sn.netuid) is not None
    assert subtensor.subnets.weights_rate_limit(alice_sn.netuid) is not None

    # Wait until next epoch so we can set root weights
    subtensor.wait_for_block(
        subtensor.subnets.get_next_epoch_start_block(alice_sn.netuid) + 1
    )

    # Change hyperparams so we can execute pow_register
    alice_sn.execute_steps(
        [
            SUDO_SET_DIFFICULTY(alice_wallet, AdminUtils, True, NETUID, 1_000_000),
            SUDO_SET_NETWORK_POW_REGISTRATION_ALLOWED(
                alice_wallet, AdminUtils, True, NETUID, True
            ),
        ]
    )

    # Fetch neuron lite for sn one and assert Alice participation
    sn_one_neurons = subtensor.neurons.neurons_lite(netuid=alice_sn.netuid)
    assert (
        sn_one_neurons[alice_uid_sn_2].coldkey == alice_wallet.coldkeypub.ss58_address
    )
    assert sn_one_neurons[alice_uid_sn_2].hotkey == alice_wallet.hotkey.ss58_address
    assert sn_one_neurons[alice_uid_sn_2].validator_permit is True


@pytest.mark.asyncio
async def test_root_reg_hyperparams_async(
    async_subtensor, templates, alice_wallet, bob_wallet
):
    """
    Async test root weights and hyperparameters in the Subtensor network.

    Steps:
        1. Register Alice in the root network (netuid=0).
        2. Create a new subnet (netuid=1) and register Alice on this subnet using burned registration.
        3. Verify that the subnet's `immunity_period` and `tempo` match the default values.
        4. Run Alice as a validator in the background.
        5. Fetch Alice's UID on the subnet and record the blocks since her last update.
        6. Verify that the subnet was created successfully by checking `subnetwork_n`.
        7. Verify hyperparameters related to weights: `min_allowed_weights`, `max_weight_limit`, and `weights_rate_limit`.
        8. Wait until the next epoch and set root weights for netuids 0 and 1.
        9. Verify that the weights are correctly set on the chain.
        10. Adjust hyperparameters to allow proof-of-work (PoW) registration.
        11. Verify that the `blocks_since_last_update` has incremented.
        12. Fetch neurons using `neurons_lite` for the subnet and verify Alice's participation.

    Raises:
        AssertionError: If any of the checks or verifications fail.
    """
    # Default immunity period and tempo set through the subtensor side
    default_immunity_period = 5000
    default_tempo = 10 if await async_subtensor.chain.is_fast_blocks() else 360

    # Register Alice in root network (0)
    assert (await async_subtensor.extrinsics.root_register(alice_wallet)).success

    # Assert Alice is successfully registered to root
    alice_root_neuron = (await async_subtensor.neurons.neurons(netuid=0))[0]
    assert alice_root_neuron.coldkey == alice_wallet.coldkeypub.ss58_address
    assert alice_root_neuron.hotkey == alice_wallet.hotkey.ss58_address

    # Create netuid = 2
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    # Ensure correct immunity period and tempo is being fetched
    assert (
        await async_subtensor.subnets.immunity_period(netuid=alice_sn.netuid)
        == default_immunity_period
    )
    assert await async_subtensor.subnets.tempo(netuid=alice_sn.netuid) == default_tempo

    assert (
        await async_subtensor.staking.add_stake(
            wallet=bob_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(1),
            period=16,
        )
    ).success, "Unable to stake from Bob to Alice"

    async with templates.validator(alice_wallet, alice_sn.netuid):
        await asyncio.sleep(5)  # Wait a bit for chain to process data

        # Fetch uid against Alice's hotkey on sn 2 (it will be 0 as she is the only registered neuron)
        alice_uid_sn_2 = await async_subtensor.subnets.get_uid_for_hotkey_on_subnet(
            alice_wallet.hotkey.ss58_address, alice_sn.netuid
        )

        # Fetch the block since last update for the neuron
        block_since_update = await async_subtensor.subnets.blocks_since_last_update(
            netuid=alice_sn.netuid, uid=alice_uid_sn_2
        )
        assert block_since_update is not None

    # Use subnetwork_n hyperparam to check sn creation
    assert await async_subtensor.subnets.subnetwork_n(alice_sn.netuid) == 1  # TODO?
    assert await async_subtensor.subnets.subnetwork_n(alice_sn.netuid + 1) is None

    # Ensure correct hyperparams are being fetched regarding weights
    assert (
        await async_subtensor.subnets.min_allowed_weights(alice_sn.netuid) is not None
    )
    assert await async_subtensor.subnets.max_weight_limit(alice_sn.netuid) is not None
    assert await async_subtensor.subnets.weights_rate_limit(alice_sn.netuid) is not None

    # Wait until next epoch so we can set root weights
    next_epoch_start_block = await async_subtensor.subnets.get_next_epoch_start_block(
        alice_sn.netuid
    )
    await async_subtensor.wait_for_block(next_epoch_start_block + 1)

    # Change hyperparams so we can execute pow_register
    await alice_sn.async_execute_steps(
        [
            SUDO_SET_DIFFICULTY(alice_wallet, AdminUtils, True, NETUID, 1_000_000),
            SUDO_SET_NETWORK_POW_REGISTRATION_ALLOWED(
                alice_wallet, AdminUtils, True, NETUID, True
            ),
        ]
    )

    # Fetch neuron lite for sn one and assert Alice participation
    sn_one_neurons = await async_subtensor.neurons.neurons_lite(netuid=alice_sn.netuid)
    assert (
        sn_one_neurons[alice_uid_sn_2].coldkey == alice_wallet.coldkeypub.ss58_address
    )
    assert sn_one_neurons[alice_uid_sn_2].hotkey == alice_wallet.hotkey.ss58_address
    assert sn_one_neurons[alice_uid_sn_2].validator_permit is True
