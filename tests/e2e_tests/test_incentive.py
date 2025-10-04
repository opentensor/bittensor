import asyncio

import pytest

from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils import (
    TestSubnet,
    AdminUtils,
    NETUID,
    ACTIVATE_SUBNET,
    REGISTER_NEURON,
    REGISTER_SUBNET,
    SUDO_SET_ADMIN_FREEZE_WINDOW,
    SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED,
    SUDO_SET_WEIGHTS_SET_RATE_LIMIT,
)


@pytest.mark.asyncio
async def test_incentive(subtensor, templates, alice_wallet, bob_wallet):
    """
    Test the incentive mechanism and interaction of miners/validators

    Steps:
        1. Register a subnet as Alice and register Bob
        2. Run Alice as validator & Bob as miner. Wait Epoch
        3. Verify miner has correct: trust, rank, consensus, incentive
        4. Verify validator has correct: validator_permit, validator_trust, dividends, stake
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    alice_sn = TestSubnet(subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(alice_wallet),
        SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
            alice_wallet, AdminUtils, True, NETUID, False
        ),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    alice_sn.execute_steps(steps)

    # Assert two neurons are in network
    assert len(subtensor.neurons.neurons(netuid=alice_sn.netuid)) == 2, (
        "Alice & Bob not registered in the subnet"
    )

    # Wait for the first epoch to pass
    subtensor.wait_for_block(
        subtensor.subnets.get_next_epoch_start_block(alice_sn.netuid) + 5
    )

    # Get current miner/validator stats
    alice_neuron = subtensor.neurons.neurons(netuid=alice_sn.netuid)[0]

    assert alice_neuron.validator_permit is True
    assert alice_neuron.dividends == 0
    assert alice_neuron.validator_trust == 0
    assert alice_neuron.incentive == 0
    assert alice_neuron.consensus == 0
    assert alice_neuron.rank == 0

    bob_neuron = subtensor.neurons.neurons(netuid=alice_sn.netuid)[1]

    assert bob_neuron.incentive == 0
    assert bob_neuron.consensus == 0
    assert bob_neuron.rank == 0
    assert bob_neuron.trust == 0

    # update weights_set_rate_limit for fast-blocks
    tempo = subtensor.subnets.tempo(alice_sn.netuid)
    alice_sn.execute_one(
        SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, tempo)
    )

    # max attempts to run miner and validator
    max_attempt = 3
    while True:
        try:
            async with templates.miner(bob_wallet, alice_sn.netuid) as miner:
                await asyncio.wait_for(miner.started.wait(), 60)

                async with templates.validator(
                    alice_wallet, alice_sn.netuid
                ) as validator:
                    # wait for the Validator to process and set_weights
                    await asyncio.wait_for(validator.set_weights.wait(), 60)
            break
        except asyncio.TimeoutError:
            if max_attempt > 0:
                max_attempt -= 1
                continue
            raise

    # wait one tempo (fast block)
    next_epoch_start_block = subtensor.subnets.get_next_epoch_start_block(
        alice_sn.netuid
    )
    subtensor.wait_for_block(next_epoch_start_block + tempo + 1)

    validators = subtensor.metagraphs.get_metagraph_info(
        alice_sn.netuid, selected_indices=[72]
    ).validators

    alice_uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=alice_wallet.hotkey.ss58_address, netuid=alice_sn.netuid
    )
    assert validators[alice_uid] == 1

    bob_uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=bob_wallet.hotkey.ss58_address, netuid=alice_sn.netuid
    )
    assert validators[bob_uid] == 0

    while True:
        try:
            neurons = subtensor.neurons.neurons(netuid=alice_sn.netuid)
            logging.info(f"neurons: {neurons}")

            # Get current emissions and validate that Alice has gotten tao
            alice_neuron = neurons[0]

            assert alice_neuron.validator_permit is True
            assert alice_neuron.dividends == 1.0
            assert alice_neuron.stake.tao > 0
            assert alice_neuron.validator_trust > 0.99
            assert alice_neuron.incentive < 0.5
            assert alice_neuron.consensus < 0.5
            assert alice_neuron.rank < 0.5

            bob_neuron = neurons[1]

            assert bob_neuron.incentive > 0.5
            assert bob_neuron.consensus > 0.5
            assert bob_neuron.rank > 0.5
            assert bob_neuron.trust == 1

            bonds = subtensor.subnets.bonds(alice_sn.netuid)

            assert bonds == [
                (
                    0,
                    [
                        (0, 65535),
                        (1, 65535),
                    ],
                ),
                (
                    1,
                    [],
                ),
            ]

            break
        except Exception:
            subtensor.wait_for_block(subtensor.block)
            continue


@pytest.mark.asyncio
async def test_incentive_async(async_subtensor, templates, alice_wallet, bob_wallet):
    """
    Test the incentive mechanism and interaction of miners/validators

    Steps:
        1. Register a subnet as Alice and register Bob
        2. Run Alice as validator & Bob as miner. Wait Epoch
        3. Verify miner has correct: trust, rank, consensus, incentive
        4. Verify validator has correct: validator_permit, validator_trust, dividends, stake
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(alice_wallet),
        SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
            alice_wallet, AdminUtils, True, NETUID, False
        ),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    # Assert two neurons are in network
    assert len(await async_subtensor.neurons.neurons(netuid=alice_sn.netuid)) == 2, (
        "Alice & Bob not registered in the subnet"
    )

    # Wait for the first epoch to pass
    next_epoch_start_block = await async_subtensor.subnets.get_next_epoch_start_block(
        netuid=alice_sn.netuid
    )
    await async_subtensor.wait_for_block(next_epoch_start_block + 5)

    # Get current miner/validator stats
    alice_neuron = (await async_subtensor.neurons.neurons(netuid=alice_sn.netuid))[0]

    assert alice_neuron.validator_permit is True
    assert alice_neuron.dividends == 0
    assert alice_neuron.validator_trust == 0
    assert alice_neuron.incentive == 0
    assert alice_neuron.consensus == 0
    assert alice_neuron.rank == 0

    bob_neuron = (await async_subtensor.neurons.neurons(netuid=alice_sn.netuid))[1]

    assert bob_neuron.incentive == 0
    assert bob_neuron.consensus == 0
    assert bob_neuron.rank == 0
    assert bob_neuron.trust == 0

    # update weights_set_rate_limit for fast-blocks
    tempo = await async_subtensor.subnets.tempo(alice_sn.netuid)
    await alice_sn.async_execute_one(
        SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, tempo)
    )

    # max attempts to run miner and validator
    max_attempt = 3
    while True:
        try:
            async with templates.miner(bob_wallet, alice_sn.netuid) as miner:
                await asyncio.wait_for(miner.started.wait(), 60)

                async with templates.validator(
                    alice_wallet, alice_sn.netuid
                ) as validator:
                    # wait for the Validator to process and set_weights
                    await asyncio.wait_for(validator.set_weights.wait(), 60)
            break
        except asyncio.TimeoutError:
            if max_attempt > 0:
                max_attempt -= 1
                continue
            raise

    # wait one tempo (fast block)
    next_epoch_start_block = await async_subtensor.subnets.get_next_epoch_start_block(
        alice_sn.netuid
    )
    await async_subtensor.wait_for_block(next_epoch_start_block + tempo + 1)

    validators = (
        await async_subtensor.metagraphs.get_metagraph_info(
            alice_sn.netuid, selected_indices=[72]
        )
    ).validators

    alice_uid = await async_subtensor.subnets.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=alice_wallet.hotkey.ss58_address, netuid=alice_sn.netuid
    )
    assert validators[alice_uid] == 1

    bob_uid = await async_subtensor.subnets.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=bob_wallet.hotkey.ss58_address, netuid=alice_sn.netuid
    )
    assert validators[bob_uid] == 0

    while True:
        try:
            neurons = await async_subtensor.neurons.neurons(netuid=alice_sn.netuid)
            logging.info(f"neurons: {neurons}")

            # Get current emissions and validate that Alice has gotten tao
            alice_neuron = neurons[0]

            assert alice_neuron.validator_permit is True
            assert alice_neuron.dividends == 1.0
            assert alice_neuron.stake.tao > 0
            assert alice_neuron.validator_trust > 0.99
            assert alice_neuron.incentive < 0.5
            assert alice_neuron.consensus < 0.5
            assert alice_neuron.rank < 0.5

            bob_neuron = neurons[1]

            assert bob_neuron.incentive > 0.5
            assert bob_neuron.consensus > 0.5
            assert bob_neuron.rank > 0.5
            assert bob_neuron.trust == 1

            bonds = await async_subtensor.subnets.bonds(alice_sn.netuid)

            assert bonds == [
                (
                    0,
                    [
                        (0, 65535),
                        (1, 65535),
                    ],
                ),
                (
                    1,
                    [],
                ),
            ]

            break
        except Exception:
            await async_subtensor.wait_for_block(await async_subtensor.block)
            continue
