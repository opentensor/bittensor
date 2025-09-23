import asyncio

import pytest

from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.chain_interactions import (
    async_sudo_set_admin_utils,
    async_wait_epoch,
    sudo_set_admin_utils,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    async_wait_to_start_call,
    wait_to_start_call,
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

    logging.console.info("Testing [blue]test_incentive[/blue]")
    alice_subnet_netuid = subtensor.subnets.get_total_subnets()  # 2
    # turn off admin freeze window limit for testing
    assert (
        sudo_set_admin_utils(
            substrate=subtensor.substrate,
            wallet=alice_wallet,
            call_function="sudo_set_admin_freeze_window",
            call_params={"window": 0},
        )[0]
        is True
    ), "Failed to set admin freeze window to 0"

    # Register root as Alice - the subnet owner and validator
    assert subtensor.subnets.register_subnet(alice_wallet), "Subnet wasn't created"

    # Verify subnet <netuid> created successfully
    assert subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    # Disable commit_reveal on the subnet to check proper behavior
    status, error = sudo_set_admin_utils(
        substrate=subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_commit_reveal_weights_enabled",
        call_params={
            "netuid": alice_subnet_netuid,
            "enabled": False,
        },
    )
    assert status is True, error

    assert wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid)

    # Register Bob as a neuron on the subnet
    assert subtensor.subnets.burned_register(bob_wallet, alice_subnet_netuid).success, (
        "Unable to register Bob as a neuron"
    )

    # Assert two neurons are in network
    assert len(subtensor.neurons.neurons(netuid=alice_subnet_netuid)) == 2, (
        "Alice & Bob not registered in the subnet"
    )

    # Wait for the first epoch to pass
    subtensor.wait_for_block(
        subtensor.subnets.get_next_epoch_start_block(alice_subnet_netuid) + 1
    )

    # Get current miner/validator stats
    alice_neuron = subtensor.neurons.neurons(netuid=alice_subnet_netuid)[0]

    assert alice_neuron.validator_permit is True
    assert alice_neuron.dividends == 0
    assert alice_neuron.validator_trust == 0
    assert alice_neuron.incentive == 0
    assert alice_neuron.consensus == 0
    assert alice_neuron.rank == 0

    bob_neuron = subtensor.neurons.neurons(netuid=alice_subnet_netuid)[1]

    assert bob_neuron.incentive == 0
    assert bob_neuron.consensus == 0
    assert bob_neuron.rank == 0
    assert bob_neuron.trust == 0

    # update weights_set_rate_limit for fast-blocks
    tempo = subtensor.subnets.tempo(alice_subnet_netuid)
    status, error = sudo_set_admin_utils(
        substrate=subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_weights_set_rate_limit",
        call_params={
            "netuid": alice_subnet_netuid,
            "weights_set_rate_limit": tempo,
        },
    )

    assert error is None
    assert status is True

    # max attempts to run miner and validator
    max_attempt = 3
    while True:
        try:
            async with templates.miner(bob_wallet, alice_subnet_netuid) as miner:
                await asyncio.wait_for(miner.started.wait(), 60)

                async with templates.validator(
                    alice_wallet, alice_subnet_netuid
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
        alice_subnet_netuid
    )
    subtensor.wait_for_block(next_epoch_start_block + tempo + 1)

    validators = subtensor.metagraphs.get_metagraph_info(
        alice_subnet_netuid, field_indices=[72]
    ).validators

    alice_uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=alice_wallet.hotkey.ss58_address, netuid=alice_subnet_netuid
    )
    assert validators[alice_uid] == 1

    bob_uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=bob_wallet.hotkey.ss58_address, netuid=alice_subnet_netuid
    )
    assert validators[bob_uid] == 0

    while True:
        try:
            neurons = subtensor.neurons.neurons(netuid=alice_subnet_netuid)
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

            bonds = subtensor.subnets.bonds(alice_subnet_netuid)

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

            print("✅ Passed test_incentive")
            break
        except Exception:
            subtensor.wait_for_block(subtensor.block)
            continue

    logging.console.success("Test [green]test_incentive[/green] passed.")


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

    logging.console.info("Testing [blue]test_incentive[/blue]")
    alice_subnet_netuid = await async_subtensor.subnets.get_total_subnets()  # 2

    # turn off admin freeze window limit for testing
    assert (
        await async_sudo_set_admin_utils(
            substrate=async_subtensor.substrate,
            wallet=alice_wallet,
            call_function="sudo_set_admin_freeze_window",
            call_params={"window": 0},
        )
    )[0] is True, "Failed to set admin freeze window to 0"

    # Register root as Alice - the subnet owner and validator
    assert await async_subtensor.subnets.register_subnet(alice_wallet), (
        "Subnet wasn't created"
    )

    # Verify subnet <netuid> created successfully
    assert await async_subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    # Disable commit_reveal on the subnet to check proper behavior
    status, error = await async_sudo_set_admin_utils(
        substrate=async_subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_commit_reveal_weights_enabled",
        call_params={
            "netuid": alice_subnet_netuid,
            "enabled": False,
        },
    )
    assert status is True, error

    assert await async_wait_to_start_call(
        async_subtensor, alice_wallet, alice_subnet_netuid
    )

    # Register Bob as a neuron on the subnet
    assert (
        await async_subtensor.subnets.burned_register(bob_wallet, alice_subnet_netuid)
    ).success, "Unable to register Bob as a neuron"

    # Assert two neurons are in network
    assert (
        len(await async_subtensor.neurons.neurons(netuid=alice_subnet_netuid)) == 2
    ), "Alice & Bob not registered in the subnet"

    # Wait for the first epoch to pass
    next_epoch_start_block = await async_subtensor.subnets.get_next_epoch_start_block(
        netuid=alice_subnet_netuid
    )
    await async_subtensor.wait_for_block(next_epoch_start_block + 1)

    # Get current miner/validator stats
    alice_neuron = (await async_subtensor.neurons.neurons(netuid=alice_subnet_netuid))[
        0
    ]

    assert alice_neuron.validator_permit is True
    assert alice_neuron.dividends == 0
    assert alice_neuron.validator_trust == 0
    assert alice_neuron.incentive == 0
    assert alice_neuron.consensus == 0
    assert alice_neuron.rank == 0

    bob_neuron = (await async_subtensor.neurons.neurons(netuid=alice_subnet_netuid))[1]

    assert bob_neuron.incentive == 0
    assert bob_neuron.consensus == 0
    assert bob_neuron.rank == 0
    assert bob_neuron.trust == 0

    # update weights_set_rate_limit for fast-blocks
    tempo = await async_subtensor.subnets.tempo(alice_subnet_netuid)
    status, error = await async_sudo_set_admin_utils(
        substrate=async_subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_weights_set_rate_limit",
        call_params={
            "netuid": alice_subnet_netuid,
            "weights_set_rate_limit": tempo,
        },
    )

    assert error is None
    assert status is True

    # max attempts to run miner and validator
    max_attempt = 3
    while True:
        try:
            async with templates.miner(bob_wallet, alice_subnet_netuid) as miner:
                await asyncio.wait_for(miner.started.wait(), 60)

                async with templates.validator(
                    alice_wallet, alice_subnet_netuid
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
        alice_subnet_netuid
    )
    await async_subtensor.wait_for_block(next_epoch_start_block + tempo + 1)

    validators = (
        await async_subtensor.metagraphs.get_metagraph_info(
            alice_subnet_netuid, field_indices=[72]
        )
    ).validators

    alice_uid = await async_subtensor.subnets.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=alice_wallet.hotkey.ss58_address, netuid=alice_subnet_netuid
    )
    assert validators[alice_uid] == 1

    bob_uid = await async_subtensor.subnets.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=bob_wallet.hotkey.ss58_address, netuid=alice_subnet_netuid
    )
    assert validators[bob_uid] == 0

    while True:
        try:
            neurons = await async_subtensor.neurons.neurons(netuid=alice_subnet_netuid)
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

            bonds = await async_subtensor.subnets.bonds(alice_subnet_netuid)

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

            print("✅ Passed test_incentive")
            break
        except Exception:
            await async_subtensor.wait_for_block(await async_subtensor.block)
            continue

    logging.console.success("Test [green]test_incentive[/green] passed.")
