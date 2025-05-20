import asyncio

import pytest

from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.chain_interactions import (
    sudo_set_admin_utils,
    wait_epoch,
)
from tests.e2e_tests.utils.e2e_test_utils import wait_to_start_call

DURATION_OF_START_CALL = 10


@pytest.mark.asyncio
async def test_incentive(local_chain, subtensor, templates, alice_wallet, bob_wallet):
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

    print("Testing test_incentive")
    alice_subnet_netuid = subtensor.get_total_subnets()  # 2

    # Register root as Alice - the subnet owner and validator
    assert subtensor.register_subnet(alice_wallet, True, True), "Subnet wasn't created"

    # Verify subnet <netuid> created successfully
    assert subtensor.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid)

    # Register Bob as a neuron on the subnet
    assert subtensor.burned_register(bob_wallet, alice_subnet_netuid), (
        "Unable to register Bob as a neuron"
    )

    # Assert two neurons are in network
    assert len(subtensor.neurons(netuid=alice_subnet_netuid)) == 2, (
        "Alice & Bob not registered in the subnet"
    )

    # Wait for the first epoch to pass
    await wait_epoch(subtensor, alice_subnet_netuid)

    # Get current miner/validator stats
    alice_neuron = subtensor.neurons(netuid=alice_subnet_netuid)[0]

    assert alice_neuron.validator_permit is True
    assert alice_neuron.dividends == 0
    assert alice_neuron.validator_trust == 0
    assert alice_neuron.incentive == 0
    assert alice_neuron.consensus == 0
    assert alice_neuron.rank == 0

    bob_neuron = subtensor.neurons(netuid=alice_subnet_netuid)[1]

    assert bob_neuron.incentive == 0
    assert bob_neuron.consensus == 0
    assert bob_neuron.rank == 0
    assert bob_neuron.trust == 0

    # update weights_set_rate_limit for fast-blocks
    tempo = subtensor.tempo(alice_subnet_netuid)
    status, error = sudo_set_admin_utils(
        local_chain,
        alice_wallet,
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

    # wait one tempo (fast block
    subtensor.wait_for_block(subtensor.block + subtensor.tempo(alice_subnet_netuid))

    alice_uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=alice_wallet.hotkey.ss58_address, netuid=alice_subnet_netuid
    )
    assert (
        subtensor.metagraphs.get_metagraph_info(
            alice_subnet_netuid, field_indices=[72]
        ).validators[alice_uid]
        == 1
    )

    bob_uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=bob_wallet.hotkey.ss58_address, netuid=alice_subnet_netuid
    )
    assert (
        subtensor.metagraphs.get_metagraph_info(
            alice_subnet_netuid, field_indices=[72]
        ).validators[bob_uid]
        == 0
    )

    while True:
        try:
            neurons = subtensor.neurons(netuid=alice_subnet_netuid)
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

            bonds = subtensor.bonds(alice_subnet_netuid)

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
