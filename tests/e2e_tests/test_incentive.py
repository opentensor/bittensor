import asyncio

import pytest

from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.chain_interactions import (
    root_set_subtensor_hyperparameter_values,
    sudo_set_admin_utils,
    wait_epoch,
    wait_interval,
)

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
    netuid = 2

    # Register root as Alice - the subnet owner and validator
    assert subtensor.register_subnet(alice_wallet)

    # Verify subnet <netuid> created successfully
    assert subtensor.subnet_exists(netuid), "Subnet wasn't created successfully"

    # Register Bob as a neuron on the subnet
    assert subtensor.burned_register(
        bob_wallet, netuid
    ), "Unable to register Bob as a neuron"

    # Assert two neurons are in network
    assert (
        len(subtensor.neurons(netuid=netuid)) == 2
    ), "Alice & Bob not registered in the subnet"

    # Wait for the first epoch to pass
    await wait_epoch(subtensor, netuid)

    # Get latest metagraph
    metagraph = subtensor.metagraph(netuid)

    # Get current miner/validator stats
    alice_neuron = metagraph.neurons[0]

    assert alice_neuron.validator_permit is True
    assert alice_neuron.dividends == 0
    assert alice_neuron.stake.tao == 0
    assert alice_neuron.validator_trust == 0
    assert alice_neuron.incentive == 0
    assert alice_neuron.consensus == 0
    assert alice_neuron.rank == 0

    bob_neuron = metagraph.neurons[1]

    assert bob_neuron.incentive == 0
    assert bob_neuron.consensus == 0
    assert bob_neuron.rank == 0
    assert bob_neuron.trust == 0

    subtensor.wait_for_block(DURATION_OF_START_CALL)

    # Subnet "Start Call" https://github.com/opentensor/bits/pull/13
    status, error = await root_set_subtensor_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="start_call",
        call_params={
            "netuid": netuid,
        },
    )

    assert status is True, error

    # update weights_set_rate_limit for fast-blocks
    tempo = subtensor.tempo(netuid)
    status, error = sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_function="sudo_set_weights_set_rate_limit",
        call_params={
            "netuid": netuid,
            "weights_set_rate_limit": tempo,
        },
    )

    assert error is None
    assert status is True

    async with templates.miner(bob_wallet, netuid):
        async with templates.validator(alice_wallet, netuid) as validator:
            # wait for the Validator to process and set_weights
            await asyncio.wait_for(validator.set_weights.wait(), 120)

            # Wait till new epoch
            await wait_interval(tempo, subtensor, netuid)

    # Sometimes the network does not have time to release data, and it requires several additional blocks (subtensor issue)
    # Call get_metagraph_info since if faster and chipper
    while subtensor.get_metagraph_info(netuid).incentives[0] == 0:
        logging.console.info(
            f"Additional fast block to wait chain data updated: {subtensor.block}"
        )
        await asyncio.sleep(0.25)

    # Refresh metagraph
    metagraph = subtensor.metagraph(netuid)

    # Get current emissions and validate that Alice has gotten tao
    alice_neuron = metagraph.neurons[0]

    assert alice_neuron.validator_permit is True
    assert alice_neuron.dividends == 1.0
    assert alice_neuron.stake.tao > 0
    assert alice_neuron.validator_trust > 0.99
    assert alice_neuron.incentive < 0.5
    assert alice_neuron.consensus < 0.5
    assert alice_neuron.rank < 0.5

    bob_neuron = metagraph.neurons[1]

    assert bob_neuron.incentive > 0.5
    assert bob_neuron.consensus > 0.5
    assert bob_neuron.rank > 0.5
    assert bob_neuron.trust == 1

    bonds = subtensor.bonds(netuid)

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
