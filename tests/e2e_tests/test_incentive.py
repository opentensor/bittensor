import asyncio

import pytest

from bittensor import Balance

from tests.e2e_tests.utils.chain_interactions import (
    sudo_set_hyperparameter_values,
    wait_epoch,
    sudo_set_admin_utils,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("local_chain", [False], indirect=True)
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

    # Change tempo to 10
    tempo_set = 10
    assert (
        sudo_set_admin_utils(
            local_chain,
            alice_wallet,
            call_function="sudo_set_tempo",
            call_params={"netuid": netuid, "tempo": tempo_set},
            return_error_message=True,
        )[0]
        is True
    )
    tempo = subtensor.get_subnet_hyperparameters(netuid=netuid).tempo
    assert tempo_set == tempo

    # Register Bob as a neuron on the subnet
    assert subtensor.burned_register(
        bob_wallet, netuid
    ), "Unable to register Bob as a neuron"

    # Assert two neurons are in network
    assert (
        len(subtensor.neurons(netuid=netuid)) == 2
    ), "Alice & Bob not registered in the subnet"

    # Add stake for Alice
    assert subtensor.add_stake(
        alice_wallet,
        netuid=netuid,
        amount=Balance.from_tao(1_000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    ), "Failed to add stake for Alice"

    # Wait for the first epoch to pass
    await wait_epoch(subtensor, netuid)

    # Add further stake so validator permit is activated
    assert subtensor.add_stake(
        alice_wallet,
        netuid=netuid,
        amount=Balance.from_tao(1_000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    ), "Failed to add stake for Alice"

    # Get latest metagraph
    metagraph = subtensor.metagraph(netuid)

    # Get current miner/validator stats
    alice_neuron = metagraph.neurons[0]

    assert alice_neuron.validator_permit is True
    assert alice_neuron.dividends == 0
    assert alice_neuron.stake.tao > 0
    assert alice_neuron.validator_trust == 0

    bob_neuron = metagraph.neurons[1]

    assert bob_neuron.incentive == 0
    assert bob_neuron.consensus == 0
    assert bob_neuron.rank == 0
    assert bob_neuron.trust == 0

    # update weights_set_rate_limit for fast-blocks
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_weights_set_rate_limit",
        call_params={"netuid": netuid, "weights_set_rate_limit": 10},
        return_error_message=True,
    )

    async with templates.miner(bob_wallet, netuid):
        async with templates.validator(alice_wallet, netuid):
            # Wait for the Validator to process and set_weights
            await asyncio.sleep(5)

            # Wait few epochs
            await wait_epoch(subtensor, netuid, times=2)

            # Refresh metagraph
            metagraph = subtensor.metagraph(netuid)

    # Get current emissions and validate that Alice has gotten tao
    alice_neuron = metagraph.neurons[0]

    assert alice_neuron.validator_permit is True
    assert alice_neuron.dividends == 1.0
    assert alice_neuron.stake.tao > 0
    assert alice_neuron.validator_trust == 1

    bob_neuron = metagraph.neurons[1]
    assert bob_neuron.incentive == 1
    assert bob_neuron.consensus == 1
    assert bob_neuron.rank == 1
    assert bob_neuron.trust == 1

    print("✅ Passed test_incentive")
