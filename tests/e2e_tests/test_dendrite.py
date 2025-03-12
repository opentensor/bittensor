import asyncio

import pytest

from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.chain_interactions import (
    sudo_set_admin_utils,
    sudo_set_hyperparameter_values,
    wait_epoch,
)


@pytest.mark.asyncio
async def test_dendrite(local_chain, subtensor, templates, alice_wallet, bob_wallet):
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

    logging.console.info("Testing test_dendrite")
    netuid = 2

    assert sudo_set_admin_utils(
        subtensor, 
        alice_wallet,
        "sudo_set_network_rate_limit",
        call_params={"rate_limit", "0"},
    ), "Unable to set network rate limit"

    # Register a subnet, netuid 2
    assert subtensor.register_subnet(alice_wallet), "Subnet wasn't created"

    # Verify subnet <netuid> created successfully
    assert subtensor.subnet_exists(netuid), "Subnet wasn't created successfully"

    # update max_allowed_validators so only one neuron can get validator_permit
    assert sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_function="sudo_set_max_allowed_validators",
        call_params={
            "netuid": netuid,
            "max_allowed_validators": 1,
        },
        return_error_message=True,
    )

    # update weights_set_rate_limit for fast-blocks
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_weights_set_rate_limit",
        call_params={
            "netuid": netuid,
            "weights_set_rate_limit": 10,
        },
        return_error_message=True,
    )

    # Register Bob to the network
    assert subtensor.burned_register(
        bob_wallet, netuid
    ), "Unable to register Bob as a neuron"

    metagraph = subtensor.metagraph(netuid)

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
    alpha, _ = subtensor.subnet(netuid).tao_to_alpha_with_slippage(tao)

    assert subtensor.add_stake(
        bob_wallet,
        netuid=netuid,
        amount=tao,
    )

    # Refresh metagraph
    metagraph = subtensor.metagraph(netuid)
    bob_neuron = metagraph.neurons[1]

    # Assert alpha is close to stake equivalent
    assert 0.95 < bob_neuron.stake.rao / alpha.rao < 1.05

    # Assert neuron is not a validator yet
    assert bob_neuron.active is True
    assert bob_neuron.validator_permit is False
    assert bob_neuron.validator_trust == 0.0
    assert bob_neuron.pruning_score == 0

    async with templates.validator(bob_wallet, netuid):
        await asyncio.sleep(5)  # wait for 5 seconds for the Validator to process

        await wait_epoch(subtensor, netuid=netuid)

        # Refresh metagraph
        metagraph = subtensor.metagraph(netuid)

    # Refresh validator neuron
    updated_neuron = metagraph.neurons[1]

    assert len(metagraph.neurons) == 2
    assert updated_neuron.active is True
    assert updated_neuron.validator_permit is True
    assert updated_neuron.hotkey == bob_wallet.hotkey.ss58_address
    assert updated_neuron.coldkey == bob_wallet.coldkey.ss58_address
    assert updated_neuron.pruning_score != 0

    logging.console.info("✅ Passed test_dendrite")
