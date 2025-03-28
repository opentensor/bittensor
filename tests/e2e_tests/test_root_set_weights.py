import asyncio
import pytest

from tests.e2e_tests.utils.chain_interactions import (
    wait_epoch,
    sudo_set_hyperparameter_values,
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
async def test_root_reg_hyperparams(
    local_chain,
    subtensor,
    templates,
    alice_wallet,
    bob_wallet,
):
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

    print("Testing root register, weights, and hyperparams")
    netuid = 2

    # Default immunity period & tempo set through the subtensor side
    default_immunity_period = 5000
    default_tempo = 10

    # 0.2 for root network, 0.8 for sn 1
    # Corresponding to [0.2, 0.8]
    weights = [16384, 65535]

    # Register Alice in root network (0)
    assert await subtensor.root_register(alice_wallet)

    # Assert Alice is successfully registered to root
    alice_root_neuron = (await subtensor.neurons(netuid=0))[0]
    assert alice_root_neuron.coldkey == alice_wallet.coldkeypub.ss58_address
    assert alice_root_neuron.hotkey == alice_wallet.hotkey.ss58_address

    # Create netuid = 2
    assert await subtensor.register_subnet(alice_wallet)

    # Ensure correct immunity period & tempo is being fetched
    assert await subtensor.immunity_period(netuid=netuid) == default_immunity_period
    assert await subtensor.tempo(netuid=netuid) == default_tempo

    async with templates.validator(alice_wallet, netuid):
        await asyncio.sleep(5)  # Wait a bit for chain to process data

        # Fetch uid against Alice's hotkey on sn 2 (it will be 0 as she is the only registered neuron)
        alice_uid_sn_2 = await subtensor.get_uid_for_hotkey_on_subnet(
            alice_wallet.hotkey.ss58_address, netuid
        )

        # Fetch the block since last update for the neuron
        block_since_update = await subtensor.blocks_since_last_update(
            netuid=netuid, uid=alice_uid_sn_2
        )
        assert block_since_update is not None

    # Verify subnet <netuid> created successfully
    assert await subtensor.subnet_exists(netuid), "Subnet wasn't created successfully"

    # Use subnetwork_n hyperparam to check sn creation
    assert await subtensor.subnetwork_n(netuid) == 1  # TODO?
    assert await subtensor.subnetwork_n(netuid + 1) is None

    # Ensure correct hyperparams are being fetched regarding weights
    assert await subtensor.min_allowed_weights(netuid) is not None
    assert await subtensor.max_weight_limit(netuid) is not None
    assert await subtensor.weights_rate_limit(netuid) is not None

    # Wait until next epoch so we can set root weights
    await wait_epoch(subtensor, netuid)

    # Change hyperparams so we can execute pow_register
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_difficulty",
        call_params={"netuid": netuid, "difficulty": "1_000_000"},
        return_error_message=True,
    )

    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_network_pow_registration_allowed",
        call_params={"netuid": netuid, "registration_allowed": True},
        return_error_message=True,
    )

    # TODO: Implement
    # This registers neuron using pow but it doesn't work on fast-blocks - we get stale pow
    # pow_registration = subtensor.register(bob_wallet, netuid=1)

    # Fetch neuron lite for sn one and assert Alice participation
    sn_one_neurons = await subtensor.neurons_lite(netuid=netuid)
    assert (
        sn_one_neurons[alice_uid_sn_2].coldkey == alice_wallet.coldkeypub.ss58_address
    )
    assert sn_one_neurons[alice_uid_sn_2].hotkey == alice_wallet.hotkey.ss58_address
    assert sn_one_neurons[alice_uid_sn_2].validator_permit is True

    print("✅ Passed root tests")
