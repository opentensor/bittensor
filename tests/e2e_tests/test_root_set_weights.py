import asyncio
import sys
import numpy as np
import pytest

from bittensor.core.subtensor import Subtensor
from tests.e2e_tests.utils.chain_interactions import (
    register_subnet,
    wait_epoch,
    sudo_set_hyperparameter_values,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    setup_wallet,
    template_path,
    templates_repo,
)

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
async def test_root_reg_hyperparams(local_chain):
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
    netuid = 1

    # Default immunity period & tempo set through the subtensor side
    default_immunity_period = 5000
    default_tempo = 360

    # 0.2 for root network, 0.8 for sn 1
    weights = [0.2, 0.8]

    # Create Alice, SN1 owner and root network member
    alice_keypair, alice_wallet = setup_wallet("//Alice")
    subtensor = Subtensor(network="ws://localhost:9945")

    # Register Alice in root network (0)
    assert subtensor.root_register(alice_wallet)

    # Assert Alice is successfully registered to root
    alice_root_neuron = subtensor.neurons(netuid=0)[0]
    assert alice_root_neuron.coldkey == alice_wallet.coldkeypub.ss58_address
    assert alice_root_neuron.hotkey == alice_wallet.hotkey.ss58_address

    # Create netuid = 1
    register_subnet(local_chain, alice_wallet)

    # Register Alice as a neuron on the subnet
    assert subtensor.burned_register(
        alice_wallet, netuid
    ), "Unable to register Alice as a neuron"

    # Ensure correct immunity period & tempo is being fetched
    assert subtensor.immunity_period(netuid=netuid) == default_immunity_period
    assert subtensor.tempo(netuid=netuid) == default_tempo

    # Prepare to run Alice as validator
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/neurons/validator.py"',
            "--netuid",
            str(netuid),
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--wallet.path",
            alice_wallet.path,
            "--wallet.name",
            alice_wallet.name,
            "--wallet.hotkey",
            "default",
            "--logging.trace",
        ]
    )

    # Run Alice as validator in the background
    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    print("Neuron Alice is now validating")
    await asyncio.sleep(5)  # Wait a bit for chain to process data

    # Fetch uid against Alice's hotkey on sn 1 (it will be 0 as she is the only registered neuron)
    alice_uid_sn_1 = subtensor.get_uid_for_hotkey_on_subnet(
        alice_wallet.hotkey.ss58_address, netuid
    )

    # Fetch the block since last update, so we can compare later
    initial_block = subtensor.blocks_since_last_update(
        netuid=netuid, uid=alice_uid_sn_1
    )

    # Verify subnet <netuid> created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [netuid]
    ).serialize(), "Subnet wasn't created successfully"

    # Use subnetwork_n hyperparam to check sn creation
    assert subtensor.subnetwork_n(netuid) == netuid
    assert subtensor.subnetwork_n(2) is None

    # Ensure correct hyperparams are being fetched regarding weights
    assert subtensor.min_allowed_weights(netuid=1) is not None
    assert subtensor.max_weight_limit(netuid=1) is not None
    assert subtensor.weights_rate_limit(netuid) is not None

    # Wait until next epoch so we can set root weights
    await wait_epoch(subtensor)

    # Set root weights for netuids 0, 1
    assert subtensor.root_set_weights(
        alice_wallet,
        [0, 1],
        weights,
        wait_for_inclusion=False,
        wait_for_finalization=True,
    )

    # Query the weights from the chain
    weights_raw = local_chain.query("SubtensorModule", "Weights", [0, 0]).serialize()

    weights_array = np.array(weights_raw)
    normalized_weights = weights_array[:, 1] / max(np.sum(weights_array, axis=0)[1], 1)
    rounded_weights = [round(weight, 1) for weight in normalized_weights]

    # Assert correct weights were set for root and sn 1
    assert weights == rounded_weights

    # Register Bob as miner
    bob_keypair, bob_wallet = setup_wallet("//Bob")

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

    updated_block = subtensor.blocks_since_last_update(netuid=netuid, uid=0)
    # Ensure updates are reflected through incremental block numbers
    assert updated_block > initial_block

    # TODO: Implement
    # This registers neuron using pow but it doesn't work on fast-blocks - we get stale pow
    # pow_registration = subtensor.register(bob_wallet, netuid=1)

    # Fetch neuron lite for sn one and assert Alice participation
    sn_one_neurons = subtensor.neurons_lite(netuid=netuid)
    assert (
        sn_one_neurons[alice_uid_sn_1].coldkey == alice_wallet.coldkeypub.ss58_address
    )
    assert sn_one_neurons[alice_uid_sn_1].hotkey == alice_wallet.hotkey.ss58_address
    assert sn_one_neurons[alice_uid_sn_1].validator_permit is True

    print("✅ Passed root tests")
