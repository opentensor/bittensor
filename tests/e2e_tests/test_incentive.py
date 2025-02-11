import asyncio
import sys
import time

import pytest

from tests.e2e_tests.utils.chain_interactions import (
    sudo_set_hyperparameter_values,
    wait_epoch,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    template_path,
    templates_repo,
)


@pytest.mark.asyncio
async def test_incentive(local_chain, subtensor, alice_wallet, bob_wallet):
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

    # Prepare to run Bob as miner
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/miner.py"',
            "--netuid",
            str(netuid),
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--wallet.path",
            bob_wallet.path,
            "--wallet.name",
            bob_wallet.name,
            "--wallet.hotkey",
            "default",
            "--logging.trace",
        ]
    )

    # Run Bob as miner in the background
    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    print("Neuron Bob is now mining")

    # Prepare to run Alice as validator
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/validator.py"',
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

    # wait for the Validator to process and set_weights
    await asyncio.sleep(5)

    # Wait until next epoch
    await wait_epoch(subtensor, netuid)

    # Refresh metagraph
    metagraph = subtensor.metagraph(netuid)

    # Get current emissions and validate that Alice has gotten tao
    alice_neuron = metagraph.neurons[0]
    time.sleep(5)

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
