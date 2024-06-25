import asyncio
import logging
import sys

import pytest

import bittensor
from bittensor.commands import (
    RegisterCommand,
    RegisterSubnetworkCommand,
    StakeCommand,
    RootRegisterCommand,
    RootSetBoostCommand,
    SubnetSudoCommand,
    RootSetWeightsCommand,
    SetTakeCommand,
)
from tests.e2e_tests.utils import (
    setup_wallet,
    template_path,
    templates_repo,
    wait_interval,
)

logging.basicConfig(level=logging.INFO)

"""
Test the emissions mechanism. 

Verify that for the miner:
* trust
* rank
* consensus
* incentive
* emission
are updated with proper values after an epoch has passed. 

For the validator verify that:
* validator_permit
* validator_trust
* dividends
* stake
are updated with proper values after an epoch has passed. 

"""


@pytest.mark.asyncio
async def test_emissions(local_chain):
    # Register root as Alice - the subnet owner and validator
    alice_keypair, alice_exec_command, alice_wallet = setup_wallet("//Alice")
    alice_exec_command(RegisterSubnetworkCommand, ["s", "create"])
    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    # Register Bob as miner
    bob_keypair, bob_exec_command, bob_wallet = setup_wallet("//Bob")

    # Register Alice as neuron to the subnet
    alice_exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
        ],
    )

    # Register Bob as neuron to the subnet
    bob_exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
        ],
    )

    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    # assert two neurons are in network
    assert len(subtensor.neurons(netuid=1)) == 2

    # Alice to stake to become to top neuron after the first epoch
    alice_exec_command(
        StakeCommand,
        [
            "stake",
            "add",
            "--amount",
            "10000",
        ],
    )

    # register Alice as validator
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/neurons/validator.py"',
            "--no_prompt",
            "--netuid",
            "1",
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
    # run validator in the background

    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    await asyncio.sleep(10)

    # register validator with root network
    alice_exec_command(
        RootRegisterCommand,
        [
            "root",
            "register",
            "--netuid",
            "1",
        ],
    )

    wait_interval(600, subtensor)

    alice_exec_command(
        RootSetBoostCommand,
        [
            "root",
            "boost",
            "--netuid",
            "1",
            "--increase",
            "1000",
        ],
    )

    # register Bob as miner
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/neurons/miner.py"',
            "--no_prompt",
            "--netuid",
            "1",
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

    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    wait_interval(600, subtensor)

    logging.warning("Setting root set weights")
    alice_exec_command(
        RootSetWeightsCommand,
        [
            "root",
            "weights",
            "--netuid",
            "1",
            "--weights",
            "0.3",
            "--wallet.name",
            "default",
            "--wallet.hotkey",
            "default",
        ],
    )

    # Set delegate take for Alice
    alice_exec_command(SetTakeCommand, ["r", "set_take", "--take", "0.15"])

    # Lower the rate limit
    alice_exec_command(
        SubnetSudoCommand,
        [
            "sudo",
            "set",
            "hyperparameters",
            "--netuid",
            "1",
            "--wallet.name",
            alice_wallet.name,
            "--param",
            "weights_rate_limit",
            "--value",
            "1",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    # wait epoch until for emissions to get distributed
    wait_interval(600, subtensor)

    await asyncio.sleep(
        10
    )  # wait for 5 seconds for the metagraph and subtensor to refresh with latest data

    # refresh metagraph
    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    # get current emissions and validate that Alice has gotten tao
    weights = [(0, [(0, 65535), (1, 65535)])]
    assert subtensor.weights(netuid=1) == weights

    neurons = subtensor.neurons(netuid=1)
    bob = neurons[1]
    alice = neurons[0]

    assert bob.emission > 0
    assert bob.consensus == 1
    assert bob.incentive == 1
    assert bob.rank == 1
    assert bob.trust == 1

    assert alice.emission > 0
    assert alice.bonds == [(1, 65535)]
    assert alice.dividends == 1
    assert alice.stake.tao > 10000  # assert an increase in stake
    assert alice.validator_permit is True
    assert alice.validator_trust == 1

    assert alice.weights == [(0, 65535), (1, 65535)]

    assert (
        subtensor.get_emission_value_by_subnet(netuid=1) > 0
    )  # emission on this subnet is strictly greater than 0
