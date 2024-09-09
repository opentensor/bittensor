import asyncio
import sys

import pytest

import bittensor
from bittensor import logging
from bittensor.commands import (
    RegisterCommand,
    RegisterSubnetworkCommand,
    RootRegisterCommand,
    RootSetBoostCommand,
    RootSetWeightsCommand,
    SetTakeCommand,
    StakeCommand,
    SubnetSudoCommand,
)
from tests.e2e_tests.utils import (
    setup_wallet,
    template_path,
    templates_repo,
    wait_epoch,
)

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
@pytest.mark.skip
async def test_emissions(local_chain):
    logging.info("Testing test_emissions")
    netuid = 1
    # Register root as Alice - the subnet owner and validator
    alice_keypair, alice_exec_command, alice_wallet = setup_wallet("//Alice")
    alice_exec_command(RegisterSubnetworkCommand, ["s", "create"])
    # Verify subnet <netuid> created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [netuid]
    ).serialize(), "Subnet wasn't created successfully"

    # Register Bob as miner
    bob_keypair, bob_exec_command, bob_wallet = setup_wallet("//Bob")

    # Register Alice as neuron to the subnet
    alice_exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            str(netuid),
        ],
    )

    # Register Bob as neuron to the subnet
    bob_exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            str(netuid),
        ],
    )

    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    # assert two neurons are in network
    assert len(subtensor.neurons(netuid=netuid)) == 2

    # Alice to stake to become to top neuron after the first epoch
    alice_exec_command(
        StakeCommand,
        [
            "stake",
            "add",
            "--amount",
            "10000",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    # register Alice as validator
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/neurons/validator.py"',
            "--no_prompt",
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
    # run validator in the background

    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    logging.info("Neuron Alice is now validating")
    await asyncio.sleep(5)

    # register validator with root network
    alice_exec_command(
        RootRegisterCommand,
        [
            "root",
            "register",
            "--netuid",
            str(netuid),
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    await wait_epoch(subtensor, netuid=netuid)

    alice_exec_command(
        RootSetBoostCommand,
        [
            "root",
            "boost",
            "--netuid",
            str(netuid),
            "--increase",
            "1000",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    # register Bob as miner
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/neurons/miner.py"',
            "--no_prompt",
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

    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    logging.info("Neuron Bob is now mining")
    await wait_epoch(subtensor)

    logging.warning("Setting root set weights")
    alice_exec_command(
        RootSetWeightsCommand,
        [
            "root",
            "weights",
            "--netuid",
            str(netuid),
            "--weights",
            "0.3",
            "--wallet.name",
            "default",
            "--wallet.hotkey",
            "default",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
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
            str(netuid),
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
    await wait_epoch(subtensor)

    await asyncio.sleep(
        5
    )  # wait for 5 seconds for the metagraph and subtensor to refresh with latest data

    # refresh metagraph
    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    # get current emissions and validate that Alice has gotten tao
    weights = [(0, [(0, 65535), (1, 65535)])]
    assert (
        subtensor.weights(netuid=netuid) == weights
    ), "Weights set vs weights in subtensor don't match"

    neurons = subtensor.neurons(netuid=netuid)
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
        subtensor.get_emission_value_by_subnet(netuid=netuid) > 0
    ), (
        "Emissions are not greated than 0"
    )  # emission on this subnet is strictly greater than 0
    logging.info("Passed test_emissions")
