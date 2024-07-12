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
)
from tests.e2e_tests.utils import (
    setup_wallet,
    template_path,
    repo_name,
    wait_epoch,
)

logging.basicConfig(level=logging.INFO)

"""
Test the incentive mechanism. 

Verify that for the miner:
* trust
* rank
* consensus
* incentive
are updated with proper values after an epoch has passed. 

For the validator verify that:
* validator_permit
* validator_trust
* dividends
* stake
are updated with proper values after an epoch has passed. 

"""


@pytest.mark.asyncio
async def test_incentive(local_chain):
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

    # register Bob as miner
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{repo_name}/neurons/miner.py"',
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

    miner_process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    await asyncio.sleep(
        5
    )  # wait for 5 seconds for the metagraph to refresh with latest data

    # register Alice as validator
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{repo_name}/neurons/validator.py"',
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

    validator_process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    await asyncio.sleep(
        5
    )  # wait for 5 seconds for the metagraph and subtensor to refresh with latest data

    # register validator with root network
    alice_exec_command(
        RootRegisterCommand,
        [
            "root",
            "register",
            "--netuid",
            "1",
            "--wallet.name",
            "default",
            "--wallet.hotkey",
            "default",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
        ],
    )

    alice_exec_command(
        RootSetBoostCommand,
        [
            "root",
            "boost",
            "--netuid",
            "1",
            "--increase",
            "100",
            "--wallet.name",
            "default",
            "--wallet.hotkey",
            "default",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
        ],
    )

    # get latest metagraph
    metagraph = bittensor.metagraph(netuid=1, network="ws://localhost:9945")

    # get current emissions
    bob_neuron = metagraph.neurons[1]
    assert bob_neuron.incentive == 0
    assert bob_neuron.consensus == 0
    assert bob_neuron.rank == 0
    assert bob_neuron.trust == 0

    alice_neuron = metagraph.neurons[0]
    assert alice_neuron.validator_permit is False
    assert alice_neuron.dividends == 0
    assert alice_neuron.stake.tao == 10_000.0
    assert alice_neuron.validator_trust == 0

    # wait until 360 blocks pass (subnet tempo)
    wait_epoch(360, subtensor)

    # for some reason the weights do not get set through the template. Set weight manually.
    alice_wallet = bittensor.wallet()
    alice_wallet._hotkey = alice_keypair
    subtensor._do_set_weights(
        wallet=alice_wallet,
        uids=[1],
        vals=[65535],
        netuid=1,
        version_key=0,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # wait epoch until weight go into effect
    wait_epoch(360, subtensor)

    # refresh metagraph
    metagraph = bittensor.metagraph(netuid=1, network="ws://localhost:9945")

    # get current emissions and validate that Alice has gotten tao
    bob_neuron = metagraph.neurons[1]
    assert bob_neuron.incentive == 1
    assert bob_neuron.consensus == 1
    assert bob_neuron.rank == 1
    assert bob_neuron.trust == 1

    alice_neuron = metagraph.neurons[0]
    assert alice_neuron.validator_permit is True
    assert alice_neuron.dividends == 1
    assert alice_neuron.stake.tao == 10_000.0
    assert alice_neuron.validator_trust == 1
