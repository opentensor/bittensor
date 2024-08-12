import asyncio
import sys
import textwrap

import pytest

import bittensor
from bittensor import logging
from bittensor.commands import (
    RegisterCommand,
    RegisterSubnetworkCommand,
)
from bittensor.utils import networking
from tests.e2e_tests.utils import (
    setup_wallet,
    template_path,
    templates_repo,
)

"""
Test the axon mechanism. 

Verify that:
* axon is registered on network as a miner
* ip
* type
* port

are set correctly, and that the miner is currently running

"""

INDENT = "   "


@pytest.mark.asyncio
async def test_axon(local_chain):
    logging.info("Testing test_axon")
    netuid = 1
    # Register root as Alice
    alice_keypair, exec_command, wallet = setup_wallet("//Alice")
    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # Verify subnet <netuid> created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [netuid]
    ).serialize(), "Subnet wasn't created successfully"

    # Register a neuron to the subnet
    exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            str(netuid),
        ],
    )

    metagraph = bittensor.metagraph(netuid=netuid, network="ws://localhost:9945")

    # validate one miner with ip of none
    old_axon = metagraph.axons[0]

    assert len(metagraph.axons) == 1, "Expected 1 axon, but got len(metagraph.axons)"
    assert old_axon.hotkey == alice_keypair.ss58_address
    assert old_axon.coldkey == alice_keypair.ss58_address
    assert old_axon.ip == "0.0.0.0"
    assert old_axon.port == 0
    assert old_axon.ip_type == 0

    # register miner
    # "python neurons/miner.py --netuid 1 --subtensor.chain_endpoint ws://localhost:9945 --wallet.name wallet.name --wallet.hotkey wallet.hotkey.ss58_address"
    args = [
        f"{template_path}{templates_repo}/neurons/miner.py",
        "--no_prompt",
        "--netuid",
        str(netuid),
        "--subtensor.network",
        "local",
        "--subtensor.chain_endpoint",
        "ws://localhost:9945",
        "--wallet.path",
        wallet.path,
        "--wallet.name",
        wallet.name,
        "--wallet.hotkey",
        "default",
    ]

    cmd = sys.executable + " " + " ".join(args)

    axon_process = await asyncio.create_subprocess_exec(
        sys.executable,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.sleep(1)  # wait for the miner to start up or fail
    if axon_process.returncode != None:
        stdout, stderr = await axon_process.communicate()
        stdout = textwrap.indent(stdout.decode("UTF8"), INDENT)
        stderr = textwrap.indent(stderr.decode("UTF8"), INDENT)
        assert (
            axon_process.returncode == None
        ), f"miner process stopped immediately or did not start:\n{cmd}\nstdout:\n{stdout}\nstderr:\n{stderr}"

    logging.info(f"Neuron Alice is now mining: {axon_process}")
    await asyncio.sleep(
        5
    )  # wait for 5 seconds for the metagraph to refresh with latest data
    if axon_process.returncode != None:
        stdout, stderr = await axon_process.communicate()
        stdout = textwrap.indent(stdout.decode("UTF8"), INDENT)
        stderr = textwrap.indent(stderr.decode("UTF8"), INDENT)
        assert (
            axon_process.returncode == None
        ), f"miner process stopped unexpectedly:\n{cmd}\nstdout:\n{stdout}\nstderr:\n{stderr}"

    # refresh metagraph
    metagraph = bittensor.metagraph(netuid=netuid, network="ws://localhost:9945")
    updated_axon = metagraph.axons[0]
    external_ip = networking.get_external_ip()

    assert len(metagraph.axons) == 1
    assert updated_axon.ip == external_ip
    assert updated_axon.ip_type == networking.ip_version(external_ip)
    assert updated_axon.port == 8091
    assert updated_axon.hotkey == alice_keypair.ss58_address
    assert updated_axon.coldkey == alice_keypair.ss58_address

    # Terminate miner and make sure nothing bad happened.
    if axon_process.returncode == None:
        logging.info("terminating miner")
        try:
            # try/except to prevent exception in case the process just ends now
            axon_process.terminate()
        except:
            pass
    await asyncio.sleep(1)

    if axon_process.returncode == None:
        logging.info("killing miner")
        try:
            # try/except to prevent exception in case the process just ends now
            axon_process.kill()
        except:
            pass

    stdout, stderr = await axon_process.communicate()
    stdout = textwrap.indent(stdout.decode("UTF8"), INDENT)
    stderr = textwrap.indent(stderr.decode("UTF8"), INDENT)
    assert (
        "Exception" not in stderr
    ), f"miner output indicates issue:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    logging.info("Passed test_axon")
