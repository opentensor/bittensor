import subprocess
import sys
import time

import bittensor
from bittensor.commands import (
    RegisterCommand,
    RegisterSubnetworkCommand,
)
from tests.e2e_tests.utils import (
    setup_wallet,
    uninstall_templates,
    template_path,
    repo_name,
)


def test_parameter_validation_from_subtensor(local_chain):
    # Register root as Alice
    alice_keypair, exec_command, wallet_path = setup_wallet("//Alice")
    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # define values
    uid = 0

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    # Register a neuron to the subnet
    exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
            "--wallet.name",
            "default",
            "--wallet.hotkey",
            "default",
            "--wallet.path",
            wallet_path,
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--no_prompt",
        ],
    )

    # Create a test wallet and set the coldkey, coldkeypub, and hotkey
    # wallet = bittensor.wallet(path="/tmp/btcli-wallet")
    # wallet.set_coldkey(keypair=alice_keypair, encrypt=False, overwrite=True)
    # wallet.set_coldkeypub(keypair=alice_keypair, encrypt=False, overwrite=True)
    # wallet.set_hotkey(keypair=alice_keypair, encrypt=False, overwrite=True)

    metagraph = bittensor.metagraph(netuid=1, network="ws://localhost:9945")
    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    # validate miner with stake, ip, hotkey
    new_axon = metagraph.axons[0]
    assert new_axon.hotkey == alice_keypair.ss58_address
    assert new_axon.coldkey == alice_keypair.ss58_address
    assert new_axon.ip == "0.0.0.0"
    assert new_axon.port == 0
    assert new_axon.ip_type == 0

    # register miner
    # "python neurons/miner.py --netuid 1 --subtensor.chain_endpoint ws://localhost:9945 --wallet.name wallet.name --wallet.hotkey wallet.hotkey.ss58_address"

    process = subprocess.Popen(
        [
            sys.executable,
            f"{template_path}/{repo_name}/neurons/miner.py",
            "--no_prompt",
            "--netuid",
            "1",
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--wallet.path",
            wallet_path,
            "--wallet.name",
            "default",
            "--wallet.hotkey",
            "default",
        ],
        shell=True,
        stdout=subprocess.PIPE,
    )

    try:
        outs, errs = process.communicate(timeout=15)
        time.sleep(15)
    except subprocess.TimeoutExpired:
        process.kill()
    # validate miner with new ip
    new_axon = metagraph.axons[0]

    uninstall_templates(local_chain.templates_dir)
