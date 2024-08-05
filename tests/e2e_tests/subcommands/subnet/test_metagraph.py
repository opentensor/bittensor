import bittensor
from bittensor.commands import (
    MetagraphCommand,
    RegisterCommand,
    RegisterSubnetworkCommand,
)
from tests.e2e_tests.utils import setup_wallet

"""
Test the metagraph command before and after registering neurons. 

Verify that:
* Metagraph gets displayed
* Initially empty
-------------------------
* Register 2 neurons one by one 
* Ensure both are visible in metagraph
"""


def test_metagraph_command(local_chain, capsys):
    # Register root as Alice
    keypair, exec_command, wallet = setup_wallet("//Alice")
    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    metagraph = subtensor.metagraph(netuid=1)

    # Assert metagraph is empty
    assert len(metagraph.uids) == 0

    # Execute btcli metagraph command
    exec_command(MetagraphCommand, ["subnet", "metagraph", "--netuid", "1"])

    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    # Assert metagraph is printed for netuid 1
    assert "Metagraph: net: local:1" in lines[2]

    # Register Bob as neuron to the subnet
    bob_keypair, bob_exec_command, bob_wallet = setup_wallet("//Bob")
    bob_exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
        ],
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    # Assert neuron was registered
    assert "✅ Registered" in lines[3]

    # Refresh the metagraph
    metagraph = subtensor.metagraph(netuid=1)

    # Assert metagraph has registered neuron
    assert len(metagraph.uids) == 1
    assert metagraph.hotkeys[0] == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    # Execute btcli metagraph command
    exec_command(MetagraphCommand, ["subnet", "metagraph", "--netuid", "1"])

    captured = capsys.readouterr()

    # Assert the neuron is registered and displayed
    assert "Metagraph: net: local:1" and "N: 1/1" in captured.out

    # Register Dave as neuron to the subnet
    dave_keypair, dave_exec_command, dave_wallet = setup_wallet("//Dave")
    dave_exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
        ],
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    # Assert neuron was registered
    assert "✅ Registered" in lines[3]

    # Refresh the metagraph
    metagraph = subtensor.metagraph(netuid=1)

    # Assert metagraph has registered neuron
    assert len(metagraph.uids) == 2
    assert metagraph.hotkeys[1] == "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"

    # Execute btcli metagraph command
    exec_command(MetagraphCommand, ["subnet", "metagraph", "--netuid", "1"])

    captured = capsys.readouterr()

    # Assert the neuron is registered and displayed
    assert "Metagraph: net: local:1" and "N: 2/2" in captured.out
