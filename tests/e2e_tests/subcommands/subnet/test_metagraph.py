import bittensor
from bittensor import logging
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
    logging.info("Testing test_metagraph_command")
    # Register root as Alice
    keypair, exec_command, wallet = setup_wallet("//Alice")

    # The rest of the test will use subnet 2
    netuid = 2

    for i in range(netuid):
        # Register subnet <i+1>
        exec_command(RegisterSubnetworkCommand, ["s", "create"])
        # Verify subnet <i+1> created successfully
        assert local_chain.query(
            "SubtensorModule", "NetworksAdded", [i + 1]
        ).serialize(), f"Subnet {netuid+1} wasn't created successfully"

    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    metagraph = subtensor.metagraph(netuid=netuid)

    # Assert metagraph is empty
    assert len(metagraph.uids) == 0, "Metagraph is not empty"

    # Execute btcli metagraph command
    exec_command(MetagraphCommand, ["subnet", "metagraph", "--netuid", str(netuid)])

    captured = capsys.readouterr()

    # Assert metagraph is printed for netuid

    assert (
        f"Metagraph: net: local:{netuid}" in captured.out
    ), f"Netuid {netuid} was not displayed in metagraph"

    # Register Bob as neuron to the subnet
    bob_keypair, bob_exec_command, bob_wallet = setup_wallet("//Bob")
    bob_exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            str(netuid),
        ],
    )

    captured = capsys.readouterr()

    # Assert neuron was registered

    assert "✅ Registered" in captured.out, "Neuron was not registered"

    # Refresh the metagraph
    metagraph = subtensor.metagraph(netuid=netuid)

    # Assert metagraph has registered neuron
    assert len(metagraph.uids) == 1, "Metagraph doesn't have exactly 1 neuron"
    assert (
        metagraph.hotkeys[0] == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    ), "Neuron's hotkey in metagraph doesn't match"
    # Execute btcli metagraph command
    exec_command(MetagraphCommand, ["subnet", "metagraph", "--netuid", str(netuid)])

    captured = capsys.readouterr()

    # Assert the neuron is registered and displayed
    assert (
        f"Metagraph: net: local:{netuid}" in captured.out and "N: 1/1" in captured.out
    ), "Neuron isn't displayed in metagraph"

    # Create a secondary metagraph to test .load() later on
    metagraph_pre_dave = subtensor.metagraph(netuid=netuid)

    # Register Dave as neuron to the subnet
    dave_keypair, dave_exec_command, dave_wallet = setup_wallet("//Dave")
    dave_exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            str(netuid),
        ],
    )

    captured = capsys.readouterr()

    # Assert neuron was registered

    assert "✅ Registered" in captured.out, "Neuron was not registered"

    # Refresh the metagraph
    metagraph = subtensor.metagraph(netuid=netuid)

    # Assert metagraph has registered neuron
    assert len(metagraph.uids) == 2
    assert (
        metagraph.hotkeys[1] == "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"
    ), "Neuron's hotkey in metagraph doesn't match"

    # Execute btcli metagraph command
    exec_command(MetagraphCommand, ["subnet", "metagraph", "--netuid", str(netuid)])

    captured = capsys.readouterr()

    # Assert the neuron is registered and displayed
    assert f"Metagraph: net: local:{netuid}" in captured.out
    assert "N: 2/2" in captured.out

    # Check save/load cycle
    metagraph.save()
    metagraph_pre_dave.load()

    # Assert in progressive detail
    assert len(metagraph.uids) == len(metagraph_pre_dave.uids)
    assert (metagraph.uids == metagraph_pre_dave.uids).all()

    assert len(metagraph.axons) == len(metagraph_pre_dave.axons)
    assert metagraph.axons[1].hotkey == metagraph_pre_dave.axons[1].hotkey
    assert metagraph.axons == metagraph_pre_dave.axons

    assert len(metagraph.neurons) == len(metagraph_pre_dave.neurons)
    assert metagraph.neurons == metagraph_pre_dave.neurons

    logging.info("Passed test_metagraph_command")
