import bittensor
from bittensor.commands import (
    RegisterCommand,
    StakeCommand,
    RegisterSubnetworkCommand,
    GetChildrenCommand,
    SetChildCommand,
)
from tests.e2e_tests.utils import setup_wallet

"""
Test the view child hotkeys on a subnet mechanism. 

Verify that:
* Call GetChildren without any children returns empty list
* Call GetChildren with children returns a table with children
"""


def test_get_children_info(local_chain, capsys):
    # Register root as Alice
    alice_keypair, alice_exec_command, alice_wallet = setup_wallet("//Alice")
    alice_exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    # Register Bob
    bob_keypair, bob_exec_command, bob_wallet = setup_wallet("//Bob")

    # Register Alice neuron to the subnet
    alice_exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
        ],
    )

    # Alice to Stake to become to top neuron after the first epoch
    alice_exec_command(
        StakeCommand,
        [
            "stake",
            "add",
            "--amount",
            "100000",
        ],
    )

    # Register Bob neuron to the subnet

    bob_exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
        ],
    )

    # Run get children with no children
    # btcli stake get_children --netuid 1
    alice_exec_command(
        GetChildrenCommand,
        [
            "stake",
            "get_children",
            "--netuid",
            "1",
        ],
    )

    output = capsys.readouterr().out
    assert "There are currently no child hotkeys on subnet 1" in output

    # Assert no child hotkeys on subnet
    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    assert (
        subtensor.get_children_info(netuid=1) == []
    ), "Child hotkeys are already set on new subnet. "

    # Run set child
    # btcli stake set_child --child <child_hotkey> --hotkey <parent_hotkey> --netuid 1 --proportion 0.3
    alice_exec_command(
        SetChildCommand,
        [
            "stake",
            "set_child",
            "--netuid",
            "1",
            "--child",
            str(bob_keypair.ss58_address),
            "--hotkey",
            str(alice_keypair.ss58_address),
            "--proportion",
            "0.3",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    assert len(subtensor.get_children_info(netuid=1)) == 1, "failed to set child hotkey"

    output = capsys.readouterr().out
    assert "✅ Finalized" in output

    # Run get children with a child
    # btcli stake get_children --netuid 1
    alice_exec_command(
        GetChildrenCommand,
        ["stake", "get_children", "--netuid", "1"],
    )

    output = capsys.readouterr().out
    # Assert table shows 1 child key with its data
    assert (
        "Total (  1) | Total (  1) | Total (  1.000000) | Total (2147483648.0000) | Avg (\n0.0000) | Avg (    0.0000) | Total (  0.179995)"
        in output
    )
