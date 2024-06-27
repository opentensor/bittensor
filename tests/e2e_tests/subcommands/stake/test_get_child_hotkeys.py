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
* No children hotkeys at subnet creation
* Subnet owner an set a child hotkey
* Child hotkey is set properly with proportion
"""


def test_set_child(local_chain, capsys):
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
    assert "0 children" in output

    # Assert no child hotkeys on subnet
    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    assert (
        subtensor.get_children(hotkey=alice_keypair.ss58_address, netuid=1) == 0
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
            bob_wallet.hotkey_str,
            "--hotkey",
            alice_wallet.hotkey_str,
            "--proportion",
            "0.3",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    assert (
        subtensor.get_children(hotkey=alice_keypair.ss58_address, netuid=1) == 1
    ), "failed to set child hotkey"

    output = capsys.readouterr().out
    assert "✅ Set child hotkey." in output

    # Run get children with a child
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
    assert "1 children" in output
    assert "0.3 proportion" in output
