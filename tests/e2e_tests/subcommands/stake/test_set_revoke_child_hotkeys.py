import bittensor
from bittensor.commands import (
    RegisterCommand,
    StakeCommand,
    RegisterSubnetworkCommand,
    SetChildCommand,
    SetChildrenCommand,
)
from bittensor.commands.unstake import RevokeChildCommand
from tests.e2e_tests.utils import setup_wallet

"""
Test the set child hotkey singular mechanism. 

Verify that:
* No children hotkeys at subnet creation
* Subnet owner an set a child hotkey
* Child hotkey is set properly with proportion
* Subnet owner can revoke child hotkey
* Child hotkey is properly removed from subnet
"""


def test_set_revoke_child(local_chain, capsys):
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

    # Run revoke child
    # btcli stake revoke_child --child <child_hotkey> --hotkey <parent_hotkey> --netuid 1
    alice_exec_command(
        RevokeChildCommand,
        [
            "stake",
            "revoke_child",
            "--netuid",
            "1",
            "--child",
            bob_wallet.hotkey_str,
            "--hotkey",
            alice_wallet.hotkey_str,
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    assert (
        subtensor.get_children(hotkey=alice_keypair.ss58_address, netuid=1) == 0
    ), "failed to revoke child hotkey"

    output = capsys.readouterr().out
    assert "✅ Revoked child hotkey." in output


"""
Test the set children hotkey multiple mechanism. 

Verify that:
* No children hotkeys at subnet creation
* Subnet owner an set multiple children in one call
* Child hotkeys are set properly with correct proportions
"""


def test_set_children(local_chain, capsys):
    # Register root as Alice
    alice_keypair, alice_exec_command, alice_wallet = setup_wallet("//Alice")
    alice_exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    # Register Bob
    bob_keypair, bob_exec_command, bob_wallet = setup_wallet("//Bob")
    dan_keypair, dan_exec_command, dan_wallet = setup_wallet("//Dan")

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

    dan_exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
        ],
    )

    # Assert no child hotkeys on subnet
    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    assert (
        subtensor.get_children(hotkey=alice_keypair.ss58_address, netuid=1) == 0
    ), "Child hotkeys are already set on new subnet. "

    # Run set child
    # btcli stake set_child --child <child_hotkey> --hotkey <parent_hotkey> --netuid 1 --proportion 0.3
    alice_exec_command(
        SetChildrenCommand,
        [
            "stake",
            "set_children",
            "--netuid",
            "1",
            "--children",
            bob_wallet.hotkey_str + ", " + dan_wallet.hotkey_str,
            "--hotkey",
            alice_wallet.hotkey_str,
            "--proportion",
            "0.3, 0.3",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    assert (
        subtensor.get_children(hotkey=alice_keypair.ss58_address, netuid=1) == 2
    ), "failed to set children hotkeys"

    output = capsys.readouterr().out
    assert "✅ Set children hotkeys." in output
