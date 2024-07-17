import bittensor
from bittensor.commands import (
    RegisterCommand,
    StakeCommand,
    RegisterSubnetworkCommand,
    SetChildrenCommand,
    RevokeChildrenCommand,
    GetChildrenCommand,
)
from tests.e2e_tests.utils import setup_wallet
from unittest.mock import patch


def test_set_revoke_children(local_chain, capsys):
    # Setup
    alice_keypair, alice_exec_command, alice_wallet = setup_wallet("//Alice")
    bob_keypair, bob_exec_command, bob_wallet = setup_wallet("//Bob")
    eve_keypair, eve_exec_command, eve_wallet = setup_wallet("//Eve")

    alice_exec_command(RegisterSubnetworkCommand, ["s", "create"])
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    for wallet in [alice_wallet, bob_wallet, eve_wallet]:
        wallet.exec_command(RegisterCommand, ["s", "register", "--netuid", "1"])

    alice_exec_command(StakeCommand, ["stake", "add", "--amount", "100000"])

    # Test 1: Set multiple children
    alice_exec_command(
        SetChildrenCommand,
        [
            "stake",
            "set_children",
            "--netuid",
            "1",
            "--children",
            f"{bob_keypair.ss58_address},{eve_keypair.ss58_address}",
            "--hotkey",
            str(alice_keypair.ss58_address),
            "--proportions",
            "0.3,0.4",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    children_info = subtensor.get_children_info(netuid=1)[alice_keypair.ss58_address]
    assert len(children_info) == 2, "Failed to set children hotkeys"
    assert (
        children_info[0].proportion == 0.3 and children_info[1].proportion == 0.4
    ), "Incorrect proportions set"

    # Test 2: Get children information
    alice_exec_command(GetChildrenCommand, ["stake", "get_children", "--netuid", "1"])
    output = capsys.readouterr().out
    assert "Total (  2) | Total (  1) | Total (  0.700000)" in output

    # Test 3: Revoke all children
    alice_exec_command(
        RevokeChildrenCommand,
        [
            "stake",
            "revoke_children",
            "--netuid",
            "1",
            "--hotkey",
            str(alice_keypair.ss58_address),
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    assert (
        subtensor.get_children_info(netuid=1) == {}
    ), "Failed to revoke children hotkeys"

    # Test 4: Get children after revocation
    alice_exec_command(GetChildrenCommand, ["stake", "get_children", "--netuid", "1"])
    output = capsys.readouterr().out
    assert "There are currently no child hotkeys on subnet 1" in output


def test_error_handling(local_chain, capsys):
    alice_keypair, alice_exec_command, alice_wallet = setup_wallet("//Alice")
    bob_keypair, bob_exec_command, bob_wallet = setup_wallet("//Bob")

    # Test 5: Set children with invalid proportions
    alice_exec_command(
        SetChildrenCommand,
        [
            "stake",
            "set_children",
            "--netuid",
            "1",
            "--children",
            f"{bob_keypair.ss58_address}",
            "--hotkey",
            str(alice_keypair.ss58_address),
            "--proportions",
            "1.1",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )
    output = capsys.readouterr().out
    assert "Error" in output and "Invalid proportion" in output

    # Test 6: Set children on non-existent network
    alice_exec_command(
        SetChildrenCommand,
        [
            "stake",
            "set_children",
            "--netuid",
            "999",
            "--children",
            f"{bob_keypair.ss58_address}",
            "--hotkey",
            str(alice_keypair.ss58_address),
            "--proportions",
            "0.5",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )
    output = capsys.readouterr().out
    assert "Error" in output and "Subnet does not exist" in output

    # Test 7: Set child that's the same as parent
    alice_exec_command(
        SetChildrenCommand,
        [
            "stake",
            "set_children",
            "--netuid",
            "1",
            "--children",
            f"{alice_keypair.ss58_address}",
            "--hotkey",
            str(alice_keypair.ss58_address),
            "--proportions",
            "0.5",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )
    output = capsys.readouterr().out
    assert "Error" in output and "Child cannot be the same as parent" in output


def test_prompts_and_confirmations(local_chain, capsys):
    alice_keypair, alice_exec_command, alice_wallet = setup_wallet("//Alice")
    bob_keypair, bob_exec_command, bob_wallet = setup_wallet("//Bob")

    # Test 8: Set children with prompt (confirm)
    with patch("builtins.input", return_value="y"):
        alice_exec_command(
            SetChildrenCommand,
            [
                "stake",
                "set_children",
                "--netuid",
                "1",
                "--children",
                f"{bob_keypair.ss58_address}",
                "--hotkey",
                str(alice_keypair.ss58_address),
                "--proportions",
                "0.5",
                "--prompt",
            ],
        )
    output = capsys.readouterr().out
    assert "✅ Finalized" in output

    # Test 9: Revoke children with prompt (cancel)
    with patch("builtins.input", return_value="n"):
        alice_exec_command(
            RevokeChildrenCommand,
            [
                "stake",
                "revoke_children",
                "--netuid",
                "1",
                "--hotkey",
                str(alice_keypair.ss58_address),
                "--prompt",
            ],
        )
    output = capsys.readouterr().out
    assert "Cancelled" in output


def test_get_children_edge_cases(local_chain, capsys):
    alice_keypair, alice_exec_command, alice_wallet = setup_wallet("//Alice")

    # Test 10: Get children with invalid netuid
    alice_exec_command(GetChildrenCommand, ["stake", "get_children", "--netuid", "999"])
    output = capsys.readouterr().out
    assert "Error" in output and "Invalid netuid" in output

    # Test 11: Verify APY calculation
    # This test would require setting up a more complex scenario with emissions
    # and running for multiple epochs. For simplicity, we'll just check if the APY
    # column exists in the output.
    alice_exec_command(GetChildrenCommand, ["stake", "get_children", "--netuid", "1"])
    output = capsys.readouterr().out
    assert "APY" in output
