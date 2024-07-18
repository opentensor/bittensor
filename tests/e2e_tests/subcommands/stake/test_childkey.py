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
    """
    Test the setting and revoking of children hotkeys for staking.

    This test case covers the following scenarios:
    1. Setting multiple children hotkeys with specified proportions
    2. Retrieving children information
    3. Revoking all children hotkeys
    4. Verifying the absence of children after revocation

    The test uses three wallets (Alice, Bob, and Eve) and performs operations
    on a local blockchain.

    Args:
        local_chain: A fixture providing access to the local blockchain
        capsys: A pytest fixture for capturing stdout and stderr

    The test performs the following steps:
    - Set up wallets for Alice, Bob, and Eve
    - Create a subnet and register wallets
    - Add stake to Alice's wallet
    - Set Bob and Eve as children of Alice with specific proportions
    - Verify the children are set correctly
    - Get and verify children information
    - Revoke all children
    - Verify children are revoked
    - Check that no children exist after revocation

    This test ensures the proper functioning of setting children hotkeys,
    retrieving children information, and revoking children in the staking system.
    """
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
    """
    Test error handling scenarios for setting child hotkeys in the staking system.

    This test case covers the following error scenarios:
    1. Setting children with invalid proportions
    2. Attempting to set children on a non-existent network
    3. Trying to set a child hotkey that's the same as the parent

    Args:
        local_chain: Fixture providing access to the local blockchain
        capsys: Pytest fixture for capturing stdout and stderr

    The test performs the following steps:
    1. Set up wallets for Alice and Bob
    2. Attempt to set a child with an invalid proportion (> 1.0)
    3. Verify that an appropriate error message is displayed
    4. Try to set a child on a non-existent subnet (netuid 999)
    5. Confirm that an error about non-existent subnet is shown
    6. Attempt to set the parent (Alice) as its own child
    7. Check that an error preventing self-assignment as child is displayed

    This test ensures proper error handling and user feedback in various
    invalid scenarios when setting child hotkeys for staking.
    """
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
    """
    Test user prompts and confirmations for setting and revoking child hotkeys.

    This test case covers the following scenarios:
    1. Setting children with user confirmation prompt
    2. Attempting to revoke children with user cancellation

    Args:
        local_chain: Fixture providing access to the local blockchain
        capsys: Pytest fixture for capturing stdout and stderr

    The test performs the following steps:
    1. Set up wallets for Alice and Bob
    2. Simulate user confirming the setting of a child hotkey
       - Use Bob as Alice's child with 0.5 proportion
       - Verify that the operation is finalized successfully
    3. Simulate user cancelling the revocation of children
       - Attempt to revoke Alice's children
       - Verify that the operation is cancelled

    This test ensures that the user prompts work correctly for both
    confirming and cancelling operations related to child hotkeys when staking.
    It uses mocked user inputs to simulate user interactions.
    """
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
    """
    Test edge cases and specific scenarios for retrieving child hotkey information.

    This test case covers the following scenarios:
    1. Attempting to get children information with an invalid netuid
    2. Verifying the presence of APY information in the output

    Args:
        local_chain: Fixture providing access to the local blockchain
        capsys: Pytest fixture for capturing stdout and stderr

    The test performs the following steps:
    1. Set up a wallet for Alice
    2. Attempt to get children information with an invalid netuid (999)
       - Verify that an appropriate error message is displayed
    3. Retrieve children information for a valid netuid (1)
       - Check that the APY column is present in the output

    Note:
    - The APY calculation test is simplified and only checks for the presence
      of the "APY" column in the output. A more comprehensive test would involve
      setting up a complex scenario with emissions over multiple epochs.

    This test ensures proper error handling for invalid netuids and
    verifies that important staking information like APY is included in the output.
    """
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
