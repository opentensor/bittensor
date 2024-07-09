from bittensor.commands import (
    ScheduleColdKeySwapCommand,
    CheckColdKeySwapCommand,
)
from tests.e2e_tests.utils import (
    setup_wallet,
)


def test_check_and_schedule_coldkey_swap(local_chain, capsys):
    # Register root as Alice
    alice_keypair, alice_exec_command, alice_wallet_path = setup_wallet("//Alice")
    bob_keypair, bob_exec_command, bob_wallet_path = setup_wallet("//Bob")

    # Get status for cold keys
    alice_exec_command(
        CheckColdKeySwapCommand,
        [
            "wallet",
            "check_coldkey_swap",
        ],
    )

    # Assert no cold key swaps in place.
    output = capsys.readouterr().out
    assert "There has been no previous key swap initiated for your coldkey." in output

    # Schedule a swap
    alice_exec_command(
        ScheduleColdKeySwapCommand,
        [
            "wallet",
            "schedule_coldkey_swap",
            "--new_coldkey",
            bob_keypair.ss58_address,
            "--prompt",
            "False",
        ],
    )

    # Assert swap scheduled successfully.
    output = capsys.readouterr().out
    assert "Scheduled Cold Key Swap Successfully." in output

    # Run check cold key swap status again. -> we should see that a swap has been scheduled already
    alice_exec_command(
        CheckColdKeySwapCommand,
        [
            "wallet",
            "check_coldkey_swap",
        ],
    )

    # Assert one swap has already been scheduled
    output = capsys.readouterr().out
    assert "There has been 1 swap request made for this coldkey already." in output

    # Schedule another swap.
    alice_exec_command(
        ScheduleColdKeySwapCommand,
        [
            "wallet",
            "schedule_coldkey_swap",
            "--new_coldkey",
            bob_keypair.ss58_address,
        ],
    )

    output = capsys.readouterr().out
    assert "Scheduled Cold Key Swap Successfully." in output

    # Run check cold key swap status again. -> we should see that its in arbitration
    alice_exec_command(
        CheckColdKeySwapCommand,
        [
            "wallet",
            "check_coldkey_swap",
        ],
    )

    # Assert coldkey is currently in arbitration
    output = capsys.readouterr().out
    assert "This coldkey is currently in arbitration with a total swaps of 2." in output
