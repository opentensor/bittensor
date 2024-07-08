import pytest

from bittensor.commands import (
    ScheduleColdKeySwapCommand,
)
from tests.e2e_tests.utils import (
    setup_wallet,
)


@pytest.mark.parametrize("local_chain", [False], indirect=True)
def test_schedule_coldkey_swap(local_chain, capsys):
    # Register root as Alice
    keypair, exec_command, wallet_path = setup_wallet("//Alice")
    keypair_bob, _, _ = setup_wallet("//Bob")

    block = local_chain.query(
        "SubtensorModule", "ColdkeyArbitrationBlock", [keypair.ss58_address]
    )

    assert block == 0

    exec_command(
        ScheduleColdKeySwapCommand,
        [
            "wallet",
            "schedule_coldkey_swap",
            "--new_coldkey",
            keypair_bob.ss58_address,
        ],
    )
    output = capsys.readouterr().out
    assert (
        "Good news. There has been no previous key swap initiated for your coldkey swap."
        in output
    )

    block = local_chain.query(
        "SubtensorModule", "ColdkeyArbitrationBlock", [keypair.ss58_address]
    )

    assert block == 1

    exec_command(
        ScheduleColdKeySwapCommand,
        [
            "wallet",
            "schedule_coldkey_swap",
            "--new_coldkey",
            keypair_bob.ss58_address,
        ],
    )

    output = capsys.readouterr().out
    assert "There has been a swap request made for this key previously." in output
    block = local_chain.query(
        "SubtensorModule", "ColdkeyArbitrationBlock", [keypair.ss58_address]
    )

    assert block == 2

    exec_command(
        ScheduleColdKeySwapCommand,
        [
            "wallet",
            "schedule_coldkey_swap",
            "--new_coldkey",
            keypair_bob.ss58_address,
        ],
    )

    output = capsys.readouterr().out
    assert "This key is currently in arbitration." in output
