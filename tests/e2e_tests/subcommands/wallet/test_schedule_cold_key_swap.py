import pytest

import bittensor
from bittensor import logging
from bittensor.commands import (
    RegisterCommand,
    RegisterSubnetworkCommand,
    RunFaucetCommand,
    ScheduleColdKeySwapCommand,
)
from tests.e2e_tests.utils import (
    setup_wallet,
)


@pytest.mark.parametrize("local_chain", [False], indirect=True)
def test_faucet(local_chain):
    # Register root as Alice
    keypair, exec_command, wallet_path = setup_wallet("//Alice")
    keypair_bob, _, _ = setup_wallet("//Bob")

    exec_command(
        ScheduleColdKeySwapCommand,
        [
            "wallet",
            "schedule_coldkey_swap",
            "--new_coldkey",
            keypair_bob.ss58_address,
        ],
    )
