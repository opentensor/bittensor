import pytest

import bittensor
from bittensor import logging
from bittensor.commands import (
    RegisterCommand,
    RegisterSubnetworkCommand,
    RunFaucetCommand,
)
from tests.e2e_tests.utils import (
    setup_wallet,
)


@pytest.mark.asyncio
async def test_faucet(local_chain):
    # Register root as Alice
    alice_keypair, exec_command, wallet_path = setup_wallet("//Alice")
    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    # Register a neuron to the subnet
    exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
            "--wallet.name",
            "default",
            "--wallet.hotkey",
            "default",
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--no_prompt",
        ],
    )

    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    alice_wallet_balance = subtensor.get_balance(alice_keypair.ss58_address)
    # verify current balance
    assert alice_wallet_balance.tao == 998999.0

    # run faucet 3 times
    for i in range(3):
        logging.info(f"running faucet for the {i}th time.")
        print(f"running faucet for the {i}th time.")
        exec_command(
            RunFaucetCommand,
            [
                "wallet",
                "faucet",
                "--wallet.name",
                "default",
                "--wallet.hotkey",
                "default",
                "--subtensor.chain_endpoint",
                "ws://localhost:9945",
            ],
        )

    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    alice_wallet_balance = subtensor.get_balance(alice_keypair.ss58_address)
    # verify balance increase
    assert alice_wallet_balance.tao == 999899.0
