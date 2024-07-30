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


@pytest.mark.skip
@pytest.mark.parametrize("local_chain", [False], indirect=True)
def test_faucet(local_chain):
    logging.info("Testing test_faucet")
    # Register root as Alice
    keypair, exec_command, wallet = setup_wallet("//Alice")
    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # Verify subnet 1 created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [1]
    ).serialize(), "Subnet wasn't created successfully"

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

    # verify current balance
    wallet_balance = subtensor.get_balance(keypair.ss58_address)
    assert wallet_balance.tao == 998999.0, "Balance wasn't as expected"

    # run faucet 3 times
    for i in range(3):
        logging.info(f"faucet run #:{i + 1}")
        try:
            exec_command(
                RunFaucetCommand,
                [
                    "wallet",
                    "faucet",
                    "--wallet.name",
                    wallet.name,
                    "--wallet.hotkey",
                    "default",
                    "--wait_for_inclusion",
                    "True",
                    "--wait_for_finalization",
                    "True",
                ],
            )
            logging.info(
                f"wallet balance is {subtensor.get_balance(keypair.ss58_address).tao} tao"
            )
        except SystemExit as e:
            logging.warning(
                "Block not generated fast enough to be within 3 block seconds window."
            )
            # Handle the SystemExit exception
            assert e.code == 1  # Assert that the exit code is 1
        except Exception as e:
            logging.warning(f"Unexpected exception occurred on faucet: {e}")

    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    new_wallet_balance = subtensor.get_balance(keypair.ss58_address)
    # verify balance increase
    assert (
        wallet_balance.tao < new_wallet_balance.tao
    ), "Old wallet balance is not less than the new wallet"
    logging.info("Passed test_faucet")
