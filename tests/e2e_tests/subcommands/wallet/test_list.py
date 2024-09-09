from bittensor.commands.list import ListCommand
from bittensor.commands.wallets import WalletCreateCommand
from bittensor.subtensor import subtensor

from ...utils import setup_wallet


def test_wallet_list(capsys):
    """
    Test the listing of wallets in the Bittensor network.

    Steps:
        1. Set up a default wallet
        2. List existing wallets and verify the default setup
        3. Create a new wallet
        4. List wallets again and verify the new wallet is present

    Raises:
        AssertionError: If any of the checks or verifications fail
    """

    wallet_path_name = "//Alice"
    base_path = f"/tmp/btcli-e2e-wallet-list-{wallet_path_name.strip('/')}"
    keypair, exec_command, wallet = setup_wallet(wallet_path_name)

    # List initial wallets
    exec_command(
        ListCommand,
        [
            "wallet",
            "list",
        ],
    )

    captured = capsys.readouterr()
    # Assert the default wallet is present in the display
    assert "default" in captured.out
    assert "└── default" in captured.out

    # Create a new wallet
    exec_command(
        WalletCreateCommand,
        [
            "wallet",
            "create",
            "--wallet.name",
            "new_wallet",
            "--wallet.hotkey",
            "new_hotkey",
            "--no_password",
            "--overwrite_coldkey",
            "--overwrite_hotkey",
            "--no_prompt",
            "--wallet.path",
            base_path,
        ],
    )

    # List wallets again
    exec_command(
        ListCommand,
        [
            "wallet",
            "list",
        ],
    )

    captured = capsys.readouterr()

    # Verify the new wallet is displayed
    assert "new_wallet" in captured.out
    assert "new_hotkey" in captured.out
