import os
import re

from bittensor.commands.list import ListCommand
from bittensor.commands.wallets import (
    NewColdkeyCommand,
    NewHotkeyCommand,
    WalletCreateCommand,
)
from bittensor.subtensor import subtensor

from ...utils import setup_wallet

"""
Test the wallet with various commands

Verify commands:
* btcli w list
* btcli w create
* btcli w new_coldkey
* btcli w new_hotkey
"""


# This function physically checks the existence of wallets (coldkeys and hotkeys)
def verify_wallet_dir(base_path, wallet_name, hotkey_name=None):
    wallet_path = os.path.join(base_path, wallet_name)

    # Check if wallet directory exists
    if not os.path.isdir(wallet_path):
        return False, f"Wallet directory {wallet_name} not found in {base_path}"

    # Check if coldkey file exists
    coldkey_path = os.path.join(wallet_path, "coldkey")
    if not os.path.isfile(coldkey_path):
        return False, f"Coldkey file not found in {wallet_name}"

    # Check if hotkey directory and file exists
    if hotkey_name:
        hotkeys_path = os.path.join(wallet_path, "hotkeys")
        if not os.path.isdir(hotkeys_path):
            return False, f"Hotkeys directory not found in {wallet_name}"

        hotkey_file_path = os.path.join(hotkeys_path, hotkey_name)
        if not os.path.isfile(hotkey_file_path):
            return (
                False,
                f"Hotkey file {hotkey_name} not found in {wallet_name}/hotkeys",
            )

    return True, f"Wallet {wallet_name} verified successfully"


def verify_key_pattern(output, wallet_name):
    # Pattern: <wallet_name> <space> <Key starting with '5' followed by 47 chars>
    pattern = rf"{wallet_name}\s*\((5[A-Za-z0-9]{{47}})\)"

    # Find instance of the pattern
    match = re.search(pattern, output)

    # Assert instance is found
    assert match is not None, f"{wallet_name} not found in wallet list"

    # Assert key starts with 5
    assert match.group(1).startswith("5"), f"{wallet_name} should start with '5'"

    # Assert length of key is 48 characters
    assert (
        len(match.group(1)) == 48
    ), f"Key for {wallet_name} should be 48 characters long"
    return match.group(1)


def test_wallet_creations(local_chain: subtensor, capsys):
    wallet_path_name = "//Alice"
    base_path = f"/tmp/btcli-e2e-wallet-{wallet_path_name.strip('/')}"
    keypair, exec_command, wallet = setup_wallet(wallet_path_name)

    exec_command(
        ListCommand,
        [
            "wallet",
            "list",
        ],
    )

    captured = capsys.readouterr()
    # Assert the coldkey and hotkey are present in the display with keys
    assert "default" and "└── default" in captured.out
    wallet_status, message = verify_wallet_dir(base_path, "default", "default")
    assert wallet_status, message

    # -----------------------------
    # Command 1: <btcli w create>
    # -----------------------------
    # Create a new wallet (coldkey + hotkey)
    exec_command(
        WalletCreateCommand,
        [
            "wallet",
            "create",
            "--wallet.name",
            "alice_create_wallet",
            "--wallet.hotkey",
            "alice_create_wallet_hotkey",
            "--no_password",
            "--overwrite_coldkey",
            "--overwrite_hotkey",
            "--no_prompt",
            "--wallet.path",
            base_path,
        ],
    )

    # List the wallets
    exec_command(
        ListCommand,
        [
            "wallet",
            "list",
        ],
    )

    captured = capsys.readouterr()

    # Verify coldkey "alice_create_wallet" is displayed with key
    verify_key_pattern(captured.out, "alice_create_wallet")

    # Verify hotkey "alice_create_wallet_hotkey" is displayed with key
    verify_key_pattern(captured.out, "alice_create_wallet_hotkey")

    # Physically verify "alice_create_wallet" and "alice_create_wallet_hotkey" are present
    wallet_status, message = verify_wallet_dir(
        base_path, "alice_create_wallet", "alice_create_wallet_hotkey"
    )
    assert wallet_status, message

    # -----------------------------
    # Command 2: <btcli w new_coldkey>
    # -----------------------------
    # Create a new wallet (coldkey)
    exec_command(
        NewColdkeyCommand,
        [
            "wallet",
            "new_coldkey",
            "--wallet.name",
            "alice_new_coldkey",
            "--no_password",
            "--no_prompt",
            "--overwrite_coldkey",
            "--wallet.path",
            base_path,
        ],
    )

    # List the wallets
    exec_command(
        ListCommand,
        [
            "wallet",
            "list",
        ],
    )

    captured = capsys.readouterr()

    # Verify coldkey "alice_new_coldkey" is displayed with key
    verify_key_pattern(captured.out, "alice_create_wallet")

    # Physically verify "alice_new_coldkey" is present
    wallet_status, message = verify_wallet_dir(base_path, "alice_new_coldkey")
    assert wallet_status, message

    # -----------------------------
    # Command 3: <btcli w new_hotkey>
    # -----------------------------
    # Create a new hotkey for alice_new_coldkey wallet
    exec_command(
        NewHotkeyCommand,
        [
            "wallet",
            "new_hotkey",
            "--wallet.name",
            "alice_new_coldkey",
            "--wallet.hotkey",
            "alice_new_hotkey",
            "--no_prompt",
            "--overwrite_hotkey",
            "--wallet.path",
            base_path,
        ],
    )

    # List the wallets
    exec_command(
        ListCommand,
        [
            "wallet",
            "list",
        ],
    )
    captured = capsys.readouterr()

    # Verify hotkey "alice_new_hotkey" is displyed with key
    verify_key_pattern(captured.out, "alice_new_hotkey")

    # Physically verify "alice_new_coldkey" and "alice_new_hotkey" are present
    wallet_status, message = verify_wallet_dir(
        base_path, "alice_new_coldkey", "alice_new_hotkey"
    )
    assert wallet_status, message
