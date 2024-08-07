import os
import re
import time
from typing import Dict, Optional, Tuple

from bittensor import logging
from bittensor.commands.list import ListCommand
from bittensor.commands.wallets import (
    NewColdkeyCommand,
    NewHotkeyCommand,
    RegenColdkeyCommand,
    RegenColdkeypubCommand,
    RegenHotkeyCommand,
    WalletCreateCommand,
)
from bittensor.subtensor import subtensor

from ...utils import setup_wallet

"""
Verify commands:

* btcli w list
* btcli w create
* btcli w new_coldkey
* btcli w new_hotkey
* btcli w regen_coldkey
* btcli w regen_coldkeypub
* btcli w regen_hotkey
"""


def verify_wallet_dir(
    base_path: str,
    wallet_name: str,
    hotkey_name: Optional[str] = None,
    coldkeypub_name: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Verifies the existence of wallet directory, coldkey, and optionally the hotkey.

    Args:
        base_path (str): The base directory path where wallets are stored.
        wallet_name (str): The name of the wallet directory to verify.
        hotkey_name (str, optional): The name of the hotkey file to verify. If None,
                                     only the wallet and coldkey file are checked.
        coldkeypub_name (str, optional): The name of the coldkeypub file to verify. If None
                                         only the wallet and coldkey is checked

    Returns:
        tuple: Returns a tuple containing a boolean and a message. The boolean is True if
               all checks pass, otherwise False.
    """
    wallet_path = os.path.join(base_path, wallet_name)

    # Check if wallet directory exists
    if not os.path.isdir(wallet_path):
        return False, f"Wallet directory {wallet_name} not found in {base_path}"

    # Check if coldkey file exists
    coldkey_path = os.path.join(wallet_path, "coldkey")
    if not os.path.isfile(coldkey_path):
        return False, f"Coldkey file not found in {wallet_name}"

    # Check if coldkeypub exists
    if coldkeypub_name:
        coldkeypub_path = os.path.join(wallet_path, coldkeypub_name)
        if not os.path.isfile(coldkeypub_path):
            return False, f"Coldkeypub file not found in {wallet_name}"

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


def verify_key_pattern(output: str, wallet_name: str) -> Optional[str]:
    """
    Verifies that a specific wallet key pattern exists in the output text.

    Args:
        output (str): The string output where the wallet key should be verified.
        wallet_name (str): The name of the wallet to search for in the output.

    Raises:
        AssertionError: If the wallet key pattern is not found, or if the key does not
                        start with '5', or if the key is not exactly 48 characters long.
    """
    split_output = output.splitlines()
    pattern = rf"{wallet_name}\s*\((5[A-Za-z0-9]{{47}})\)"
    found = False

    # Traverse each line to find instance of the pattern
    for line in split_output:
        match = re.search(pattern, line)
        if match:
            # Assert key starts with '5'
            assert match.group(1).startswith(
                "5"
            ), f"{wallet_name} should start with '5'"
            # Assert length of key is 48 characters
            assert (
                len(match.group(1)) == 48
            ), f"Key for {wallet_name} should be 48 characters long"
            found = True
            return match.group(1)

    # If no match is found in any line, raise an assertion error
    assert found, f"{wallet_name} not found in wallet list"
    return None


def extract_ss58_address(output: str, wallet_name: str) -> str:
    """
    Extracts the ss58 address from the given output for a specified wallet.

    Args:
        output (str): The captured output.
        wallet_name (str): The name of the wallet.

    Returns:
        str: ss58 address.
    """
    pattern = rf"{wallet_name}\s*\((5[A-Za-z0-9]{{47}})\)"
    lines = output.splitlines()
    for line in lines:
        match = re.search(pattern, line)
        if match:
            return match.group(1)  # Return the ss58 address

    raise ValueError(f"ss58 address not found for wallet {wallet_name}")


def extract_mnemonics_from_commands(output: str) -> Dict[str, Optional[str]]:
    """
    Extracts mnemonics of coldkeys & hotkeys from the given output for a specified wallet.

    Args:
        output (str): The captured output.

    Returns:
        dict: A dictionary keys 'coldkey' and 'hotkey', each containing their mnemonics.
    """
    mnemonics: Dict[str, Optional[str]] = {"coldkey": None, "hotkey": None}
    lines = output.splitlines()

    # Regex pattern to capture the mnemonic
    pattern = re.compile(r"btcli w regen_(coldkey|hotkey) --mnemonic ([a-z ]+)")

    for line in lines:
        line = line.strip().lower()
        match = pattern.search(line)
        if match:
            key_type = match.group(1)  # 'coldkey' or 'hotkey'
            mnemonic_phrase = match.group(2).strip()
            mnemonics[key_type] = mnemonic_phrase

    return mnemonics


def test_wallet_creations(local_chain: subtensor, capsys):
    """
    Test the creation and verification of wallet keys and directories in the Bittensor network.

    Steps:
        1. List existing wallets and verify the default setup.
        2. Create a new wallet with both coldkey and hotkey, verify their presence in the output,
           and check their physical existence.
        3. Create a new coldkey and verify both its display in the command line output and its physical file.
        4. Create a new hotkey for an existing coldkey, verify its display in the command line output,
           and check for both coldkey and hotkey files.

    Raises:
        AssertionError: If any of the checks or verifications fail
    """

    logging.info("Testing test_wallet_creations (create, new_hotkey, new_coldkey)")
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
    assert (
        "default" and "└── default" in captured.out
    ), "Default wallet not found in wallet list"
    wallet_status, message = verify_wallet_dir(
        base_path, "default", hotkey_name="default"
    )
    assert wallet_status, message

    # -----------------------------
    # Command 1: <btcli w create>
    # -----------------------------
    # Create a new wallet (coldkey + hotkey)
    logging.info("Testing wallet create command")
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

    captured = capsys.readouterr()

    # List the wallets
    exec_command(
        ListCommand,
        [
            "wallet",
            "list",
        ],
    )

    captured = capsys.readouterr()

    # Verify coldkey "new_wallet" is displayed with key
    verify_key_pattern(captured.out, "new_wallet")

    # Verify hotkey "new_hotkey" is displayed with key
    verify_key_pattern(captured.out, "new_hotkey")

    # Physically verify "new_wallet" and "new_hotkey" are present
    wallet_status, message = verify_wallet_dir(
        base_path, "new_wallet", hotkey_name="new_hotkey"
    )
    assert wallet_status, message

    # -----------------------------
    # Command 2: <btcli w new_coldkey>
    # -----------------------------
    # Create a new wallet (coldkey)
    logging.info("Testing wallet new_coldkey command")
    exec_command(
        NewColdkeyCommand,
        [
            "wallet",
            "new_coldkey",
            "--wallet.name",
            "new_coldkey",
            "--no_password",
            "--no_prompt",
            "--overwrite_coldkey",
            "--wallet.path",
            base_path,
        ],
    )

    captured = capsys.readouterr()

    # List the wallets
    exec_command(
        ListCommand,
        [
            "wallet",
            "list",
        ],
    )

    captured = capsys.readouterr()

    # Verify coldkey "new_coldkey" is displayed with key
    verify_key_pattern(captured.out, "new_coldkey")

    # Physically verify "new_coldkey" is present
    wallet_status, message = verify_wallet_dir(base_path, "new_coldkey")
    assert wallet_status, message

    # -----------------------------
    # Command 3: <btcli w new_hotkey>
    # -----------------------------
    # Create a new hotkey for alice_new_coldkey wallet
    logging.info("Testing wallet new_hotkey command")
    exec_command(
        NewHotkeyCommand,
        [
            "wallet",
            "new_hotkey",
            "--wallet.name",
            "new_coldkey",
            "--wallet.hotkey",
            "new_hotkey",
            "--no_prompt",
            "--overwrite_hotkey",
            "--wallet.path",
            base_path,
        ],
    )

    captured = capsys.readouterr()

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
    verify_key_pattern(captured.out, "new_hotkey")

    # Physically verify "alice_new_coldkey" and "alice_new_hotkey" are present
    wallet_status, message = verify_wallet_dir(
        base_path, "new_coldkey", hotkey_name="new_hotkey"
    )
    assert wallet_status, message
    logging.info("Passed test_wallet_creations")


def test_wallet_regen(local_chain: subtensor, capsys):
    """
    Test the regeneration of coldkeys, hotkeys, and coldkeypub files using mnemonics or ss58 address.

    Steps:
        1. List existing wallets and verify the default setup.
        2. Regenerate the coldkey using the mnemonics and verify using mod time.
        3. Regenerate the coldkeypub using ss58 address and verify using mod time
        4. Regenerate the hotkey using mnemonics and verify using mod time.

    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    logging.info(
        "Testing test_wallet_regen (regen_coldkey, regen_hotkey, regen_coldkeypub)"
    )
    wallet_path_name = "//Bob"
    base_path = f"/tmp/btcli-e2e-wallet-{wallet_path_name.strip('/')}"
    keypair, exec_command, wallet = setup_wallet(wallet_path_name)

    # Create a new wallet (coldkey + hotkey)
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

    captured = capsys.readouterr()
    mnemonics = extract_mnemonics_from_commands(captured.out)

    wallet_status, message = verify_wallet_dir(
        base_path,
        "new_wallet",
        hotkey_name="new_hotkey",
        coldkeypub_name="coldkeypub.txt",
    )
    assert wallet_status, message  # Ensure wallet exists

    # -----------------------------
    # Command 1: <btcli w regen_coldkey>
    # -----------------------------

    logging.info("Testing w regen_coldkey")
    coldkey_path = os.path.join(base_path, "new_wallet", "coldkey")
    initial_coldkey_mod_time = os.path.getmtime(coldkey_path)

    exec_command(
        RegenColdkeyCommand,
        [
            "wallet",
            "regen_coldkey",
            "--wallet.name",
            "new_wallet",
            "--wallet.path",
            base_path,
            "--no_prompt",
            "--overwrite_coldkey",
            "--mnemonic",
            mnemonics["coldkey"],
            "--no_password",
        ],
    )

    # Wait a bit to ensure file system updates modification time
    time.sleep(1)

    new_coldkey_mod_time = os.path.getmtime(coldkey_path)

    assert (
        initial_coldkey_mod_time != new_coldkey_mod_time
    ), "Coldkey file was not regenerated as expected"

    # -----------------------------
    # Command 2: <btcli w regen_coldkeypub>
    # -----------------------------

    logging.info("Testing w regen_coldkeypub")
    coldkeypub_path = os.path.join(base_path, "new_wallet", "coldkeypub.txt")
    initial_coldkeypub_mod_time = os.path.getmtime(coldkeypub_path)

    # List the wallets
    exec_command(
        ListCommand,
        [
            "wallet",
            "list",
        ],
    )
    captured = capsys.readouterr()
    ss58_address = extract_ss58_address(captured.out, "new_wallet")

    exec_command(
        RegenColdkeypubCommand,
        [
            "wallet",
            "regen_coldkeypub",
            "--wallet.name",
            "new_wallet",
            "--wallet.path",
            base_path,
            "--no_prompt",
            "--overwrite_coldkeypub",
            "--ss58_address",
            ss58_address,
        ],
    )

    # Wait a bit to ensure file system updates modification time
    time.sleep(1)

    new_coldkeypub_mod_time = os.path.getmtime(coldkeypub_path)

    assert (
        initial_coldkeypub_mod_time != new_coldkeypub_mod_time
    ), "Coldkeypub file was not regenerated as expected"

    # -----------------------------
    # Command 3: <btcli w regen_hotkey>
    # -----------------------------

    logging.info("Testing w regen_hotkey")
    hotkey_path = os.path.join(base_path, "new_wallet", "hotkeys", "new_hotkey")
    initial_hotkey_mod_time = os.path.getmtime(hotkey_path)

    exec_command(
        RegenHotkeyCommand,
        [
            "wallet",
            "regen_hotkey",
            "--no_prompt",
            "--overwrite_hotkey",
            "--wallet.name",
            "new_wallet",
            "--wallet.hotkey",
            "new_hotkey",
            "--wallet.path",
            base_path,
            "--mnemonic",
            mnemonics["hotkey"],
        ],
    )

    # Wait a bit to ensure file system updates modification time
    time.sleep(1)

    new_hotkey_mod_time = os.path.getmtime(hotkey_path)

    assert (
        initial_hotkey_mod_time != new_hotkey_mod_time
    ), "Hotkey file was not regenerated as expected"

    logging.info("Passed test_wallet_regen")
