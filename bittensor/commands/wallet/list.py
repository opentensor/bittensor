# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import argparse
import bittensor
from rich import print
from rich.tree import Tree

console = bittensor.__console__


class ListCommand:
    """
    Executes the ``list`` command which enumerates all wallets and their respective hotkeys present in the user's Bittensor configuration directory.

    The command organizes the information in a tree structure, displaying each wallet along with the ``ss58`` addresses for the coldkey public key and any hotkeys associated with it.

    Optional arguments:
        - ``-p``, ``--path``: The path to the Bittensor configuration directory. Defaults to '~/.bittensor'.

    The output is presented in a hierarchical tree format, with each wallet as a root node,
    and any associated hotkeys as child nodes. The ``ss58`` address is displayed for each
    coldkey and hotkey that is not encrypted and exists on the device.

    Usage:
        Upon invocation, the command scans the wallet directory and prints a list of all wallets, indicating whether the public keys are available (``?`` denotes unavailable or encrypted keys).

    Example usage::

        btcli wallet list --path ~/.bittensor

    Note:
        This command is read-only and does not modify the filesystem or the network state. It is intended for use within the Bittensor CLI to provide a quick overview of the user's wallets.
    """

    @staticmethod
    def run(cli):
        r"""Lists wallets."""
        try:
            wallets = next(os.walk(os.path.expanduser(cli.config.wallet.path)))[1]
        except StopIteration:
            # No wallet files found.
            wallets = []
        ListCommand._run(cli, wallets)

    @staticmethod
    def _run(cli: "bittensor.cli", wallets, return_value=False):
        root = Tree("Wallets")
        for w_name in wallets:
            wallet_for_name = bittensor.wallet(path=cli.config.wallet.path, name=w_name)
            try:
                if (
                    wallet_for_name.coldkeypub_file.exists_on_device()
                    and not wallet_for_name.coldkeypub_file.is_encrypted()
                ):
                    coldkeypub_str = wallet_for_name.coldkeypub.ss58_address
                else:
                    coldkeypub_str = "?"
            except:
                coldkeypub_str = "?"

            wallet_tree = root.add(
                f"[bold red]Coldkey[/bold red] [name<[green]{w_name}[/green]>, as58_address<[green]{coldkeypub_str}[/green]>]"
            )
            hotkeys_path = os.path.join(cli.config.wallet.path, w_name, "hotkeys")
            try:
                hotkeys = next(os.walk(os.path.expanduser(hotkeys_path)))
                if len(hotkeys) > 1:
                    for h_name in hotkeys[2]:
                        hotkey_for_name = bittensor.wallet(
                            path=cli.config.wallet.path, name=w_name, hotkey=h_name
                        )
                        try:
                            if (
                                hotkey_for_name.hotkey_file.exists_on_device()
                                and not hotkey_for_name.hotkey_file.is_encrypted()
                            ):
                                hotkey_str = hotkey_for_name.hotkey.ss58_address
                            else:
                                hotkey_str = "?"
                        except:
                            hotkey_str = "?"
                        wallet_tree.add("[bold yellow]Hotkey[/bold yellow] [name<[green]{}[/green]>, as58_address<[green]{}[/green]>]\n".format(h_name, hotkey_str))
            except:
                continue

        if len(wallets) == 0:
            root.add("[bold red]No wallets found.")

        # Uses rich print to display the tree.
        if not return_value:
            print(root)
        else:
            return root

    @staticmethod
    def check_config(config: "bittensor.config"):
        pass

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_parser = parser.add_parser("list", help="""List wallets""")
        bittensor.wallet.add_args(list_parser)
        bittensor.subtensor.add_args(list_parser)

    @staticmethod
    def get_tree(cli):
        try:
            wallets = next(os.walk(os.path.expanduser(cli.config.wallet.path)))[1]
        except StopIteration:
            # No wallet files found.
            wallets = []
        return ListCommand._run(cli=cli, wallets=wallets, return_value=True)
