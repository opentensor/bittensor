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
    @staticmethod
    def run (cli):
        r""" Lists wallets."""
        try:
            wallets = next(os.walk(os.path.expanduser(cli.config.wallet.path)))[1]
        except StopIteration:
            # No wallet files found.
            wallets = []

        root = Tree("Wallets")
        for w_name in wallets:
            wallet_for_name = bittensor.wallet( path = cli.config.wallet.path, name = w_name)
            try:
                if wallet_for_name.coldkeypub_file.exists_on_device() and not wallet_for_name.coldkeypub_file.is_encrypted():
                    coldkeypub_str = wallet_for_name.coldkeypub.ss58_address
                else:
                    coldkeypub_str = '?'
            except:
                coldkeypub_str = '?'

            wallet_tree = root.add("\n[bold white]{} ({})".format(w_name, coldkeypub_str))
            hotkeys_path = os.path.join(cli.config.wallet.path, w_name, 'hotkeys')
            try:
                hotkeys = next(os.walk(os.path.expanduser(hotkeys_path)))
                if len( hotkeys ) > 1:
                    for h_name in hotkeys[2]:
                        hotkey_for_name = bittensor.wallet( path = cli.config.wallet.path, name = w_name, hotkey = h_name)
                        try:
                            if hotkey_for_name.hotkey_file.exists_on_device() and not hotkey_for_name.hotkey_file.is_encrypted():
                                hotkey_str = hotkey_for_name.hotkey.ss58_address
                            else:
                                hotkey_str = '?'
                        except:
                            hotkey_str = '?'
                        wallet_tree.add("[bold grey]{} ({})".format(h_name, hotkey_str))
            except:
                continue

        if len(wallets) == 0:
            root.add("[bold red]No wallets found.")

        # Uses rich print to display the tree.
        print(root)

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        pass

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        list_parser = parser.add_parser(
            'list', 
            help='''List wallets'''
        )
        list_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        list_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )
        bittensor.wallet.add_args( list_parser )
        bittensor.subtensor.add_args( list_parser )