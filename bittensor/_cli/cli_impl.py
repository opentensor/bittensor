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

import bittensor

import os
import sys
import time
import json
import requests
import importlib
from pathlib import Path
from rich.tree import Tree
from rich import print
from tqdm import tqdm
from rich.table import Table
from rich.prompt import Confirm
from rich.prompt import Prompt

class CLI:
    """
    Implementation of the CLI class, which handles the coldkey, hotkey and money transfer 
    """
    def __init__(self, config: 'bittensor.Config' ):
        r""" Initialized a bittensor.CLI object.
            Args:
                config (:obj:`bittensor.Config`, `required`): 
                    bittensor.cli.config()
        """
        self.config = config

    def run ( self ):
        """ Execute the command from config 
        """
        if self.config.command == "run":
            self.run_miner ()
        elif self.config.command == "transfer":
            self.transfer ()
        elif self.config.command == "register":
            self.register()
        elif self.config.command == "unstake":
            self.unstake()
        elif self.config.command == "stake":
            self.stake()
        elif self.config.command == "overview":
            self.overview()
        elif self.config.command == "list":
            self.list()
        elif self.config.command == "new_coldkey":
            self.create_new_coldkey()
        elif self.config.command == "new_hotkey":
            self.create_new_hotkey()
        elif self.config.command == "regen_coldkey":
            self.regen_coldkey()
        elif self.config.command == "regen_hotkey":
            self.regen_hotkey()
        elif self.config.command == "metagraph":
            self.metagraph()

    def create_new_coldkey ( self ):
        r""" Creates a new coldkey under this wallet.
        """
        wallet = bittensor.wallet(config = self.config)
        wallet.create_new_coldkey( n_words = self.config.n_words, use_password = self.config.use_password, overwrite = False)   

    def create_new_hotkey ( self ):
        r""" Creates a new hotke under this wallet.
        """
        wallet = bittensor.wallet(config = self.config)
        wallet.create_new_hotkey( n_words = self.config.n_words, use_password = self.config.use_password, overwrite = False)   

    def regen_coldkey ( self ):
        r""" Creates a new coldkey under this wallet.
        """
        wallet = bittensor.wallet(config = self.config)
        wallet.regenerate_coldkey( mnemonic = self.config.mnemonic, use_password = self.config.use_password, overwrite = False )

    def regen_hotkey ( self ):
        r""" Creates a new coldkey under this wallet.
        """
        wallet = bittensor.wallet(config = self.config)
        wallet.regenerate_hotkey( mnemonic = self.config.mnemonic, use_password = self.config.use_password, overwrite = False)

    def run_miner ( self ):

        # Check coldkey.
        wallet = bittensor.wallet( config = self.config )
        if not wallet.coldkeypub_file.exists_on_device():
            if Confirm.ask("Coldkey: [bold]'{}'[/bold] does not exist, do you want to create it".format(self.config.wallet.name)):
                wallet.create_new_coldkey()
            else:
                sys.exit()

        # Check hotkey.
        if not wallet.hotkey_file.exists_on_device():
            if Confirm.ask("Hotkey: [bold]'{}'[/bold] does not exist, do you want to create it".format(self.config.wallet.hotkey)):
                wallet.create_new_hotkey()
            else:
                sys.exit()

        if wallet.hotkey_file.is_encrypted():
            bittensor.__console__.print("Decrypting hotkey ... ")
        wallet.hotkey

        if wallet.coldkeypub_file.is_encrypted():
            bittensor.__console__.print("Decrypting coldkeypub ... ")
        wallet.coldkeypub

        # Check registration
        self.register()

        # Run miner.
        file_path = Path(str(Path(__file__).resolve()) + "/../../../miners/text/template_miner.py").resolve()
        miner = importlib.machinery.SourceFileLoader('Miner',os.path.expanduser(file_path)).load_module()
        miner.Miner( config = self.config ).run()
        bittensor.__console__.clear()

    def register( self ):
        r""" Register 
        """
        wallet = bittensor.wallet( config = self.config )
        subtensor = bittensor.subtensor( config = self.config )
        subtensor.register( wallet = wallet, prompt = not self.config.no_prompt)

    def transfer( self ):
        r""" Transfers token of amount to destination.
        """
        wallet = bittensor.wallet( config = self.config )
        subtensor = bittensor.subtensor( config = self.config )
        subtensor.transfer( wallet = wallet, dest = self.config.dest, amount = self.config.amount, wait_for_finalization = True, prompt = not self.config.no_prompt )

    def unstake( self ):
        r""" Unstaked token of amount from uid.
        """
        wallet = bittensor.wallet( config = self.config )
        subtensor = bittensor.subtensor( config = self.config )
        subtensor.unstake( wallet, amount = None if self.config.unstake_all else self.config.amount, wait_for_finalization = True, prompt = not self.config.no_prompt )

    def stake( self ):
        r""" Staked token of amount to uid.
        """
        console = bittensor.__console__
        wallet = bittensor.wallet( config = self.config )
        subtensor = bittensor.subtensor( config = self.config )
        subtensor.add_stake( wallet, amount = None if self.config.stake_all else self.config.amount, wait_for_finalization = True, prompt = not self.config.no_prompt )

    def list(self):
        r""" Lists wallets.
        """
        wallets = next(os.walk(os.path.expanduser(self.config.wallet.path)))[1]
        root = Tree("Wallets")
        for w_name in wallets:
            wallet_for_name = bittensor.wallet( path = self.config.wallet.path, name = w_name)
            try:
                if wallet_for_name.coldkeypub_file.exists_on_device() and not wallet_for_name.coldkeypub_file.is_encrypted():
                    coldkeypub_str = wallet_for_name.coldkeypub.ss58_address
                else:
                    coldkeypub_str = '?'
            except:
                coldkeypub_str = '?'


            wallet_tree = root.add("\n\n[bold white]{} - {}".format(w_name, coldkeypub_str))
            hotkeys_path = self.config.wallet.path + w_name + '/hotkeys'
            try:
                hotkeys = next(os.walk(os.path.expanduser(hotkeys_path)))
                if len( hotkeys ) > 1:
                    for h_name in hotkeys[2]:
                        hotkey_for_name = bittensor.wallet( path = self.config.wallet.path, name = w_name, hotkey = h_name)
                        try:
                            if hotkey_for_name.hotkey_file.exists_on_device() and not hotkey_for_name.hotkey_file.is_encrypted():
                                hotkey_str = wallet_for_name.hotkey.ss58_address
                            else:
                                hotkey_str = '?'
                        except:
                            hotkey_str = '?'

                        wallet_tree.add("[bold grey]{} - {}".format(h_name, hotkey_str))
            except:
                pass

        print(root)

    def metagraph(self):
        r""" Prints an overview for the wallet's colkey.
        """
        console = bittensor.__console__
        subtensor = bittensor.subtensor( config = self.config )
        metagraph = bittensor.metagraph( subtensor = subtensor )
        with console.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(self.config.subtensor.network)):
            metagraph.load()
            metagraph.sync()
            metagraph.save()

        TABLE_DATA = [] 
        total_stake = 0.0
        total_rank = 0.0
        total_trust = 0.0
        total_consensus = 0.0
        total_incentive = 0.0
        total_dividends = 0.0
        total_emission = 0.0  
        for uid in metagraph.uids:
            ep = metagraph.endpoint_objs[uid]
            row = [
                str(ep.uid), 
                '{:.5f}'.format( metagraph.stake[uid]),
                '{:.5f}'.format( metagraph.ranks[uid]), 
                '{:.5f}'.format( metagraph.trust[uid]), 
                '{:.5f}'.format( metagraph.consensus[uid]), 
                '{:.5f}'.format( metagraph.incentive[uid]),
                '{:.5f}'.format( metagraph.dividends[uid]),
                '{:.5f}'.format( metagraph.emission[uid]),
                str(metagraph.last_update[uid].item()),
                str( metagraph.active[uid].item() ), 
                ep.ip + ':' + str(ep.port) if ep.is_serving else '[yellow]none[/yellow]', 
                ep.hotkey[:10],
                ep.coldkey[:10]
            ]
            total_stake += metagraph.stake[uid]
            total_rank += metagraph.ranks[uid]
            total_trust += metagraph.trust[uid]
            total_consensus += metagraph.consensus[uid]
            total_incentive += metagraph.incentive[uid]
            total_dividends += metagraph.dividends[uid]
            total_emission += metagraph.emission[uid]
            TABLE_DATA.append(row)
        total_neurons = len(metagraph.uids)                
        table = Table(show_footer=False)
        table.title = (
            "[white]Metagraph([bold grey]{}[/bold grey])".format(subtensor.network)
        )
        table.add_column("[overline white]UID",  str(total_neurons), footer_style = "overline white", style='yellow')
        table.add_column("[overline white]STAKE", '{:.5f}'.format(total_stake), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]RANK", '{:.5f}'.format(total_rank), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]TRUST", '{:.5f}'.format(total_trust), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]CONSENSUS", '{:.5f}'.format(total_consensus), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]INCENTIVE", '{:.5f}'.format(total_incentive), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]DIVIDENDS", '{:.5f}'.format(total_dividends), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]EMISSION", '{:.5f}'.format(total_emission), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]LastUpdate (blocks)", justify='right', no_wrap=True)
        table.add_column("[overline white]ACTIVE", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]AXON", justify='left', style='dim blue', no_wrap=True) 
        table.add_column("[overline white]HOTKEY", style='dim blue', no_wrap=False)
        table.add_column("[overline white]COLDKEY", style='dim purple', no_wrap=False)
        table.show_footer = True

        for row in TABLE_DATA:
            table.add_row(*row)
        table.box = None
        table.pad_edge = False
        table.width = None
        console.print(table)

    def overview(self):
        r""" Prints an overview for the wallet's colkey.
        """
        console = bittensor.__console__
        wallet = bittensor.wallet( config = self.config )
        subtensor = bittensor.subtensor( config = self.config )
        metagraph = bittensor.metagraph( subtensor = subtensor )
        with console.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(self.config.subtensor.network)):
            metagraph.load()
            metagraph.sync()
            metagraph.save()
            balance = subtensor.get_balance( wallet.coldkeypub.ss58_address )

        owned_endpoints = [] 
        endpoints = metagraph.endpoint_objs
        for uid, cold in enumerate(metagraph.coldkeys):
            if cold == wallet.coldkeypub.ss58_address:
                owned_endpoints.append( endpoints[uid] )

        TABLE_DATA = []  
        total_stake = 0.0
        total_rank = 0.0
        total_trust = 0.0
        total_consensus = 0.0
        total_incentive = 0.0
        total_dividends = 0.0
        total_emission = 0.0      
        for ep in tqdm(owned_endpoints):
            uid = ep.uid
            active = metagraph.active[ uid ].item()
            stake = metagraph.S[ uid ].item()
            rank = metagraph.R[ uid ].item()
            trust = metagraph.T[ uid ].item()
            consensus = metagraph.C[ uid ].item()
            incentive = metagraph.I[ uid ].item()
            dividends = metagraph.D[ uid ].item()
            emission = metagraph.E[ uid ].item() / 1000000000
            last_update = int(metagraph.block - metagraph.last_update[uid ])
            row = [
                str(ep.uid), 
                str(active), 
                '{:.5f}'.format(stake),
                '{:.5f}'.format(rank), 
                '{:.5f}'.format(trust), 
                '{:.5f}'.format(consensus), 
                '{:.5f}'.format(incentive),
                '{:.5f}'.format(dividends),
                '{:.5f}'.format(emission),
                str(last_update),
                ep.ip + ':' + str(ep.port) if ep.is_serving else '[yellow]none[/yellow]', 
                ep.hotkey
            ]
            total_stake += stake
            total_rank += rank
            total_trust += trust
            total_consensus += consensus
            total_incentive += incentive
            total_dividends += dividends
            total_emission += emission
            TABLE_DATA.append(row)
            
        total_neurons = len(owned_endpoints)                
        table = Table(show_footer=False)
        table.title = (
            "[white]Wallet - {}:{}".format(self.config.wallet.name, wallet.coldkeypub.ss58_address)
        )
        table.add_column("[overline white]UID",  str(total_neurons), footer_style = "overline white", style='yellow')
        table.add_column("[overline white]ACTIVE", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]STAKE", '{:.5f}'.format(total_stake), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]RANK", '{:.5f}'.format(total_rank), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]TRUST", '{:.5f}'.format(total_trust), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]CONSENSUS", '{:.5f}'.format(total_consensus), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]INCENTIVE", '{:.5f}'.format(total_incentive), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]DIVIDENDS", '{:.5f}'.format(total_dividends), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]EMISSION", '{:.5f}'.format(total_emission), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]LastUpdate (blocks)", justify='right', no_wrap=True)
        table.add_column("[overline white]AXON", justify='left', style='dim blue', no_wrap=True) 
        table.add_column("[overline white]HOTKEY", style='dim blue', no_wrap=False)
        table.show_footer = True
        table.caption = "[white]Wallet balance: [green]\u03C4" + str(balance.tao)

        console.clear()
        for row in TABLE_DATA:
            table.add_row(*row)
        table.box = None
        table.pad_edge = False
        table.width = None
        console.print(table)