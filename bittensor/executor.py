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
import argparse
import sys
import os
import pandas as pd

from munch import Munch
from loguru import logger
from termcolor import colored
from prettytable import PrettyTable

import bittensor
from bittensor.utils.neurons import Neuron, Neurons
from bittensor.utils.balance import Balance

class Executor:

    def __init__(   
            self, 
            config: 'Munch' = None, 
            wallet: 'bittensor.wallet.Wallet' = None,
            subtensor: 'bittensor.subtensor.Subtensor' = None,
            metagraph: 'bittensor.metagraph.Metagraph' = None,
            **kwargs,
        ):
        r""" Initializes a new Metagraph chain interface.
            Args:
                config (:obj:`Munch`, `optional`): 
                    neuron.Neuron.config()
                wallet (:obj:`bittensor.wallet.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                subtensor (:obj:`bittensor.subtensor.Subtensor`, `optional`):
                    subtensor interface utility.
                metagraph (:obj:`bittensor.metagraph.Metagraph`, `optional`):
                    bittensor metagraph object.
        """
        if config == None:
            config = Executor.default_config()
        bittensor.config.Config.update_with_kwargs(config, kwargs) 
        Executor.check_config(config)
        self.config = config

        if wallet == None:
            wallet = bittensor.wallet.Wallet( self.config )
        self.wallet = wallet

        # Only load subtensor if we need it.
        if self.config.command in ["transfer", "unstake", "stake", "overview", "save_state"]:
            if subtensor == None:
                subtensor = bittensor.subtensor.Subtensor( self.config, self.wallet )
            self.subtensor = subtensor

        if self.config.command in ["overview", "save_state"]:
            if metagraph == None:
                metagraph = bittensor.metagraph.Metagraph( self.config, self.wallet, self.subtensor)
            self.metagraph = metagraph
            

    @staticmethod
    def default_config () -> Munch:
         # Build top level parser.
        parser = argparse.ArgumentParser(description="Bittensor cli", usage="bittensor-cli <command> <command args>", add_help=True)
        parser._positionals.title = "commands"
        parser.add_argument("--debug", default=False, help="Turn on debugging information", action="store_true")
        Executor.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args (parser: argparse.ArgumentParser):
        cmd_parsers = parser.add_subparsers(dest='command')

        overview_parser = cmd_parsers.add_parser('overview', 
            help='''Show account overview.''')
        save_state_parser = cmd_parsers.add_parser('save_state', 
            help='''Saves the metagraph state to a json file.''')
        transfer_parser = cmd_parsers.add_parser('transfer', 
            help='''Transfer Tao between accounts.''')

        unstake_parser = cmd_parsers.add_parser('unstake', 
            help='''Unstake from hotkey accounts.''')
        stake_parser = cmd_parsers.add_parser('stake', 
            help='''Stake to your hotkey accounts.''')

        regen_coldkey_parser = cmd_parsers.add_parser('regen_coldkey',
            help='''Regenerates a coldkey from a passed mnemonic''')
        regen_hotkey_parser = cmd_parsers.add_parser('regen_hotkey',
            help='''Regenerates a hotkey from a passed mnemonic''')

        new_coldkey_parser = cmd_parsers.add_parser('new_coldkey', 
            help='''Creates a new hotkey (for running a miner) under the specified path. ''')
        new_hotkey_parser = cmd_parsers.add_parser('new_hotkey', 
            help='''Creates a new coldkey (for containing balance) under the specified path. ''')
            
        # Fill arguments for the regen coldkey command.
        regen_coldkey_parser.add_argument("--mnemonic", required=True, nargs="+", 
            help='Mnemonic used to regen your key i.e. horse cart dog ...') 
        regen_coldkey_parser.add_argument('--use_password', dest='use_password', action='store_true', help='''Set protect the generated bittensor key with a password.''')
        regen_coldkey_parser.add_argument('--no_password', dest='use_password', action='store_false', help='''Set off protects the generated bittensor key with a password.''')
        regen_coldkey_parser.set_defaults(use_password=True)
        bittensor.wallet.Wallet.add_args( regen_coldkey_parser )

        # Fill arguments for the regen hotkey command.
        regen_hotkey_parser.add_argument("--mnemonic", required=True, nargs="+", 
            help='Mnemonic used to regen your key i.e. horse cart dog ...') 
        bittensor.wallet.Wallet.add_args( regen_hotkey_parser )

        # Fill arguments for the new coldkey command.
        new_coldkey_parser.add_argument('--n_words', type=int, choices=[12,15,18,21,24], default=12, 
            help='''The number of words representing the mnemonic. i.e. horse cart dog ... x 24''')
        new_coldkey_parser.add_argument('--use_password', dest='use_password', action='store_true', help='''Set protect the generated bittensor key with a password.''')
        new_coldkey_parser.add_argument('--no_password', dest='use_password', action='store_false', help='''Set off protects the generated bittensor key with a password.''')
        new_coldkey_parser.set_defaults(use_password=True)
        bittensor.wallet.Wallet.add_args( new_coldkey_parser )

        # Fill arguments for the new hotkey command.
        new_hotkey_parser.add_argument('--n_words', type=int, choices=[12,15,18,21,24], default=12, 
            help='''The number of words representing the mnemonic. i.e. horse cart dog ... x 24''')
        bittensor.wallet.Wallet.add_args( new_hotkey_parser )

        # Fill arguments for the overview command
        bittensor.subtensor.Subtensor.add_args( overview_parser )
        bittensor.metagraph.Metagraph.add_args( overview_parser )

        # Fill argument for the save_state command
        bittensor.subtensor.Subtensor.add_args( save_state_parser )
        bittensor.metagraph.Metagraph.add_args( save_state_parser )

        # Fill arguments for unstake command. 
        unstake_parser.add_argument('--all', dest="unstake_all", action='store_true')
        unstake_parser.add_argument('--uid', dest="uid", type=int, required=False)
        unstake_parser.add_argument('--amount', dest="amount", type=float, required=False)
        bittensor.wallet.Wallet.add_args( unstake_parser )
        bittensor.subtensor.Subtensor.add_args( unstake_parser )

        # Fill arguments for stake command.
        stake_parser.add_argument('--uid', dest="uid", type=int, required=False)
        stake_parser.add_argument('--amount', dest="amount", type=float, required=False)
        bittensor.wallet.Wallet.add_args( stake_parser )
        bittensor.subtensor.Subtensor.add_args( stake_parser )

        # Fill arguments for transfer
        transfer_parser.add_argument('--dest', dest="dest", type=str, required=True)
        transfer_parser.add_argument('--amount', dest="amount", type=float, required=True)
        bittensor.wallet.Wallet.add_args( transfer_parser )
        bittensor.subtensor.Subtensor.add_args( transfer_parser )

        # Hack to print formatted help
        if len(sys.argv) == 1:
    	    parser.print_help()
    	    sys.exit(0)
        
    @staticmethod   
    def check_config (config: Munch):
        if config.command == "transfer":
            if not config.dest:
                print(colored("The --dest argument is required for this command", 'red'))
                quit()
            if not config.amount:
                print(colored("The --amount argument is required for this command", 'red'))
                quit()
        elif config.command == "unstake":
            if not config.unstake_all:
                if config.uid is None:
                    print(colored("The --uid argument is required for this command", 'red'))
                    quit()
                if not config.amount:
                    print(colored("The --amount argument is required for this command", 'red'))
                    quit()
        elif config.command == "stake":
            if config.uid is None:
                print(colored("The --uid argument is required for this command", 'red'))
                quit()
            if config.amount is None:
                print(colored("The --amount argument is required for this command", 'red'))
                quit()

    def run_command(self):
        if self.config.command == "transfer":
            self.transfer()
        elif self.config.command == "unstake":
            if self.config.unstake_all:
                self.unstake_all()
            else:
                self.unstake()
        elif self.config.command == "stake":
            self.stake()
        elif self.config.command == "overview":
            self.overview()
        elif self.config.command == "save_state":
            self.save_state()
        elif self.config.command == "new_coldkey":
            self.create_new_coldkey()
        elif self.config.command == "new_hotkey":
            self.create_new_hotkey()
        elif self.config.command == "regen_coldkey":
            self.regenerate_coldkey()
        elif self.config.command == "regen_hotkey":
            self.regenerate_hotkey()
        else:
            print(colored("The command {} not implemented".format( self.config.command ), 'red'))
            quit()
            
    def regenerate_coldkey ( self ):
        r""" Regenerates a colkey under this wallet.
        """
        self.wallet.regenerate_coldkey( self.config.mnemonic, self.config.use_password )

    def regenerate_hotkey ( self ):
        r""" Regenerates a hotkey under this wallet.
        """
        self.wallet.regenerate_hotkey( self.config.mnemonic )

    def create_new_coldkey ( self ):
        r""" Creates a new coldkey under this wallet.
        """
        self.wallet.create_new_coldkey( self.config.n_words, self.config.use_password )   

    def create_new_hotkey ( self ):  
        r""" Creates a new hotkey under this wallet.
        """
        self.wallet.create_new_hotkey( self.config.n_words )  

    def _associated_neurons( self ) -> Neurons:
        r""" Returns a list of neurons associate with this wallet's coldkey.
        """
        print(colored("Retrieving all nodes associated with cold key : {}".format( self.wallet.coldkeypub ), 'white'))
        neurons = self.subtensor.neurons()
        neurons = Neurons.from_list( neurons )
        result = filter(lambda x : x.coldkey == self.wallet.coldkey.public_key, neurons )# These are the neurons associated with the provided cold key
        associated_neurons = Neurons(result)
        # Load stakes
        for neuron in associated_neurons:
            neuron.stake = self.subtensor.get_stake_for_uid(neuron.uid)
        return associated_neurons

    def overview ( self ): 
        r""" Prints an overview for the wallet's colkey.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        self.metagraph.sync()
        balance = self.subtensor.get_balance( self.wallet.coldkey.ss58_address )
        neurons = self._associated_neurons()

        print("BALANCE: %s : [%s]" % ( self.wallet.coldkey.ss58_address, balance ))
        print()
        print("--===[[ STAKES ]]===--")
        t = PrettyTable(["UID", "IP", "STAKE", "RANK", "INCENTIVE"])
        t.align = 'l'
        total_stake = 0.0
        for neuron in neurons:
            index = self.metagraph.state.index_for_uid[ neuron.uid ]
            stake = float(self.metagraph.S[index])
            rank = float(self.metagraph.R[index])
            incentive = float(self.metagraph.I[index])
            t.add_row([neuron.uid, neuron.ip, stake, rank, incentive])
            total_stake += neuron.stake.__float__()
        print(t.get_string())
        print("Total stake: ", total_stake)

    def save_state( self ):
        self.subtensor.connect()
        self.metagraph.sync()
        filepath = os.path.expanduser('~/.bittensor/metagraph-at-block{}.txt'.format(self.metagraph.block))
        print ('Saving metagraph.state to file: {}'.format( filepath ))
        self.metagraph.state.write_to_file( filepath )

    def unstake_all ( self ):
        r""" Unstaked from all hotkeys associated with this wallet's coldkey.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.wallet.assert_hotkey()
        self.subtensor.connect()
        neurons = self._associated_neurons()
        for neuron in neurons:
            neuron.stake = self.subtensor.get_stake_for_uid( neuron.uid )
            result = self.subtensor.unstake( neuron.stake, neuron.hotkey, wait_for_finalization = True, timeout = bittensor.__blocktime__ * 5)
            if result:
                print(colored("Unstaked: {} Tao from uid: {} to coldkey.pub: {}".format( neuron.stake, neuron.uid, self.wallet.coldkey.public_key ) , 'green'))
            else:
                print(colored("Unstaking transaction failed", 'red'))

    def unstake( self ):
        r""" Unstaked token of amount to from uid.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.wallet.assert_hotkey()
        self.subtensor.connect()
        amount = Balance.from_float( self.config.amount )
        neurons = self._associated_neurons()
        neuron = neurons.get_by_uid( self.config.uid )
        if not neuron:
            print(colored("Neuron with uid: {} is not associated with coldkey.pub: {}".format( self.config.uid, self.wallet.coldkey.public_key), 'red'))
            quit()

        neuron.stake = self.subtensor.get_stake_for_uid(neuron.uid)
        if amount > neuron.stake:
            print(colored("Neuron with uid: {} does not have enough stake ({}) to be able to unstake {}".format( self.config.uid, neuron.stake, amount), 'red'))
            quit()

        print(colored("Requesting unstake of {} rao for hotkey: {} to coldkey: {}".format(amount.rao, neuron.hotkey, self.wallet.coldkey.public_key), 'blue'))
        print(colored("Waiting for finalization...", 'white'))
        result = self.subtensor.unstake(amount, neuron.hotkey, wait_for_finalization = True, timeout = bittensor.__blocktime__ * 5)
        if result:
            print(colored("Unstaked:{} from uid:{} to coldkey.pub:{}".format(amount.tao, neuron.uid, self.wallet.coldkey.public_key), 'green'))
        else:
            print(colored("Unstaking transaction failed", 'red'))

    def stake( self ):
        r""" Stakes token of amount to hotkey uid.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.wallet.assert_hotkey()
        self.subtensor.connect()
        amount = Balance.from_float( self.config.amount )
        balance = self.subtensor.get_balance( self.wallet.coldkey.ss58_address )
        if balance < amount:
            print(colored("Not enough balance ({}) to stake {}".format(balance, amount), 'red'))
            quit()

        neurons = self._associated_neurons()
        neuron = neurons.get_by_uid( self.config.uid )
        if not neuron:
            print(colored("Neuron with uid: {} is not associated with coldkey.pub: {}".format(self.config.uid, self.wallet.coldkey.public_key), 'red'))
            quit()

        print(colored("Adding stake of {} rao from coldkey {} to hotkey {}".format(amount.rao, self.wallet.coldkey.public_key, neuron.hotkey), 'blue'))
        print(colored("Waiting for finalization...", 'white'))
        result = self.subtensor.add_stake( amount, neuron.hotkey, wait_for_finalization = True, timeout = bittensor.__blocktime__ * 5)
        if result:
            print(colored("Staked: {} Tao to uid: {} from coldkey.pub: {}".format(amount.tao, self.config.uid, self.wallet.coldkey.public_key), 'green'))
        else:
            print(colored("Stake transaction failed", 'red'))

    def transfer( self ):
        r""" Transfers token of amount to dest.
            
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.wallet.assert_hotkey()
        self.subtensor.connect()
        amount = Balance.from_float( self.config.amount )
        balance = self.subtensor.get_balance(self.wallet.coldkey.ss58_address)
        if balance < amount:
            print(colored("Not enough balance ({}) to transfer {}".format(balance, amount), 'red'))
            quit()

        print(colored("Requesting transfer of {}, from coldkey: {} to dest: {}".format(amount.rao, self.wallet.coldkey.public_key, self.config.dest), 'blue'))
        print(colored("Waiting for finalization...", 'white'))
        result = self.subtensor.transfer(self.config.dest, amount,  wait_for_finalization = True, timeout = bittensor.__blocktime__ * 5)
        if result:
            print(colored("Transfer finalized with amount: {} Tao to dest: {} from coldkey.pub: {}".format(amount.tao, self.config.dest, self.wallet.coldkey.public_key), 'green'))
        else:
            print(colored("Transfer failed", 'red'))
 

if __name__ == "__main__":
    # ---- Build and Run ----
    config = Executor.default_config(); 
    logger.info(bittensor.config.Config.toString(config))
    executor = Executor( config )
    executor.run_command()
