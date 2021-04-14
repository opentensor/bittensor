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

import sys
import argparse
import bittensor
from munch import Munch

class CLI:
    def __init__(self, config):
        if config == None:
            config = CLI.default_config()
        CLI.check_config( config )
        self.config = config
        self.executor = bittensor.executor.Executor( self.config )

    @staticmethod   
    def default_config () -> Munch:
        # Build top level parser.
        parser = argparse.ArgumentParser(description="Bittensor cli", usage="bittensor-cli <command> <command args>", add_help=True)
        parser.add_argument('--debug', dest='debug', action='store_true', help='''Turn on bittensor debugging information''')
        parser.set_defaults( debug=True )
        parser._positionals.title = "commands"
        CLI.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args (parser: argparse.ArgumentParser):
        cmd_parsers = parser.add_subparsers(dest='command')

        overview_parser = cmd_parsers.add_parser('overview', 
            help='''Show account overview.''')
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
        bittensor.executor.Executor.add_args( regen_coldkey_parser )

        # Fill arguments for the regen hotkey command.
        regen_hotkey_parser.add_argument("--mnemonic", required=True, nargs="+", 
            help='Mnemonic used to regen your key i.e. horse cart dog ...') 
        regen_hotkey_parser.add_argument('--use_password', dest='use_password', action='store_true', help='''Set protect the generated bittensor key with a password.''')
        regen_hotkey_parser.add_argument('--no_password', dest='use_password', action='store_false', help='''Set off protects the generated bittensor key with a password.''')
        regen_hotkey_parser.set_defaults( use_password=False )
        bittensor.executor.Executor.add_args( regen_hotkey_parser )

        # Fill arguments for the new coldkey command.
        new_coldkey_parser.add_argument('--n_words', type=int, choices=[12,15,18,21,24], default=12, 
            help='''The number of words representing the mnemonic. i.e. horse cart dog ... x 24''')
        new_coldkey_parser.add_argument('--use_password', dest='use_password', action='store_true', help='''Set protect the generated bittensor key with a password.''')
        new_coldkey_parser.add_argument('--no_password', dest='use_password', action='store_false', help='''Set off protects the generated bittensor key with a password.''')
        new_coldkey_parser.set_defaults(use_password=True)
        bittensor.executor.Executor.add_args(  new_coldkey_parser )

        # Fill arguments for the new hotkey command.
        new_hotkey_parser.add_argument('--n_words', type=int, choices=[12,15,18,21,24], default=12, 
            help='''The number of words representing the mnemonic. i.e. horse cart dog ... x 24''')
        new_hotkey_parser.add_argument('--use_password', dest='use_password', action='store_true', help='''Set protect the generated bittensor key with a password.''')
        new_hotkey_parser.add_argument('--no_password', dest='use_password', action='store_false', help='''Set off protects the generated bittensor key with a password.''')
        new_hotkey_parser.set_defaults( use_password=False )
        bittensor.executor.Executor.add_args(  new_hotkey_parser )

        # Fill arguments for the overview command
        bittensor.executor.Executor.add_args( overview_parser )

        # Fill arguments for unstake command. 
        unstake_parser.add_argument('--all', dest="unstake_all", action='store_true')
        unstake_parser.add_argument('--uid', dest="uid", type=int, required=False)
        unstake_parser.add_argument('--amount', dest="amount", type=float, required=False)
        bittensor.executor.Executor.add_args( unstake_parser )

        # Fill arguments for stake command.
        stake_parser.add_argument('--uid', dest="uid", type=int, required=False)
        stake_parser.add_argument('--amount', dest="amount", type=float, required=False)
        bittensor.executor.Executor.add_args(  stake_parser )

        # Fill arguments for transfer
        transfer_parser.add_argument('--dest', dest="dest", type=str, required=True)
        transfer_parser.add_argument('--amount', dest="amount", type=float, required=True)
        bittensor.executor.Executor.add_args( transfer_parser )

        # Hack to print formatted help
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)
        
    @staticmethod   
    def check_config (config: Munch):
        if config.command == "transfer":
            if not config.dest:
                bittensor.__logger__.critical("The --dest argument is required for this command")
                quit()
            if not config.amount:
                bittensor.__logger__.critical("The --amount argument is required for this command")
                quit()
        elif config.command == "unstake":
            if not config.unstake_all:
                if config.uid is None:
                    bittensor.__logger__.critical("The --uid argument is required for this command")
                    quit()
                if not config.amount:
                    bittensor.__logger__.critical("The --amount argument is required for this command")
                    quit()
        elif config.command == "stake":
            if config.uid is None:
                bittensor.__logger__.critical("The --uid argument is required for this command")
                quit()
            if config.amount is None:
                bittensor.__logger__.critical("The --amount argument is required for this command")
                quit()

    def run_command(self):
        bittensor.BITTENSOR_LOGGING_LEVEL = 'TRACE' if self.config.debug else 'SUCCESS'
        if self.config.command == "transfer":
            self.executor.transfer( amount_tao=self.config.amount, destination=self.config.dest)
        elif self.config.command == "unstake":
            if self.config.unstake_all:
                self.executor.unstake_all()
            else:
                self.executor.unstake( amount_tao =self.config.amount, uid=self.config.uid )
        elif self.config.command == "stake":
            self.executor.stake( amount_tao=self.config.amount, uid=self.config.uid )
        elif self.config.command == "overview":
            self.executor.overview()
        elif self.config.command == "new_coldkey":
            self.executor.create_new_coldkey( n_words=self.config.n_words, use_password=self.config.use_password )
        elif self.config.command == "new_hotkey":
            self.executor.create_new_hotkey( n_words=self.config.n_words, use_password=self.config.use_password )
        elif self.config.command == "regen_coldkey":
            self.executor.regenerate_coldkey( mnemonic=self.config.mnemonic, use_password=self.config.use_password )
        elif self.config.command == "regen_hotkey":
            self.executor.regenerate_hotkey( mnemonic=self.config.mnemonic, use_password=self.config.use_password )
        else:
            bittensor.__logger__.critical("The command {} not implemented".format( self.config.command ))
            quit()
