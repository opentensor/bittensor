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

from loguru import logger
logger = logger.opt(colors=True)

class CLI ( bittensor.executor.Executor ):
    def __init__(self, config: Munch, **kwargs):
        if config == None:
            config = CLI.default_config()
        CLI.check_config( config )
        self.config = config
        super(CLI, self).__init__( self.config, **kwargs )

    @staticmethod   
    def default_config () -> Munch:
        # Build top level parser.
        parser = argparse.ArgumentParser(description="Bittensor cli", usage="bittensor-cli <command> <command args>", add_help=True)
        parser._positionals.title = "commands"
        CLI.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args (parser: argparse.ArgumentParser):
        cmd_parsers = parser.add_subparsers(dest='command')

        overview_parser = cmd_parsers.add_parser(
            'overview', 
            help='''Show account overview.'''
        )
        overview_parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='''Turn on bittensor debugging information''')
        bittensor.executor.Executor.add_args( overview_parser )

        transfer_parser = cmd_parsers.add_parser(
            'transfer', 
            help='''Transfer Tao between accounts.'''
        )
        transfer_parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='''Turn on bittensor debugging information''')
        bittensor.executor.Executor.add_args( transfer_parser )

        unstake_parser = cmd_parsers.add_parser(
            'unstake', 
            help='''Unstake from hotkey accounts.'''
        )
        unstake_parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='''Turn on bittensor debugging information''')
        bittensor.executor.Executor.add_args( unstake_parser )

        stake_parser = cmd_parsers.add_parser(
            'stake', 
            help='''Stake to your hotkey accounts.'''
        )
        stake_parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='''Turn on bittensor debugging information''')
        bittensor.executor.Executor.add_args( stake_parser )

        regen_coldkey_parser = cmd_parsers.add_parser(
            'regen_coldkey',
            help='''Regenerates a coldkey from a passed mnemonic'''
        )
        regen_coldkey_parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='''Turn on bittensor debugging information''')
        bittensor.executor.Executor.add_args( regen_coldkey_parser )

        regen_hotkey_parser = cmd_parsers.add_parser(
            'regen_hotkey',
            help='''Regenerates a hotkey from a passed mnemonic'''
        )
        regen_hotkey_parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='''Turn on bittensor debugging information''')
        bittensor.executor.Executor.add_args( regen_hotkey_parser )

        new_coldkey_parser = cmd_parsers.add_parser(
            'new_coldkey', 
            help='''Creates a new hotkey (for running a miner) under the specified path. '''
        )
        new_coldkey_parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='''Turn on bittensor debugging information''')
        bittensor.executor.Executor.add_args( new_coldkey_parser )

        new_hotkey_parser = cmd_parsers.add_parser(
            'new_hotkey', 
            help='''Creates a new coldkey (for containing balance) under the specified path. '''
        )
        new_hotkey_parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='''Turn on bittensor debugging information''')
        bittensor.executor.Executor.add_args( new_hotkey_parser )

         
        # Fill arguments for the regen coldkey command.
        regen_coldkey_parser.add_argument(
            "--mnemonic", 
            required=True, 
            nargs="+", 
            help='Mnemonic used to regen your key i.e. horse cart dog ...'
        )
        regen_coldkey_parser.add_argument(
            '--use_password', 
            dest='use_password', 
            action='store_true', 
            help='''Set protect the generated bittensor key with a password.''',
            default=True,
        )
        regen_coldkey_parser.add_argument(
            '--no_password', 
            dest='use_password', 
            action='store_false', 
            help='''Set off protects the generated bittensor key with a password.''',
        )

        # Fill arguments for the regen hotkey command.
        regen_hotkey_parser.add_argument(
            "--mnemonic", 
            required=True, 
            nargs="+", 
            help='Mnemonic used to regen your key i.e. horse cart dog ...'
        )
        regen_hotkey_parser.add_argument(
            '--use_password', 
            dest='use_password', 
            action='store_true', 
            help='''Set protect the generated bittensor key with a password.''',
            default=True
        )
        regen_hotkey_parser.add_argument(
            '--no_password', 
            dest='use_password', 
            action='store_false', 
            help='''Set off protects the generated bittensor key with a password.'''
        )

        # Fill arguments for the new coldkey command.
        new_coldkey_parser.add_argument(
            '--n_words', 
            type=int, 
            choices=[12,15,18,21,24], 
            default=12, 
            help='''The number of words representing the mnemonic. i.e. horse cart dog ... x 24'''
        )
        new_coldkey_parser.add_argument(
            '--use_password', 
            dest='use_password', 
            action='store_true', 
            help='''Set protect the generated bittensor key with a password.''',
            default=True,
        )
        new_coldkey_parser.add_argument(
            '--no_password', 
            dest='use_password', 
            action='store_false', 
            help='''Set off protects the generated bittensor key with a password.'''
        )

        # Fill arguments for the new hotkey command.
        new_hotkey_parser.add_argument(
            '--n_words', 
            type=int, 
            choices=[12,15,18,21,24], 
            default=12, 
            help='''The number of words representing the mnemonic. i.e. horse cart dog ... x 24'''
        )
        new_hotkey_parser.add_argument(
            '--use_password', 
            dest='use_password', 
            action='store_true', 
            help='''Set protect the generated bittensor key with a password.''',
            default=False
        )
        new_hotkey_parser.add_argument(
            '--no_password', 
            dest='use_password', 
            action='store_false', 
            help='''Set off protects the generated bittensor key with a password.'''
        )


        # Fill arguments for unstake command. 
        unstake_parser.add_argument(
            '--all', 
            dest="unstake_all", 
            action='store_true'
        )
        unstake_parser.add_argument(
            '--uid', 
            dest="uid", 
            type=int, 
            required=False
        )
        unstake_parser.add_argument(
            '--amount', 
            dest="amount", 
            type=float, 
            required=False
        )

        # Fill arguments for stake command.
        stake_parser.add_argument(
            '--uid', 
            dest="uid", 
            type=int, 
            required=False
        )
        stake_parser.add_argument(
            '--amount', 
            dest="amount", 
            type=float, 
            required=False
        )

        # Fill arguments for transfer
        transfer_parser.add_argument(
            '--dest', 
            dest="dest", 
            type=str, 
            required=True
        )
        transfer_parser.add_argument(
            '--amount', 
            dest="amount", 
            type=float, 
            required=True
        )

        # Hack to print formatted help
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)
        
    @staticmethod   
    def check_config (config: Munch):
        if config.command == "transfer":
            if not config.dest:
                logger.critical("The --dest argument is required for this command")
                quit()
            if not config.amount:
                logger.critical("The --amount argument is required for this command")
                quit()
        elif config.command == "unstake":
            if not config.unstake_all:
                if config.uid is None:
                    logger.critical("The --uid argument is required for this command")
                    quit()
                if not config.amount:
                    logger.critical("The --amount argument is required for this command")
                    quit()
        elif config.command == "stake":
            if config.uid is None:
                logger.critical("The --uid argument is required for this command")
                quit()
            if config.amount is None:
                logger.critical("The --amount argument is required for this command")
                quit()

    def run_command(self):
        if self.config.debug: bittensor.__debug_on__ = True; logger.info('DEBUG is <green>ON</green>')
        else: logger.info('DEBUG is <red>OFF</red>')
        if self.config.command == "transfer":
            self.transfer( amount_tao=self.config.amount, destination=self.config.dest)
        elif self.config.command == "unstake":
            if self.config.unstake_all:
                self.unstake_all()
            else:
                self.unstake( amount_tao =self.config.amount, uid=self.config.uid )
        elif self.config.command == "stake":
            self.stake( amount_tao=self.config.amount, uid=self.config.uid )
        elif self.config.command == "overview":
            self.overview()
        elif self.config.command == "new_coldkey":
            self.create_new_coldkey( n_words=self.config.n_words, use_password=self.config.use_password )
        elif self.config.command == "new_hotkey":
            self.create_new_hotkey( n_words=self.config.n_words, use_password=self.config.use_password )
        elif self.config.command == "regen_coldkey":
            self.regenerate_coldkey( mnemonic=self.config.mnemonic, use_password=self.config.use_password )
        elif self.config.command == "regen_hotkey":
            self.regenerate_hotkey( mnemonic=self.config.mnemonic, use_password=self.config.use_password )
        else:
            logger.critical("The command {} not implemented".format( self.config.command ))
            quit()
