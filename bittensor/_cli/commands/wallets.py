
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
import bittensor
import os
import sys
from rich.prompt import Prompt
from typing import Optional

class RegenColdkeyCommand:
    def run ( cli ):
        r""" Creates a new coldkey under this wallet."""
        wallet = bittensor.wallet(config = cli.config)

        json_str: Optional[str] = None
        json_password: Optional[str] = None
        if cli.config.get('json'):
            file_name: str = cli.config.get('json')
            if not os.path.exists(file_name) or not os.path.isfile(file_name):
                raise ValueError('File {} does not exist'.format(file_name))
            with open(cli.config.get('json'), 'r') as f:
                json_str = f.read()
            
            # Password can be "", assume if None
            json_password = cli.config.get('json_password', "")

        wallet.regenerate_coldkey( mnemonic = cli.config.mnemonic, seed = cli.config.seed, json = (json_str, json_password), use_password = cli.config.use_password, overwrite = cli.config.overwrite_coldkey )
   
    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name  and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        
        if config.mnemonic == None and config.get( 'seed', d=None ) == None and config.get( 'json', d=None ) == None:
            prompt_answer = Prompt.ask("Enter mnemonic, seed, or json file location")
            if prompt_answer.startswith("0x"):
                config.seed = prompt_answer
            elif len(prompt_answer.split(" ")) > 1:
                config.mnemonic = prompt_answer
            else:
                config.json = prompt_answer

        if config.get( 'json', d=None ) and config.get( 'json_password', d=None ) == None:
            config.json_password = Prompt.ask("Enter json backup password", password=True)
    
    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        regen_coldkey_parser = parser.add_parser(
            'regen_coldkey',
            help='''Regenerates a coldkey from a passed value'''
        )
        regen_coldkey_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )
        regen_coldkey_parser.add_argument(
            "--mnemonic", 
            required=False, 
            nargs="+", 
            help='Mnemonic used to regen your key i.e. horse cart dog ...'
        )
        regen_coldkey_parser.add_argument(
            "--seed", 
            required=False,  
            default=None,
            help='Seed hex string used to regen your key i.e. 0x1234...'
        )
        regen_coldkey_parser.add_argument(
            "--json",
            required=False,
            default=None,
            help='''Path to a json file containing the encrypted key backup. (e.g. from PolkadotJS)'''
        )
        regen_coldkey_parser.add_argument(
            "--json_password",
            required=False,
            default=None,
            help='''Password to decrypt the json file.'''
        )
        regen_coldkey_parser.add_argument(
            '--use_password', 
            dest='use_password', 
            action='store_true', 
            help='''Set true to protect the generated bittensor key with a password.''',
            default=True,
        )
        regen_coldkey_parser.add_argument(
            '--no_password', 
            dest='use_password', 
            action='store_false', 
            help='''Set off protects the generated bittensor key with a password.''',
        )
        regen_coldkey_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        regen_coldkey_parser.add_argument(
            '--overwrite_coldkey',
            default=False,
            action='store_false',
            help='''Overwrite the old coldkey with the newly generated coldkey'''
        )
        bittensor.wallet.add_args( regen_coldkey_parser )
        bittensor.subtensor.add_args( regen_coldkey_parser )


class RegenColdkeypubCommand:
    def run ( cli ):
        r""" Creates a new coldkeypub under this wallet."""
        wallet = bittensor.wallet(config = cli.config)
        wallet.regenerate_coldkeypub( ss58_address=cli.config.get('ss58_address'), public_key=cli.config.get('public_key_hex'), overwrite = cli.config.overwrite_coldkeypub )
    
    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name  and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if config.ss58_address == None and config.public_key_hex == None:
            prompt_answer = Prompt.ask("Enter the ss58_address or the public key in hex")
            if prompt_answer.startswith("0x"):
                config.public_key_hex = prompt_answer
            else:
                config.ss58_address = prompt_answer
        if not bittensor.utils.is_valid_bittensor_address_or_public_key(address = config.ss58_address if config.ss58_address else config.public_key_hex):
            sys.exit(1)

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        regen_coldkeypub_parser = parser.add_parser(
            'regen_coldkeypub',
            help='''Regenerates a coldkeypub from the public part of the coldkey.'''
        )
        regen_coldkeypub_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )
        regen_coldkeypub_parser.add_argument(
            "--public_key",
            "--pubkey", 
            dest="public_key_hex",
            required=False,
            default=None, 
            type=str,
            help='The public key (in hex) of the coldkey to regen e.g. 0x1234 ...'
        )
        regen_coldkeypub_parser.add_argument(
            "--ss58_address", 
            "--addr",
            "--ss58",
            dest="ss58_address",
            required=False,  
            default=None,
            type=str,
            help='The ss58 address of the coldkey to regen e.g. 5ABCD ...'
        )
        regen_coldkeypub_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        regen_coldkeypub_parser.add_argument(
            '--overwrite_coldkeypub',
            default=False,
            action='store_true',
            help='''Overwrite the old coldkeypub file with the newly generated coldkeypub'''
        )
        bittensor.wallet.add_args( regen_coldkeypub_parser )
        bittensor.subtensor.add_args( regen_coldkeypub_parser )

class RegenHotkeyCommand:

    def run ( cli ):
        r""" Creates a new coldkey under this wallet."""
        wallet = bittensor.wallet(config = cli.config)

        json_str: Optional[str] = None
        json_password: Optional[str] = None
        if cli.config.get('json'):
            file_name: str = cli.config.get('json')
            if not os.path.exists(file_name) or not os.path.isfile(file_name):
                raise ValueError('File {} does not exist'.format(file_name))
            with open(cli.config.get('json'), 'r') as f:
                json_str = f.read()
            
            # Password can be "", assume if None
            json_password = cli.config.get('json_password', "")

        wallet.regenerate_hotkey( mnemonic = cli.config.mnemonic, seed=cli.config.seed, json = (json_str, json_password), use_password = cli.config.use_password, overwrite = cli.config.overwrite_hotkey)
    
    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name  and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)
        
        if config.mnemonic == None and config.get( 'seed', d=None ) == None and config.get( 'json', d=None ) == None:
            prompt_answer = Prompt.ask("Enter mnemonic, seed, or json file location")
            if prompt_answer.startswith("0x"):
                config.seed = prompt_answer
            elif len(prompt_answer.split(" ")) > 1:
                config.mnemonic = prompt_answer
            else:
                config.json = prompt_answer

        if config.get( 'json', d=None ) and config.get( 'json_password', d=None ) == None:
            config.json_password = Prompt.ask("Enter json backup password", password=True)

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        regen_hotkey_parser = parser.add_parser(
            'regen_hotkey',
            help='''Regenerates a hotkey from a passed mnemonic'''
        )
        regen_hotkey_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )
        regen_hotkey_parser.add_argument(
            "--mnemonic", 
            required=False, 
            nargs="+", 
            help='Mnemonic used to regen your key i.e. horse cart dog ...'
        )
        regen_hotkey_parser.add_argument(
            "--seed", 
            required=False,  
            default=None,
            help='Seed hex string used to regen your key i.e. 0x1234...'
        )
        regen_hotkey_parser.add_argument(
            "--json",
            required=False,
            default=None,
            help='''Path to a json file containing the encrypted key backup. (e.g. from PolkadotJS)'''
        )
        regen_hotkey_parser.add_argument(
            "--json_password",
            required=False,
            default=None,
            help='''Password to decrypt the json file.'''
        )
        regen_hotkey_parser.add_argument(
            '--use_password', 
            dest='use_password', 
            action='store_true', 
            help='''Set true to protect the generated bittensor key with a password.''',
            default=False
        )
        regen_hotkey_parser.add_argument(
            '--no_password', 
            dest='use_password', 
            action='store_false', 
            help='''Set off protects the generated bittensor key with a password.'''
        )
        regen_hotkey_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        regen_hotkey_parser.add_argument(
            '--overwrite_hotkey',
            dest='overwrite_hotkey',
            action='store_true',
            default=False,
            help='''Overwrite the old hotkey with the newly generated hotkey'''
        )
        bittensor.wallet.add_args( regen_hotkey_parser )
        bittensor.subtensor.add_args( regen_hotkey_parser )
    


class NewHotkeyCommand:

    def run( cli ):
        """ Creates a new hotke under this wallet."""
        wallet = bittensor.wallet(config = cli.config)
        wallet.create_new_hotkey( n_words = cli.config.n_words, use_password = cli.config.use_password, overwrite = cli.config.overwrite_hotkey)   

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name  and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        new_hotkey_parser = parser.add_parser( 'new_hotkey',  help='''Creates a new hotkey (for running a miner) under the specified path.''')
        new_hotkey_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )
        new_hotkey_parser.add_argument( '--n_words', 
            type=int, 
            choices=[12,15,18,21,24], 
            default=12, 
            help='''The number of words representing the mnemonic. i.e. horse cart dog ... x 24'''
        )
        new_hotkey_parser.add_argument(
            '--use_password', 
            dest='use_password', 
            action='store_true', 
            help='''Set true to protect the generated bittensor key with a password.''',
            default=False
        )
        new_hotkey_parser.add_argument(
            '--no_password', 
            dest='use_password', 
            action='store_false', 
            help='''Set off protects the generated bittensor key with a password.'''
        )
        new_hotkey_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        new_hotkey_parser.add_argument(
            '--overwrite_hotkey',
            action='store_false',
            default=False,
            help='''Overwrite the old hotkey with the newly generated hotkey'''
        )
        bittensor.wallet.add_args( new_hotkey_parser ) 
        bittensor.subtensor.add_args( new_hotkey_parser )


class NewColdkeyCommand:
    def run ( cli ):
        r""" Creates a new coldkey under this wallet."""
        wallet = bittensor.wallet(config = cli.config)
        wallet.create_new_coldkey( n_words = cli.config.n_words, use_password = cli.config.use_password, overwrite = cli.config.overwrite_coldkey)   

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name  and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        new_coldkey_parser = parser.add_parser(
            'new_coldkey', 
            help='''Creates a new coldkey (for containing balance) under the specified path. '''
        )
        new_coldkey_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )
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
            help='''Set true to protect the generated bittensor key with a password.''',
            default=True,
        )
        new_coldkey_parser.add_argument(
            '--no_password', 
            dest='use_password', 
            action='store_false', 
            help='''Set off protects the generated bittensor key with a password.'''
        )
        new_coldkey_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        new_coldkey_parser.add_argument(
            '--overwrite_coldkey',
            action='store_false',
            default=False,
            help='''Overwrite the old coldkey with the newly generated coldkey'''
        )
        bittensor.wallet.add_args( new_coldkey_parser )
        bittensor.subtensor.add_args( new_coldkey_parser )