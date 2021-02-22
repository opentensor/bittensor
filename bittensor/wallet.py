
# The MIT License (MIT)
# Copyright © 2021 Opentensor.ai

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
import json
import os
import re
import stat

from munch import Munch
from loguru import logger
from termcolor import colored

import bittensor
from bittensor.utils.cli_utils import cli_utils
from bittensor.crypto import encrypt, is_encrypted, decrypt_data, KeyError
from bittensor.crypto.keyfiles import load_keypair_from_data, KeyFileError
from bittensor.crypto.keyfiles import KeyFileError, load_keypair_from_data

class Wallet():
    """
    Bittensor wallet maintenance class. Each wallet contains a coldkey and a hotkey. 
    The coldkey is the user's primary key for holding stake in their wallet
    and is the only way that users can access Tao. Coldkeys can hold tokens and should be encrypted on your device.
    The coldkey must be used to stake and unstake funds from a running node. The hotkey, on the other hand, is only used
    for suscribing and setting weights from running code. Hotkeys are linked to coldkeys through the metagraph. 
    """
    def __init__(self, config: Munch = None):
        if config == None:
            config = Wallet.build_config()
        self.config = config
        self._hotkey = None
        self._coldkey = None
        self._coldkeypub = None
        
    @staticmethod   
    def build_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Wallet.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        Wallet.check_config(config)
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        try:
            parser.add_argument('--wallet.name', required=False, default='default', 
                                    help='''The name of the wallet to unlock for running bittensor''')
            parser.add_argument('--wallet.hotkey', required=False, default='default', 
                                    help='''The name of hotkey used to running the miner.''')
            parser.add_argument('--wallet.path', required=False, default='~/.bittensor/wallets/', 
                                    help='''The path to your bittensor wallets''')
        except:
            pass

    @staticmethod   
    def check_config(config: Munch):
        pass

    @property
    def hotkey(self) -> bittensor.substrate.base.Keypair:
        if self._hotkey == None:
            self._load_hotkey()
        return self._hotkey

    @property
    def coldkey(self) -> bittensor.substrate.base.Keypair:
        if self._coldkey == None:
            self._load_coldkey( )
        return self._coldkey

    @property
    def coldkeypub(self) -> str:
        if self._coldkeypub == None:
            self._load_coldkeypub( )
        return self._coldkeypub

    @property
    def coldkeyfile(self) -> str:
        full_path = os.path.expanduser(self.config.wallet.path + "/" + self.config.wallet.name)
        return full_path + "/coldkey"

    @property
    def coldkeypubfile(self) -> str:
        full_path = os.path.expanduser(self.config.wallet.path + "/" + self.config.wallet.name)
        return full_path + "/coldkeypub.txt"

    @property
    def hotkeyfile(self) -> str:
        full_path = os.path.expanduser(self.config.wallet.path + "/" + self.config.wallet.name)
        return full_path + "/hotkeys/" + self.config.wallet.hotkey


    def _load_coldkeypub(self):
        keyfile = os.path.expanduser( self.coldkeypubfile )
        keyfile = os.path.expanduser(keyfile)

        if not os.path.isfile(keyfile):
            print(colored("coldkeypubfile  {} does not exist".format(keyfile), 'red'))
            raise KeyFileError

        if not os.path.isfile(keyfile):
            print(colored("coldkeypubfile  {} is not a file".format(keyfile), 'red'))
            raise KeyFileError

        if not os.access(keyfile, os.R_OK):
            print(colored("coldkeypubfile  {} is not readable".format(keyfile), 'red'))
            raise KeyFileError

        with open(keyfile, "r") as file:
            key = file.readline().strip()
            if not re.match("^0x[a-z0-9]{64}$", key):
                logger.error("Cold key pub file is corrupt")
                raise KeyFileError("Cold key pub file is corrupt")

        with open(keyfile, "r") as file:
            self._coldkeypub = file.readline().strip()

        print(colored("Loaded coldkey.pub: {}".format(self._coldkeypub), 'green'))

    def _load_hotkey(self):
        keyfile = os.path.expanduser( self.hotkeyfile )
        keyfile = os.path.expanduser(keyfile)

        if not os.path.isfile(keyfile):
            print(colored("hotkeyfile  {} does not exist".format(keyfile), 'red'))
            raise KeyFileError

        if not os.path.isfile(keyfile):
            print(colored("hotkeyfile  {} is not a file".format(keyfile), 'red'))
            raise KeyFileError

        if not os.access(keyfile, os.R_OK):
            print(colored("hotkeyfile  {} is not readable".format(keyfile), 'red'))
            raise KeyFileError

        try:
            with open(keyfile, 'rb') as file:
                data = file.read()
                if is_encrypted(data):
                    password = bittensor.utils.Cli.ask_password()
                    print("decrypting key... (this may take a few moments)")
                    data = decrypt_data(password, data)
                self._hotkey = load_keypair_from_data(data)
                print(colored("Loaded hotkey: {}".format(self._hotkey.public_key), 'green'))

        except KeyError:
            print(colored("Invalid password", 'red'))
            raise KeyError("Invalid password")

        except KeyFileError as e:
            print(colored("Keyfile corrupt", 'red'))
            raise KeyFileError("Keyfile corrupt")

    def _load_coldkey(self):
        keyfile = os.path.expanduser( self.coldkeyfile )
        keyfile = os.path.expanduser(keyfile)

        if not os.path.isfile(keyfile):
            print(colored("coldkeyfile  {} does not exist".format(keyfile), 'red'))
            raise KeyFileError

        if not os.path.isfile(keyfile):
            print(colored("coldkeyfile  {} is not a file".format(keyfile), 'red'))
            raise KeyFileError

        if not os.access(keyfile, os.R_OK):
            print(colored("coldkeyfile  {} is not readable".format(keyfile), 'red'))
            raise KeyFileError

        try:
            with open(keyfile, 'rb') as file:
                data = file.read()
                if is_encrypted(data):
                    password = bittensor.utils.Cli.ask_password()
                    print("decrypting key... (this may take a few moments)")
                    data = decrypt_data(password, data)
                self._coldkey = load_keypair_from_data(data)
                print(colored("Loaded coldkey: {}".format(self._coldkey.public_key), 'green'))

        except KeyError:
            print(colored("Invalid password", 'red'))
            raise KeyError("Invalid password")

        except KeyFileError as e:
            print(colored("Keyfile corrupt", 'red'))
            raise KeyFileError("Keyfile corrupt")

    @staticmethod
    def __is_world_readable(path):
        st = os.stat(path)
        return st.st_mode & stat.S_IROTH

    @staticmethod
    def __create_keypair() -> bittensor.substrate.base.Keypair:
        return bittensor.substrate.base.Keypair.create_from_mnemonic(bittensor.substrate.base.Keypair.generate_mnemonic())

    @staticmethod
    def __save_keypair(keypair : bittensor.substrate.Keypair, path : str):
        path = os.path.expanduser(path)
        with open(path, 'w') as file:
            json.dump(keypair.toDict(), file)
            file.close()
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)

    @staticmethod
    def __has_keypair(path):
        path = os.path.expanduser(path)
        return os.path.exists(path)

    def create_new_coldkey( self, n_words:int = 12, use_password: bool = True ):      
        # Create directory 
        dir_path = os.path.expanduser(self.config.wallet.path + "/" + self.config.wallet.name )
        if not os.path.exists( dir_path ):
            os.makedirs( dir_path )

        # Create Key
        self._coldkey = cli_utils.gen_new_key( n_words )
        cli_utils.display_mnemonic_msg( self._coldkey  )
        cli_utils.write_pubkey_to_text_file( self.coldkeyfile, self._coldkey.public_key )

        # Encrypt
        if use_password:
            password = cli_utils.input_password()
            print("Encrypting coldkey ... (this might take a few moments)")
            coldkey_json_data = json.dumps( self._coldkey.toDict() ).encode()
            coldkey_data = encrypt(coldkey_json_data, password)
            del coldkey_json_data
        else:
            coldkey_data = json.dumps(self._coldkey.toDict()).encode()

        # Save
        cli_utils.save_keys( self.coldkeyfile, coldkey_data )
        cli_utils.set_file_permissions( self.coldkeyfile )

    def create_new_hotkey( self, n_words:int = 12):  
        # Create directory 
        dir_path = os.path.expanduser(self.config.wallet.path + "/" + self.config.wallet.name + "/hotkeys/" )
        if not os.path.exists( dir_path ):
            os.makedirs( dir_path )

        # Create
        hotkey_path = cli_utils.validate_create_path( self.hotkeyfile )
        self._hotkey = cli_utils.gen_new_key( n_words )
        cli_utils.display_mnemonic_msg( self._hotkey )
        hotkey_data = json.dumps(self._hotkey.toDict()).encode()

        # Save
        cli_utils.save_keys( self.hotkeyfile, hotkey_data )
        cli_utils.set_file_permissions( self.hotkeyfile )

    def regenerate_coldkey( self, mnemonic: str, use_password: bool):
        # Create directory 
        dir_path = os.path.expanduser(self.config.wallet.path + "/" + self.config.wallet.name )
        if not os.path.exists( dir_path ):
            os.makedirs( dir_path )

        # Regenerate
        self._coldkey = cli_utils.validate_generate_mnemonic( mnemonic )
        cli_utils.write_pubkey_to_text_file(self.coldkeyfile, self._coldkey.public_key )
        
        # Encrypt
        if use_password:
            password = cli_utils.input_password()
            print("Encrypting key ... (this might take a few moments)")
            json_data = json.dumps( self._coldkey.toDict() ).encode()
            coldkey_data = encrypt(json_data, password)
            del json_data
        else:
            coldkey_data = json.dumps(self._coldkey.toDict()).encode()

        # Save
        cli_utils.save_keys( self.coldkeyfile, coldkey_data ) 
        cli_utils.set_file_permissions( self.coldkeyfile )

    def regenerate_hotkey( self, mnemonic: str ):
        # Create directory 
        dir_path = os.path.expanduser(self.config.wallet.path + "/" + self.config.wallet.name + "/hotkeys/" )
        if not os.path.exists( dir_path ):
            os.makedirs( dir_path )

        # Regenerate
        self._hotkey = cli_utils.validate_generate_mnemonic( mnemonic )
        
        # Save
        hotkey_data = json.dumps(self._hotkey.toDict()).encode()
        cli_utils.save_keys( self.hotkeyfile, hotkey_data )
        cli_utils.set_file_permissions( self.hotkeyfile )
