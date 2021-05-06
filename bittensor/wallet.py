
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
import copy
import json
import os
import re
import stat
import sys

from munch import Munch
from loguru import logger
logger = logger.opt(colors=True)
from termcolor import colored

import bittensor
from bittensor.substrate import Keypair
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
    def __init__(
            self, 
            config: Munch = None,
            name: str = None,
            path: str = None,
            hotkey: str = None,
            **kwargs
        ):
        r""" Init bittensor wallet object containing a hot and coldkey.

            Args:
                config (:obj:`Munch`, `optional`): 
                name (required=False, default='default):
                    The name of the wallet to unlock for running bittensor
                hotkey (required=False, default='default):
                    The name of hotkey used to running the miner.
                path (required=False, default='~/.bittensor/wallets/'):
                    The path to your bittensor wallets
        """
        if config == None:
            config = Wallet.default_config()
        config = copy.deepcopy(config); bittensor.config.Config.update_with_kwargs(config, kwargs )
        config.wallet.name = name if name != None else config.wallet.name
        config.wallet.path = path if path != None else config.wallet.path
        config.wallet.hotkey = hotkey if hotkey != None else config.wallet.hotkey
        Wallet.check_config(config)
        self.config = config
        self._hotkey = None
        self._coldkey = None
        self._coldkeypub = None
        
    @staticmethod   
    def default_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Wallet.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        try:
            parser.add_argument('--wallet.name', required=False, default='default', 
                                    help='''The name of the wallet to unlock for running bittensor''')
            parser.add_argument('--wallet.hotkey', required=False, default='default', 
                                    help='''The name of wallet's hotkey.''')
            parser.add_argument('--wallet.path', required=False, default='~/.bittensor/wallets/', 
                                    help='''The path to your bittensor wallets''')
        except:
            pass

    @staticmethod   
    def check_config(config: Munch):
        pass

    def assert_hotkey(self):
        r""" Checks for a valid hotkey from wallet.path/wallet.name/hotkeys/wallet.hotkey or exits.
        """
        try:
            assert self.has_hotkey
        except:
            sys.exit(1)
        
    def assert_coldkey(self):
        r""" Checks for a valid coldkey from wallet.path/wallet.name/hotkeys/wallet.hotkey or exits.
        """
        try:
            assert self.has_coldkey
        except:
            sys.exit(1)

    def assert_coldkeypub(self):
        r""" Checks for a valid coldkeypub from wallet.path/wallet.name/coldkeypub.txt or exits
        """
        try:
            assert self.has_coldkeypub
        except:
            sys.exit(1)

    @property
    def has_hotkey(self) -> bool:
        r""" True if a hotkey can be loaded from wallet.path/wallet.name/hotkeys/wallet.hotkey or returns None.
            Returns:
                hotkey (bool):
                    True if the hotkey can be loaded from config arguments or False
        """
        try:
            self.hotkey
            return True
        except KeyFileError:
            return False
        except KeyError:
            return False
        except Exception:
            return False
        
    @property
    def has_coldkey(self) -> bool:
        r""" True if a coldkey can be loaded from wallet.path/wallet.name/coldkeypub.txt
            Returns:
                has_coldkey (bool):
                    True if the coldkey can be loaded from config arguments or False
        """
        try:
            self.coldkey
            return True
        except KeyFileError:
            return False
        except KeyError:
            return False
        except Exception:
            return False

    @property
    def has_coldkeypub(self) -> bool:
        r""" True if the coldkeypub can be loaded from wallet.path/wallet.name/coldkeypub.txt.
            Returns:
                has_coldkeypub (bool):
                    True if the coldkeypub can be loaded from config arguments or False
        """
        try:
            self.coldkeypub
            return True
        except KeyFileError:
            return False
        except KeyError:
            return False
        except Exception:
            return False

    @property
    def hotkey(self) -> bittensor.substrate.Keypair:
        r""" Loads the hotkey from wallet.path/wallet.name/hotkeys/wallet.hotkey or raises an error.
            Returns:
                hotkey (bittensor.substrate.Keypair):
                    hotkey loaded from config arguments.
            Raises:
                KeyFileError: Raised if the file is corrupt of non-existent.
                KeyError: Raised if the user enters an incorrec password for an encrypted keyfile.
        """
        if self._hotkey == None:
            self._hotkey = self._load_hotkey()
        return self._hotkey

    @property
    def coldkey(self) -> 'bittensor.substrate.Keypair':
        r""" Loads the hotkey from wallet.path/wallet.name/coldkey or raises an error.
            Returns:
                coldkey (bittensor.substrate.Keypair):
                    colkey loaded from config arguments.
            Raises:
                KeyFileError: Raised if the file is corrupt of non-existent.
                KeyError: Raised if the user enters an incorrec password for an encrypted keyfile.
        """
        if self._coldkey == None:
            self._coldkey = self._load_coldkey( )
        return self._coldkey

    @property
    def coldkeypub(self) -> str:
        r""" Loads the coldkeypub from wallet.path/wallet.name/coldkeypub.txt or raises an error.
            Returns:
                coldkeypub (str):
                    colkeypub loaded from config arguments.
            Raises:
                KeyFileError: Raised if the file is corrupt of non-existent.
                KeyError: Raised if the user enters an incorrec password for an encrypted keyfile.
        """
        if self._coldkeypub == None:
            self._coldkeypub = self._load_coldkeypub( )
        return self._coldkeypub

    @property
    def coldkeyfile(self) -> str:
        full_path = os.path.expanduser(os.path.join(self.config.wallet.path, self.config.wallet.name))
        return os.path.join(full_path, "coldkey")

    @property
    def coldkeypubfile(self) -> str:
        full_path = os.path.expanduser(os.path.join(self.config.wallet.path, self.config.wallet.name))
        return os.path.join(full_path, "coldkeypub.txt")

    @property
    def hotkeyfile(self) -> str:
        full_path = os.path.expanduser(
            os.path.join(self.config.wallet.path, self.config.wallet.name)
        )
        return os.path.join(full_path, "hotkeys", self.config.wallet.hotkey)

    def _load_coldkeypub(self) -> str:
        if not os.path.isfile( self.coldkeypubfile ):
            logger.critical("coldkeypubfile  {} does not exist".format( self.coldkeypubfile ))
            raise KeyFileError

        if not os.path.isfile( self.coldkeypubfile ):
            logger.critical("coldkeypubfile  {} is not a file".format( self.coldkeypubfile ))
            raise KeyFileError

        if not os.access( self.coldkeypubfile , os.R_OK):
            logger.critical("coldkeypubfile  {} is not readable".format( self.coldkeypubfile ))
            raise KeyFileError

        with open( self.coldkeypubfile, "r") as file:
            key = file.readline().strip()
            if not re.match("^0x[a-z0-9]{64}$", key):
                raise KeyFileError("Cold key pub file is corrupt")

        with open( self.coldkeypubfile , "r") as file:
            coldkeypub = file.readline().strip()

        logger.success("Loaded coldkey.pub: <cyan>{}</cyan>".format( coldkeypub ))
        return coldkeypub

    def _load_hotkey(self) -> 'bittensor.substrate.Keypair':

        if not os.path.isfile( self.hotkeyfile ):
            logger.critical("hotkeyfile  {} does not exist".format( self.hotkeyfile ))
            raise KeyFileError

        if not os.path.isfile( self.hotkeyfile ):
            logger.critical("hotkeyfile  {} is not a file".format( self.hotkeyfile ))
            raise KeyFileError

        if not os.access( self.hotkeyfile , os.R_OK):
            logger.critical("hotkeyfile  {} is not readable".format( self.hotkeyfile ))
            raise KeyFileError

        with open( self.hotkeyfile , 'rb') as file:
            data = file.read()
            try:
                # Try hotkey load.
                if is_encrypted(data):
                    password = bittensor.utils.Cli.ask_password()
                    logger.info("decrypting key... (this may take a few moments)")
                    data = decrypt_data(password, data)
                hotkey = load_keypair_from_data(data)
            except KeyError:
                logger.critical("Invalid password")
                raise KeyError("Invalid password")

            except KeyFileError as e:
                logger.critical("Keyfile corrupt")
                raise KeyFileError("Keyfile corrupt")

            logger.success("Loaded hotkey: <cyan>{}</cyan>".format(hotkey.public_key))
            return hotkey


    def _load_coldkey(self) -> 'bittensor.substrate.Keypair':
        if not os.path.isfile( self.coldkeyfile ):
            logger.critical("coldkeyfile  {} does not exist".format( self.coldkeyfile ))
            raise KeyFileError

        if not os.path.isfile( self.coldkeyfile ):
            logger.critical("coldkeyfile  {} is not a file".format( self.coldkeyfile ))
            raise KeyFileError

        if not os.access( self.coldkeyfile , os.R_OK):
            logger.critical("coldkeyfile  {} is not readable".format( self.coldkeyfile ))
            raise KeyFileError

        with open( self.coldkeyfile , 'rb') as file:
            data = file.read()
            try:
                # Try key load.
                if is_encrypted(data):
                    password = bittensor.utils.Cli.ask_password()
                    logger.info("decrypting key... (this may take a few moments)")
                    data = decrypt_data(password, data)
                coldkey = load_keypair_from_data(data)

            except KeyError:
                logger.critical("Invalid password")
                raise KeyError("Invalid password")

            except KeyFileError as e:
                logger.critical("Keyfile corrupt")
                raise KeyFileError("Keyfile corrupt")

            logger.success("Loaded coldkey: <cyan>{}</cyan>".format(coldkey.public_key))
            return coldkey

    @staticmethod
    def __is_world_readable(path):
        st = os.stat(path)
        return st.st_mode & stat.S_IROTH

    @staticmethod
    def __create_keypair() -> bittensor.substrate.Keypair:
        return bittensor.substrate.Keypair.create_from_mnemonic(bittensor.substrate.Keypair.generate_mnemonic())

    @staticmethod
    def __save_keypair(keypair : 'bittensor.substrate.Keypair', path : str):
        path = os.path.expanduser(path)
        with open(path, 'w') as file:
            json.dump(keypair.toDict(), file)
            file.close()
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)

    @staticmethod
    def __has_keypair(path):
        path = os.path.expanduser(path)
        return os.path.exists(path)

    def create_coldkey_from_uri(self, uri:str, use_password: bool = True, overwrite:bool = False):
         # Create directory 
        dir_path = os.path.expanduser(os.path.join(self.config.wallet.path, self.config.wallet.name))
        if not os.path.exists( dir_path ):
            os.makedirs( dir_path )

        # Create Key
        cli_utils.validate_create_path( self.coldkeyfile, overwrite = overwrite)
        self._coldkey = Keypair.create_from_uri( uri )
        cli_utils.display_mnemonic_msg( self._coldkey  )
        cli_utils.write_pubkey_to_text_file( self.coldkeyfile, self._coldkey.public_key )

        # Encrypt
        if use_password:
            password = cli_utils.input_password()
            logger.info("Encrypting coldkey ... (this might take a few moments)")
            coldkey_json_data = json.dumps( self._coldkey.toDict() ).encode()
            coldkey_data = encrypt(coldkey_json_data, password)
            del coldkey_json_data
        else:
            coldkey_data = json.dumps(self._coldkey.toDict()).encode()

        # Save
        cli_utils.save_keys( self.coldkeyfile, coldkey_data )
        cli_utils.set_file_permissions( self.coldkeyfile )

    def create_hotkey_from_uri( self, uri:str, use_password: bool = True, overwrite:bool = False):  
        # Create directory 
        dir_path = os.path.expanduser(
            os.path.join(self.config.wallet.path, self.config.wallet.name, "hotkeys")
        )
        if not os.path.exists( dir_path ):
            os.makedirs( dir_path )

        # Create
        cli_utils.validate_create_path( self.hotkeyfile, overwrite = overwrite)
        self._hotkey = Keypair.create_from_uri( uri )
        cli_utils.display_mnemonic_msg( self._hotkey )

        # Encrypt
        if use_password:
            password = cli_utils.input_password()
            logger.info("Encrypting hotkey ... (this might take a few moments)")
            hotkey_json_data = json.dumps( self._hotkey.toDict() ).encode()
            hotkey_data = encrypt(hotkey_json_data, password)
            del hotkey_json_data
        else:
            hotkey_data = json.dumps(self._hotkey.toDict()).encode()

        # Save
        cli_utils.save_keys( self.hotkeyfile, hotkey_data )
        cli_utils.set_file_permissions( self.hotkeyfile )


    def create_new_coldkey( self, n_words:int = 12, use_password: bool = True, overwrite:bool = False):    
        # Create directory 
        dir_path = os.path.expanduser(os.path.join(self.config.wallet.path, self.config.wallet.name))
        if not os.path.exists( dir_path ):
            os.makedirs( dir_path )

        # Create Key
        cli_utils.validate_create_path( self.coldkeyfile, overwrite = overwrite  )
        self._coldkey = cli_utils.gen_new_key( n_words )
        cli_utils.display_mnemonic_msg( self._coldkey  )
        cli_utils.write_pubkey_to_text_file( self.coldkeyfile, self._coldkey.public_key )

        # Encrypt
        if use_password:
            password = cli_utils.input_password()
            logger.info("Encrypting coldkey ... (this might take a few moments)")
            coldkey_json_data = json.dumps( self._coldkey.toDict() ).encode()
            coldkey_data = encrypt(coldkey_json_data, password)
            del coldkey_json_data
        else:
            coldkey_data = json.dumps(self._coldkey.toDict()).encode()

        # Save
        cli_utils.save_keys( self.coldkeyfile, coldkey_data )
        cli_utils.set_file_permissions( self.coldkeyfile )

    def create_new_hotkey( self, n_words:int = 12, use_password: bool = True, overwrite:bool = False):  
        # Create directory 
        dir_path = os.path.expanduser(
            os.path.join(self.config.wallet.path, self.config.wallet.name, "hotkeys")
        )
        if not os.path.exists( dir_path ):
            os.makedirs( dir_path )

        # Create
        cli_utils.validate_create_path( self.hotkeyfile, overwrite = overwrite )
        self._hotkey = cli_utils.gen_new_key( n_words )
        cli_utils.display_mnemonic_msg( self._hotkey )

        # Encrypt
        if use_password:
            password = cli_utils.input_password()
            logger.info("Encrypting hotkey ... (this might take a few moments)")
            hotkey_json_data = json.dumps( self._hotkey.toDict() ).encode()
            hotkey_data = encrypt(hotkey_json_data, password)
            del hotkey_json_data
        else:
            hotkey_data = json.dumps(self._hotkey.toDict()).encode()

        # Save
        cli_utils.save_keys( self.hotkeyfile, hotkey_data )
        cli_utils.set_file_permissions( self.hotkeyfile )

    def regenerate_coldkey( self, mnemonic: str, use_password: bool):
        # Create directory 
        dir_path = os.path.expanduser(os.path.join(self.config.wallet.path, self.config.wallet.name))
        if not os.path.exists( dir_path ):
            os.makedirs( dir_path )

        # Regenerate
        self._coldkey = cli_utils.validate_generate_mnemonic( mnemonic )
        cli_utils.write_pubkey_to_text_file( self.coldkeyfile, self._coldkey.public_key )
        
        # Encrypt
        if use_password:
            password = cli_utils.input_password()
            logger.info("Encrypting key ... (this might take a few moments)")
            json_data = json.dumps( self._coldkey.toDict() ).encode()
            coldkey_data = encrypt(json_data, password)
            del json_data
        else:
            coldkey_data = json.dumps(self._coldkey.toDict()).encode()

        # Save
        cli_utils.save_keys( self.coldkeyfile, coldkey_data ) 
        cli_utils.set_file_permissions( self.coldkeyfile )

    def regenerate_hotkey( self, mnemonic: str, use_password: bool = True, overwrite:bool = False):
        # Create directory 
        dir_path = os.path.expanduser(os.path.join(self.config.wallet.path, self.config.wallet.name, "hotkeys"))
        if not os.path.exists( dir_path ):
            os.makedirs( dir_path )

        # Regenerate
        cli_utils.validate_create_path( self.hotkeyfile, overwrite = overwrite)
        self._hotkey = cli_utils.validate_generate_mnemonic( mnemonic )

        # Encrypt
        if use_password:
            password = cli_utils.input_password()
            logger.info("Encrypting hotkey ... (this might take a few moments)")
            hotkey_json_data = json.dumps( self._hotkey.toDict() ).encode()
            hotkey_data = encrypt(hotkey_json_data, password)
            del hotkey_json_data
        else:
            hotkey_data = json.dumps(self._hotkey.toDict()).encode()
        
        # Save
        cli_utils.save_keys( self.hotkeyfile, hotkey_data )
        cli_utils.set_file_permissions( self.hotkeyfile )
