
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

import json
import os
import re
import stat
import sys

from loguru import logger
logger = logger.opt(colors=True)

import bittensor
from typing import Tuple, List, Union, Optional
from substrateinterface import Keypair
from substrateinterface.utils.ss58 import ss58_encode
from bittensor.utils.cli_utils import cli_utils
from bittensor._crypto import encrypt, is_encrypted, decrypt_data, KeyError
from bittensor._crypto.keyfiles import load_keypair_from_data, KeyFileError

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
        name:str,
        path:str,
        hotkey:str 
    ):
        r""" Init bittensor wallet object containing a hot and coldkey.
            Args:
                name (required=False, default='default):
                    The name of the wallet to unlock for running bittensor
                hotkey (required=False, default='default):
                    The name of hotkey used to running the miner.
                path (required=False, default='~/.bittensor/wallets/'):
                    The path to your bittensor wallets
        """
        self._name_string = name
        self._path_string = path
        self._hotkey_string = hotkey
        self._hotkey = None
        self._coldkey = None
        self._coldkeypub = None

    def __str__(self):
        return "Wallet ({}, {}, {})".format(self._name_string, self._hotkey_string, self._path_string)
    
    def __repr__(self):
        return self.__str__()

    def get_uid ( self, subtensor: 'bittensor.Subtensor' = None ) -> int:
        """ Returns this wallet's hotkey uid or -1 if the hotkey is not subscribed.
            Args:
                subtensor( 'bittensor.Subtensor' ):
                    Bittensor subtensor connection. Overrides with defaults if None.
            Return:
                uid (int):
                    Network uid.
        """
        try:
            if subtensor == None:
                subtensor = bittensor.subtensor()
            hotkey_uid = int(subtensor.get_uid_for_pubkey( self.hotkey.public_key ))
            if hotkey_uid == None:
                return -1
            else:
                return int(hotkey_uid)
        except Exception:
            return -1

    def get_stake ( self, subtensor: 'bittensor.Subtensor' = None ) -> 'bittensor.Balance':
        """ Returns this wallet's staking balance from passed subtensor connection.
            Args:
                subtensor( 'bittensor.Subtensor' ):
                    Bittensor subtensor connection. Overrides with defaults if None.
            Return:
                balance (bittensor.utils.balance.Balance):
                    Stake account balance
        """
        try:
            if subtensor == None:
                subtensor = bittensor.subtensor()
            self.assert_hotkey()
            hotkey_uid = self.get_uid()
            if hotkey_uid == -1:
                return bittensor.Balance(0)               
            stake = subtensor.get_stake_for_uid( hotkey_uid )
            return stake
        except Exception as e:
            print (e)
            return bittensor.Balance(0)

    def get_balance ( self, subtensor: 'bittensor.Subtensor' = None ) -> 'bittensor.Balance':
        """ Returns this wallet's coldkey balance from passed subtensor connection.
            Args:
                subtensor( 'bittensor.Subtensor' ):
                    Bittensor subtensor connection. Overrides with defaults if None.
            Return:
                balance (bittensor.utils.balance.Balance):
                    Coldkey account balance
        """
        try:
            if subtensor == None:
                subtensor = bittensor.subtensor()
            self.assert_coldkeypub()
            amount_balance = subtensor.get_balance( ss58_encode(self.coldkeypub) )
            return amount_balance
        except Exception as e:
            print (e)
            return bittensor.Balance(0)

    def stake( self, 
        amount: Union[float, bittensor.Balance] = None, 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        subtensor: 'bittensor.Subtensor' = None 
    ) -> bool:
        """ Stakes tokens from this wallet's coldkey onto it's hotkey.
            Args:
                amount_tao (float):
                    amount of tao to stake or bittensor balance object. If None, stakes all available balance.
                wait_for_inclusion (bool):
                    if set, waits for the extrinsic to enter a block before returning true, 
                    or returns false if the extrinsic fails to enter the block within the timeout.   
                wait_for_finalization (bool):
                    if set, waits for the extrinsic to be finalized on the chain before returning true,
                    or returns false if the extrinsic fails to be finalized within the timeout.
                subtensor( `bittensor.Subtensor` ):
                    Bittensor subtensor connection. Overrides with defaults if None.
            Returns:
                success (bool):
                    flag is true if extrinsic was finalized or uncluded in the block. 
                    If we did not wait for finalization / inclusion, the response is true.
        """
        self.assert_coldkey()
        self.assert_coldkeypub()
        self.assert_hotkey()
        if amount == None:
            amount = self.get_balance()
        if not isinstance(amount, bittensor.Balance):
            amount = bittensor.utils.balance.Balance.from_float( amount )
        if subtensor == None:
            subtensor = bittensor.subtensor()
        return subtensor.add_stake( wallet = self, amount = amount, wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization )

    def unstake( self, 
        amount: Union[float, bittensor.Balance] = None, 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        subtensor: 'bittensor.Subtensor' = None 
    ) -> bool:
        """ Removes stake from this wallet's hotkey and moves them onto it's coldkey balance.
            Args:
                amount_tao (float):
                    amount of tao to unstake or bittensor balance object. If None, unstakes all available hotkey balance.
                wait_for_inclusion (bool):
                    if set, waits for the extrinsic to enter a block before returning true, 
                    or returns false if the extrinsic fails to enter the block within the timeout.   
                wait_for_finalization (bool):
                    if set, waits for the extrinsic to be finalized on the chain before returning true,
                    or returns false if the extrinsic fails to be finalized within the timeout.
                subtensor( `bittensor.Subtensor` ):
                    Bittensor subtensor connection. Overrides with defaults if None.
            Returns:
                success (bool):
                    flag is true if extrinsic was finalized or uncluded in the block. 
                    If we did not wait for finalization / inclusion, the response is true.
        """
        self.assert_coldkey()
        self.assert_coldkeypub()
        self.assert_hotkey()
        if amount == None:
            amount = self.get_stake()
        if not isinstance(amount, bittensor.Balance):
            amount = bittensor.utils.balance.Balance.from_float( amount )
        if subtensor == None:
            subtensor = bittensor.subtensor()
        return subtensor.unstake( wallet = self, amount = amount, wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization )

    def transfer( 
        self, 
        dest:str,
        amount: Union[float, bittensor.Balance] , 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        subtensor: 'bittensor.Subtensor' = None 
    ) -> bool:
        """ Transfers Tao from this wallet's coldkey to the destination address.
            Args:
                dest(`type`:str, required):
                    The destination address either encoded as a ss58 or ed255 public-key string of 
                    secondary account.
                amount_tao (float, required):
                    amount of tao to transfer or a bittensor balance object.
                wait_for_inclusion (bool):
                    if set, waits for the extrinsic to enter a block before returning true, 
                    or returns false if the extrinsic fails to enter the block within the timeout.   
                wait_for_finalization (bool):
                    if set, waits for the extrinsic to be finalized on the chain before returning true,
                    or returns false if the extrinsic fails to be finalized within the timeout.
                subtensor( `bittensor.Subtensor` ):
                    Bittensor subtensor connection. Overrides with defaults if None.
            Returns:
                success (bool):
                    flag is true if extrinsic was finalized or uncluded in the block. 
                    If we did not wait for finalization / inclusion, the response is true.
        """
        self.assert_coldkey()
        self.assert_coldkeypub()
        self.assert_hotkey()
        if not isinstance(amount, bittensor.Balance):
            amount = bittensor.utils.balance.Balance.from_float( amount )
        balance = self.get_balance()
        if amount > balance:
            bittensor.logging.error(prefix='Transfer', sufix='Not enough balance to transfer: {} > {}'.format(amount, balance))
            return False
        if subtensor == None:
            subtensor = bittensor.subtensor()
        return subtensor.transfer( wallet = self, amount = amount, dest = dest, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )

    def create(self) -> 'Wallet':
        """ Checks for existing coldkeypub and hotkeys and creates them if non-existent.
        """
        # ---- Setup Wallet. ----
        if not self.has_coldkeypub:
            self.create_new_coldkey( n_words = 12, use_password = True )
        if not self.has_coldkeypub:
            raise RuntimeError('The axon must have access to a decrypted coldkeypub')
        if not self.has_hotkey:
            self.create_new_hotkey( n_words = 12, use_password = False )
        if not self.has_hotkey:
            raise RuntimeError('The axon must have access to a decrypted hotkey')
        return self

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
    def hotkey(self) -> Keypair:
        r""" Loads the hotkey from wallet.path/wallet.name/hotkeys/wallet.hotkey or raises an error.
            Returns:
                hotkey (Keypair):
                    hotkey loaded from config arguments.
            Raises:
                KeyFileError: Raised if the file is corrupt of non-existent.
                KeyError: Raised if the user enters an incorrec password for an encrypted keyfile.
        """
        if self._hotkey == None:
            self._hotkey = self._load_hotkey()
        return self._hotkey

    @property
    def coldkey(self) -> 'Keypair':
        r""" Loads the hotkey from wallet.path/wallet.name/coldkey or raises an error.
            Returns:
                coldkey (Keypair):
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
        full_path = os.path.expanduser(os.path.join(self._path_string, self._name_string))
        return os.path.join(full_path, "coldkey")

    @property
    def coldkeypubfile(self) -> str:
        full_path = os.path.expanduser(os.path.join(self._path_string, self._name_string))
        file_name = os.path.join(full_path, "coldkeypub.txt")
        return file_name

    @property
    def hotkeyfile(self) -> str:
        full_path = os.path.expanduser(
            os.path.join(self._path_string, self._name_string)
        )
        return os.path.join(full_path, "hotkeys", self._hotkey_string)

    def _load_coldkeypub(self) -> str:
        if not os.path.isfile( self.coldkeypubfile ):
            logger.critical("coldkeypubfile  {} does not exist".format( self.coldkeypubfile ))
            raise KeyFileError

        if not os.access( self.coldkeypubfile , os.R_OK):
            logger.critical("coldkeypubfile  {} is not readable".format( self.coldkeypubfile ))
            raise KeyFileError

        with open( self.coldkeypubfile, "r") as file:
            key = file.readline().strip()
            if not re.match("^0x[a-z0-9]{64}$", key):
                logger.critical("Coldkey pub file is corrupt")
                raise KeyFileError("Coldkey pub file is corrupt")

        with open( self.coldkeypubfile , "r") as file:
            coldkeypub = file.readline().strip()

        logger.success("Loaded coldkey.pub:".ljust(20) + "<blue>{}</blue>".format( coldkeypub ))
        return coldkeypub

    def _load_hotkey(self) -> 'Keypair':

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

            except KeyFileError:
                logger.critical("Keyfile corrupt")
                raise KeyFileError("Keyfile corrupt")

            logger.success("Loaded hotkey:".ljust(20) + "<blue>{}</blue>".format(hotkey.public_key))
            return hotkey


    def _load_coldkey(self) -> 'Keypair':
        
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

            except KeyFileError:
                logger.critical("Keyfile corrupt")
                raise KeyFileError("Keyfile corrupt")

            logger.success("Loaded coldkey:".ljust(20) + "<blue>{}</blue>".format(coldkey.public_key))
            return coldkey

    @staticmethod
    def __is_world_readable(path):
        st = os.stat(path)
        return st.st_mode & stat.S_IROTH

    @staticmethod
    def __create_keypair() -> Keypair:
        return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())

    @staticmethod
    def __save_keypair(keypair : 'Keypair', path : str):
        path = os.path.expanduser(path)
        with open(path, 'w') as file:
            json.dump(Wallet("", "", "").to_dict(keypair), file)
            file.close()
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)

    @staticmethod
    def __has_keypair(path):
        path = os.path.expanduser(path)
        return os.path.exists(path)

    def create_coldkey_from_uri(self, uri:str, use_password: bool = True, overwrite:bool = False) -> 'Wallet':
         # Create directory 
        dir_path = os.path.expanduser(os.path.join(self._path_string, self._name_string))
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
            coldkey_json_data = json.dumps( self.to_dict(self._coldkey) ).encode()
            coldkey_data = encrypt(coldkey_json_data, password)
            del coldkey_json_data
        else:
            coldkey_data = json.dumps(self.to_dict(self._coldkey)).encode()

        # Save
        cli_utils.save_keys( self.coldkeyfile, coldkey_data )
        cli_utils.set_file_permissions( self.coldkeyfile )
        return self

    def create_hotkey_from_uri( self, uri:str, use_password: bool = True, overwrite:bool = False) -> 'Wallet':  
        # Create directory 
        dir_path = os.path.expanduser(
            os.path.join(self._path_string, self._name_string, "hotkeys")
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
            hotkey_json_data = json.dumps( self.to_dict(self._hotkey) ).encode()
            hotkey_data = encrypt(hotkey_json_data, password)
            del hotkey_json_data
        else:
            hotkey_data = json.dumps(self.to_dict(self._hotkey)).encode()

        # Save
        cli_utils.save_keys( self.hotkeyfile, hotkey_data )
        cli_utils.set_file_permissions( self.hotkeyfile )
        return self

    def create_new_coldkey( self, n_words:int = 12, use_password: bool = True, overwrite:bool = False) -> 'Wallet':  
        # Create directory 
        dir_path = os.path.expanduser(os.path.join(self._path_string, self._name_string))
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
            coldkey_json_data = json.dumps( self.to_dict(self._coldkey) ).encode()
            coldkey_data = encrypt(coldkey_json_data, password)
            del coldkey_json_data
        else:
            coldkey_data = json.dumps(self.to_dict(self._coldkey)).encode()

        # Save
        cli_utils.save_keys( self.coldkeyfile, coldkey_data )
        cli_utils.set_file_permissions( self.coldkeyfile )
        return self

    def create_new_hotkey( self, n_words:int = 12, use_password: bool = True, overwrite:bool = False) -> 'Wallet':  
        # Create directory 
        dir_path = os.path.expanduser(
            os.path.join(self._path_string, self._name_string, "hotkeys")
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
            hotkey_json_data = json.dumps( self.to_dict(self._hotkey) ).encode()
            hotkey_data = encrypt(hotkey_json_data, password)
            del hotkey_json_data
        else:
            hotkey_data = json.dumps(self.to_dict(self._hotkey)).encode()

        # Save
        cli_utils.save_keys( self.hotkeyfile, hotkey_data )
        cli_utils.set_file_permissions( self.hotkeyfile )
        return self

    def regenerate_coldkey( self, mnemonic: str, use_password: bool,  overwrite:bool = False) -> 'Wallet':
        # Create directory 
        dir_path = os.path.expanduser(os.path.join(self._path_string, self._name_string))
        if not os.path.exists( dir_path ):
            os.makedirs( dir_path )

        # Regenerate
        cli_utils.validate_create_path( self.coldkeyfile, overwrite = overwrite)
        self._coldkey = cli_utils.validate_generate_mnemonic( mnemonic )
        cli_utils.write_pubkey_to_text_file( self.coldkeyfile, self._coldkey.public_key )
        
        # Encrypt
        if use_password:
            password = cli_utils.input_password()
            logger.info("Encrypting key ... (this might take a few moments)")
            json_data = json.dumps( self.to_dict(self._coldkey) ).encode()
            coldkey_data = encrypt(json_data, password)
            del json_data
        else:
            coldkey_data = json.dumps(self.to_dict(self._coldkey) ).encode()

        # Save
        cli_utils.save_keys( self.coldkeyfile, coldkey_data ) 
        cli_utils.set_file_permissions( self.coldkeyfile )
        return self

    def regenerate_hotkey( self, mnemonic: str, use_password: bool = True, overwrite:bool = False) -> 'Wallet':
        # Create directory 
        dir_path = os.path.expanduser(os.path.join(self._path_string, self._name_string, "hotkeys"))
        if not os.path.exists( dir_path ):
            os.makedirs( dir_path )

        # Regenerate
        cli_utils.validate_create_path( self.hotkeyfile, overwrite = overwrite)
        self._hotkey = cli_utils.validate_generate_mnemonic( mnemonic )

        # Encrypt
        if use_password:
            password = cli_utils.input_password()
            logger.info("Encrypting hotkey ... (this might take a few moments)")
            hotkey_json_data = json.dumps( self.to_dict(self._hotkey)  ).encode()
            hotkey_data = encrypt(hotkey_json_data, password)
            del hotkey_json_data
        else:
            hotkey_data = json.dumps( self.to_dict(self._hotkey)).encode()
        
        # Save
        cli_utils.save_keys( self.hotkeyfile, hotkey_data )
        cli_utils.set_file_permissions( self.hotkeyfile )
        return self

    def to_dict(self, keypair):
        return {
            'accountId': keypair.public_key,
            'publicKey': keypair.public_key,
            'secretPhrase': keypair.mnemonic,
            'secretSeed': "0x" + keypair.seed_hex,
            'ss58Address': keypair.ss58_address
        }