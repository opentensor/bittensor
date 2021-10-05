""" Implementation of the wallet class, which manages balances with staking and transfer. Also manages hotkey and coldkey.
"""
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
from multiprocessing import Value
import os
import re
import sys
import time
import requests
from types import SimpleNamespace

from typing import Union
from loguru import logger
from substrateinterface import Keypair
from substrateinterface.utils.ss58 import ss58_encode

import bittensor
from bittensor.utils.cli_utils import cli_utils
from bittensor._crypto import encrypt, is_encrypted, decrypt_data, CryptoKeyError
from bittensor._crypto.keyfiles import load_keypair_from_data, KeyFileError

logger = logger.opt(colors=True)

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
        hotkey:str,
        email:str = None
    ):
        r""" Init bittensor wallet object containing a hot and coldkey.
            Args:
                name (required=True, default='default):
                    The name of the wallet to unlock for running bittensor
                hotkey (required=True, default='default):
                    The name of hotkey used to running the miner.
                path (required=True, default='~/.bittensor/wallets/'):
                    The path to your bittensor wallets
                email (required=False, default=None):
                    Registration email.
        """
        self._name_string = name
        self._path_string = path
        self._hotkey_string = hotkey
        self._email = email
        self._hotkey = None
        self._coldkey = None
        self._coldkeypub = None

    def __str__(self):
        return "Wallet ({}, {}, {})".format(self._name_string, self._hotkey_string, self._path_string)
    
    def __repr__(self):
        return self.__str__()

    def register ( self, email:str = None, subtensor: 'bittensor.Subtensor' = None ) -> 'bittensor.Wallet':
        """ Registers this wallet on the chain.
            Args:
                subtensor( 'bittensor.Subtensor' ):
                    Bittensor subtensor connection. Overrides with defaults if None.
            Return:
                wallet.
        """
        if email == None:
            email = self._email
            if email == None:
                raise ValueError('You must pass registration email either through wallet initialization or during the register call.')
        if subtensor == None: subtensor = bittensor.subtensor()
        if self.is_registered( subtensor = subtensor ): print ('Already registered {}'.format( self.hotkey.ss58_address ))
        else:
            headers = {'Content-type': 'application/json'}
            url = 'http://' + bittensor.__registration_servers__[0] + '/register?email={}&hotkey={}&coldkey={}&hotkey_signature={}&network={}'.format( email, self.hotkey.ss58_address, self.coldkey.ss58_address, 'signaturefaked', subtensor.network)
            response = requests.post(url, headers=headers)
            response_str = str(bytes.decode(response.content)) 
            if response_str == 'Email Sent':
                print ('Waiting for confirmation from email: {}'.format(email))
                while True:
                    if self.is_registered( subtensor = subtensor ):
                        print ('Registered hotkey: {}'.format( self.hotkey.ss58_address ))
                        return self      
                    time.sleep(2)
            else:
                print ('Failed for reason: {}'.format( response_str ))
                return self

    def is_registered( self, subtensor: 'bittensor.Subtensor' = None ) -> bool:
        """ Returns true if this wallet is registered.
            Args:
                subtensor( 'bittensor.Subtensor' ):
                    Bittensor subtensor connection. Overrides with defaults if None.
                    Determines which network we check for registration.
            Return:
                is_registered (bool):
                    Is the wallet registered on the chain.
        """
        if subtensor == None: subtensor = bittensor.subtensor()
        return subtensor.is_hotkey_registered( self.hotkey.ss58_address )

    def get_neuron ( self, subtensor: 'bittensor.Subtensor' = None ) -> SimpleNamespace:
        """ Returns this wallet's neuron information from subtensor.
            Args:
                subtensor( 'bittensor.Subtensor' ):
                    Bittensor subtensor connection. Overrides with defaults if None.
            Return:
                neuron (SimpleNamespace):
                    neuron account on the chain.
        """
        self.assert_hotkey()             
        if subtensor == None: subtensor = bittensor.subtensor()
        if not self.is_registered(subtensor=subtensor): raise ValueError('This wallet is not registered. Call wallet.register( email = <your email>) before this function.')
        neuron = subtensor.neuron_for_wallet( self )
        return neuron

    def get_uid ( self, subtensor: 'bittensor.Subtensor' = None ) -> int:
        """ Returns this wallet's hotkey uid or -1 if the hotkey is not subscribed.
            Args:
                subtensor( 'bittensor.Subtensor' ):
                    Bittensor subtensor connection. Overrides with defaults if None.
            Return:
                uid (int):
                    Network uid.
        """
        if subtensor == None: subtensor = bittensor.subtensor()
        if not self.is_registered(subtensor=subtensor): raise ValueError('This wallet is not registered. Call wallet.register( email = <your email>) before this function.')
        neuron = self.get_neuron(subtensor = subtensor)
        if neuron.is_null:
            return -1
        else:
            return neuron.uid

    def get_stake ( self, subtensor: 'bittensor.Subtensor' = None ) -> 'bittensor.Balance':
        """ Returns this wallet's staking balance from passed subtensor connection.
            Args:
                subtensor( 'bittensor.Subtensor' ):
                    Bittensor subtensor connection. Overrides with defaults if None.
            Return:
                balance (bittensor.utils.balance.Balance):
                    Stake account balance
        """
        if subtensor == None: subtensor = bittensor.subtensor()
        if not self.is_registered(subtensor=subtensor): raise ValueError('This wallet is not registered. Call wallet.register( email = <your email>) before this function.')
        neuron = self.get_neuron(subtensor = subtensor)
        if neuron.is_null:
            return bittensor.Balance(0)
        else:
            return bittensor.Balance(neuron.stake)

    def get_balance( self, subtensor: 'bittensor.Subtensor' = None ) -> 'bittensor.Balance':
        """ Returns this wallet's coldkey balance from passed subtensor connection.
            Args:
                subtensor( 'bittensor.Subtensor' ):
                    Bittensor subtensor connection. Overrides with defaults if None.
            Return:
                balance (bittensor.utils.balance.Balance):
                    Coldkey balance.
        """
        if subtensor == None: subtensor = bittensor.subtensor()
        return subtensor.get_balance(address = self.coldkeypub.ss58_address)

    def add_stake( self, 
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
        if subtensor == None: subtensor = bittensor.subtensor()
        if not self.is_registered(subtensor=subtensor): raise ValueError('This wallet is not registered. Call wallet.register( email = <your email>) before this function.')
        if amount == None:
            amount = self.get_balance()
        if not isinstance(amount, bittensor.Balance):
            amount = bittensor.utils.balance.Balance.from_float( amount )
        return subtensor.add_stake( wallet = self, amount = amount, wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization )

    def remove_stake( self, 
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
        if subtensor == None: subtensor = bittensor.subtensor()
        if not self.is_registered(subtensor=subtensor): raise ValueError('This wallet is not registered. Call wallet.register( email = <your email>) before this function.')
        if amount == None:
            amount = self.get_stake()
        if not isinstance(amount, bittensor.Balance):
            amount = bittensor.utils.balance.Balance.from_float( amount )
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
        if subtensor == None: subtensor = bittensor.subtensor()
        if not self.is_registered(subtensor=subtensor): raise ValueError('This wallet is not registered. Call wallet.register( email = <your email>) before this function.')
        if not isinstance(amount, bittensor.Balance):
            amount = bittensor.utils.balance.Balance.from_float( amount )
        balance = self.get_balance()
        if amount > balance:
            bittensor.logging.error(prefix='Transfer', sufix='Not enough balance to transfer: {} > {}'.format(amount, balance))
            return False
        return subtensor.transfer( wallet = self, amount = amount, dest = dest, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )

    def create_if_non_existent( self, coldkey_use_password:bool = True, hotkey_use_password:bool = True) -> 'Wallet':
        """ Checks for existing coldkeypub and hotkeys and creates them if non-existent.
        """
        return self.create(coldkey_use_password, hotkey_use_password)

    def create (self, coldkey_use_password:bool = True, hotkey_use_password:bool = True ) -> 'Wallet':
        """ Checks for existing coldkeypub and hotkeys and creates them if non-existent.

        """
        return self.create(coldkey_use_password, hotkey_use_password)

    def create (self, coldkey_use_password:bool = True, hotkey_use_password:bool = True ) -> 'Wallet':
        """ Checks for existing coldkeypub and hotkeys and creates them if non-existent.

        """
        # ---- Setup Wallet. ----
        if not self.has_coldkeypub:
            self.create_new_coldkey( n_words = 12, use_password = coldkey_use_password )
        if not self.has_coldkeypub:
            raise RuntimeError('The axon must have access to a decrypted coldkeypub')
        if not self.has_hotkey:
            self.create_new_hotkey( n_words = 12, use_password = hotkey_use_password )
        if not self.has_hotkey:
            raise RuntimeError('The axon must have access to a decrypted hotkey')
        return self

    def assert_hotkey(self):
        r""" Checks for a valid hotkey from wallet.path/wallet.name/hotkeys/wallet.hotkey or exits.
        """
        try:
            assert self.has_hotkey
        except Exception:
            sys.exit(1)
        
    def assert_coldkey(self):
        r""" Checks for a valid coldkey from wallet.path/wallet.name/hotkeys/wallet.hotkey or exits.
        """
        try:
            assert self.has_coldkey
        except Exception:
            sys.exit(1)

    def assert_coldkeypub(self):
        r""" Checks for a valid coldkeypub from wallet.path/wallet.name/coldkeypub.txt or exits
        """
        try:
            assert self.has_coldkeypub
        except Exception:
            sys.exit(1)

    @property
    def has_hotkey(self) -> bool:
        r""" True if a hotkey can be loaded from wallet.path/wallet.name/hotkeys/wallet.hotkey or returns None.
            Returns:
                hotkey (bool):
                    True if the hotkey can be loaded from config arguments or False
        """
        try:
            if self.hotkey:
                return True
        except KeyFileError:
            return False
        except CryptoKeyError:
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
            if self.coldkey:
                return True
        except KeyFileError:
            return False
        except CryptoKeyError:
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
            if self.coldkeypub:
                return True
        except KeyFileError:
            return False
        except CryptoKeyError:
            return False
        except Exception:
            return False

    @property
    def hotkey(self) -> 'Keypair':
        r""" Loads the hotkey from wallet.path/wallet.name/hotkeys/wallet.hotkey or raises an error.
            Returns:
                hotkey (Keypair):
                    hotkey loaded from config arguments.
            Raises:
                KeyFileError: Raised if the file is corrupt of non-existent.
                CryptoKeyError: Raised if the user enters an incorrec password for an encrypted keyfile.
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
                CryptoKeyError: Raised if the user enters an incorrec password for an encrypted keyfile.
        """
        if self._coldkey == None:
            self._coldkey = self._load_coldkey( )
        return self._coldkey

    @property
    def coldkeypub(self) -> 'Keypair':
        r""" Loads the coldkeypub from wallet.path/wallet.name/coldkeypub.txt or raises an error.
            Returns:
                coldkeypub (Keypair):
                    colkeypub loaded from config arguments.
            Raises:
                KeyFileError: Raised if the file is corrupt of non-existent.
                CryptoKeyError: Raised if the user enters an incorrec password for an encrypted keyfile.
        """
        if self._coldkeypub == None:
            self._coldkeypub = self._load_coldkeypub( )
        return self._coldkeypub

    @property
    def coldkeyfile(self) -> str:
        """ Return the path where coldkey was stored
        """
        full_path = os.path.expanduser(os.path.join(self._path_string, self._name_string))
        return os.path.join(full_path, "coldkey")

    @property
    def coldkeypubfile(self) -> str:
        """ Return the path where coldkey public key was stored
        """
        full_path = os.path.expanduser(os.path.join(self._path_string, self._name_string))
        file_name = os.path.join(full_path, "coldkeypub.txt")
        return file_name

    @property
    def hotkeyfile(self) -> str:
        """ Return the path where hotkey was stored
        """
        full_path = os.path.expanduser(
            os.path.join(self._path_string, self._name_string)
        )
        return os.path.join(full_path, "hotkeys", self._hotkey_string)

    def _load_coldkeypub(self) -> 'Keypair':
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

        coldkeypub_keypair = Keypair( ss58_address = ss58_encode(coldkeypub) )
        logger.success("Loaded coldkey.pub:".ljust(20) + "<blue>{}</blue>".format( coldkeypub_keypair.ss58_address ))
        return coldkeypub_keypair

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
                    password = cli_utils.ask_password()
                    logger.info("decrypting key... (this may take a few moments)")
                    data = decrypt_data(password, data)
                hotkey = load_keypair_from_data(data)
            except CryptoKeyError:
                logger.critical("Invalid password")
                raise CryptoKeyError("Invalid password") from CryptoKeyError()

            except KeyFileError:
                logger.critical("Keyfile corrupt")
                raise KeyFileError("Keyfile corrupt") from KeyFileError()

            logger.success("Loaded hotkey:".ljust(20) + "<blue>{}</blue>".format(hotkey.ss58_address))
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
                    password = cli_utils.ask_password()
                    logger.info("decrypting key... (this may take a few moments)")
                    data = decrypt_data(password, data)
                coldkey = load_keypair_from_data(data)

            except CryptoKeyError:
                logger.critical("Invalid password")
                raise CryptoKeyError("Invalid password") from CryptoKeyError()

            except KeyFileError:
                logger.critical("Keyfile corrupt")
                raise KeyFileError("Keyfile corrupt") from KeyFileError()

            logger.success("Loaded coldkey:".ljust(20) + "<blue>{}</blue>".format(coldkey.ss58_address))
            return coldkey
            
    def create_coldkey_from_uri(self, uri:str, use_password: bool = True, overwrite:bool = False) -> 'Wallet':
        """ Creates coldkey from suri string, optionally encrypts it with the user's inputed password.
            Args:
                uri: (str, required):
                    URI string to use i.e. /Alice or /Bob
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional): 
                    Will this operation overwrite the coldkey under the same path <wallet path>/<wallet name>/coldkey
            Returns:
                wallet (bittensor.Wallet):
                    this object with newly created coldkey.
        """
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
        """ Creates hotkey from suri string, optionally encrypts it with the user's inputed password.
            Args:
                uri: (str, required):
                    URI string to use i.e. /Alice or /Bob
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional): 
                    Will this operation overwrite the hotkey under the same path <wallet path>/<wallet name>/hotkeys/<hotkey>
            Returns:
                wallet (bittensor.Wallet):
                    this object with newly created hotkey.
        """
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

    def new_coldkey( self, n_words:int = 12, use_password: bool = True, overwrite:bool = False) -> 'Wallet':  
        """ Creates a new coldkey, optionally encrypts it with the user's inputed password and saves to disk.
            Args:
                n_words: (int, optional):
                    Number of mnemonic words to use.
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional): 
                    Will this operation overwrite the coldkey under the same path <wallet path>/<wallet name>/coldkey
            Returns:
                wallet (bittensor.Wallet):
                    this object with newly created coldkey.
        """
        self.create_new_coldkey( n_words, use_password, overwrite )

    def create_new_coldkey( self, n_words:int = 12, use_password: bool = True, overwrite:bool = False) -> 'Wallet':  
        """ Creates a new coldkey, optionally encrypts it with the user's inputed password and saves to disk.
            Args:
                n_words: (int, optional):
                    Number of mnemonic words to use.
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional): 
                    Will this operation overwrite the coldkey under the same path <wallet path>/<wallet name>/coldkey
            Returns:
                wallet (bittensor.Wallet):
                    this object with newly created coldkey.
        """
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

    def new_hotkey( self, n_words:int = 12, use_password: bool = True, overwrite:bool = False) -> 'Wallet':  
        """ Creates a new hotkey, optionally encrypts it with the user's inputed password and saves to disk.
            Args:
                n_words: (int, optional):
                    Number of mnemonic words to use.
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional): 
                    Will this operation overwrite the hotkey under the same path <wallet path>/<wallet name>/hotkeys/<hotkey>
            Returns:
                wallet (bittensor.Wallet):
                    this object with newly created hotkey.
        """
        self.create_new_hotkey( n_words, use_password, overwrite )

    def create_new_hotkey( self, n_words:int = 12, use_password: bool = True, overwrite:bool = False) -> 'Wallet':  
        """ Creates a new hotkey, optionally encrypts it with the user's inputed password and saves to disk.
            Args:
                n_words: (int, optional):
                    Number of mnemonic words to use.
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional): 
                    Will this operation overwrite the hotkey under the same path <wallet path>/<wallet name>/hotkeys/<hotkey>
            Returns:
                wallet (bittensor.Wallet):
                    this object with newly created hotkey.
        """
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

    def regen_coldkey( self, mnemonic: Union[list, str], use_password: bool = True,  overwrite:bool = False) -> 'Wallet':
        """ Regenerates the coldkey from passed mnemonic, encrypts it with the user's password and save the file
            Args:
                mnemonic: (Union[list, str], optional):
                    Key mnemonic as list of words or string space separated words.
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional): 
                    Will this operation overwrite the coldkey under the same path <wallet path>/<wallet name>/coldkey
            Returns:
                wallet (bittensor.Wallet):
                    this object with newly created coldkey.
        """
        self.regenerate_coldkey(mnemonic, use_password, overwrite)

    def regenerate_coldkey( self, mnemonic: Union[list, str], use_password: bool = True,  overwrite:bool = False) -> 'Wallet':
        """ Regenerates the coldkey from passed mnemonic, encrypts it with the user's password and save the file
            Args:
                mnemonic: (Union[list, str], optional):
                    Key mnemonic as list of words or string space separated words.
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional): 
                    Will this operation overwrite the coldkey under the same path <wallet path>/<wallet name>/coldkey
            Returns:
                wallet (bittensor.Wallet):
                    this object with newly created coldkey.
        """
        if isinstance( mnemonic, str):
            mnemonic = mnemonic.split()

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

    def regen_hotkey( self, mnemonic: Union[list, str], use_password: bool = True, overwrite:bool = False) -> 'Wallet':
        """ Regenerates the hotkey from passed mnemonic, encrypts it with the user's password and save the file
            Args:
                mnemonic: (Union[list, str], optional):
                    Key mnemonic as list of words or string space separated words.
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional): 
                    Will this operation overwrite the hotkey under the same path <wallet path>/<wallet name>/hotkeys/<hotkey>
            Returns:
                wallet (bittensor.Wallet):
                    this object with newly created hotkey.
        """
        self.regenerate_hotkey(mnemonic, use_password, overwrite)

    def regenerate_hotkey( self, mnemonic: Union[list, str], use_password: bool = True, overwrite:bool = False) -> 'Wallet':
        """ Regenerates the hotkey from passed mnemonic, encrypts it with the user's password and save the file
            Args:
                mnemonic: (Union[list, str], optional):
                    Key mnemonic as list of words or string space separated words.
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional): 
                    Will this operation overwrite the hotkey under the same path <wallet path>/<wallet name>/hotkeys/<hotkey>
            Returns:
                wallet (bittensor.Wallet):
                    this object with newly created hotkey.
        """
        if isinstance( mnemonic, str):
            mnemonic = mnemonic.split()

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
        """ Convert the keypair to dictionary with accountId, publicKey, secretPhrase, secretSeed, and ss58Address  
        """
        # Needs this incase the key is URI generated.
        if keypair.seed_hex == None:
            secret_seed = "0x" + "0" * 64 
        else:
            secret_seed = "0x" + keypair.seed_hex

        return {
            'accountId': keypair.public_key,
            'publicKey': keypair.public_key,
            'secretPhrase': keypair.mnemonic,
            'secretSeed': secret_seed,
            'ss58Address': keypair.ss58_address
        }
