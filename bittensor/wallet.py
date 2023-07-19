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

import os
import copy
import argparse
import bittensor
from termcolor import colored
from substrateinterface import Keypair
from typing import Optional, Union, List, Tuple, Dict, overload
from bittensor.utils import is_valid_bittensor_address_or_public_key


def display_mnemonic_msg(keypair: Keypair, key_type: str):
    """
    Display the mnemonic and a warning message to keep the mnemonic safe.
    
    Args:
        keypair (Keypair): Keypair object.
        key_type (str): Type of the key (coldkey or hotkey).
    """
    mnemonic = keypair.mnemonic
    mnemonic_green = colored(mnemonic, 'green')
    print(colored("\nIMPORTANT: Store this mnemonic in a secure (preferable offline place), as anyone "
                  "who has possession of this mnemonic can use it to regenerate the key and access your tokens. \n", "red"))
    print("The mnemonic to the new {} is:\n\n{}\n".format(key_type, mnemonic_green))
    print("You can use the mnemonic to recreate the key in case it gets lost. The command to use to regenerate the key using this mnemonic is:")
    print("btcli regen_{} --mnemonic {}".format(key_type, mnemonic))
    print('')


class wallet:
    """
    Bittensor wallet maintenance class. Each wallet contains a coldkey and a hotkey.
    The coldkey is the user's primary key for holding stake in their wallet
    and is the only way that users can access Tao. Coldkeys can hold tokens and should be encrypted on your device.
    The coldkey must be used to stake and unstake funds from a running node. The hotkey, on the other hand, is only used
    for subscribing and setting weights from running code. Hotkeys are linked to coldkeys through the metagraph.
    """

    @classmethod
    def config(cls) -> 'bittensor.config':
        """
        Get config from the argument parser.
        
        Returns:
            bittensor.config: Config object.
        """
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        return bittensor.config(parser)

    @classmethod
    def help(cls):
        """
        Print help to stdout.
        """
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        print(cls.__new__.__doc__)
        parser.print_help()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None):
        """
        Accept specific arguments from parser.
        
        Args:
            parser (argparse.ArgumentParser): Argument parser object.
            prefix (str): Argument prefix.
        """
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            default_name = os.getenv('BT_WALLET_NAME') or 'default'
            default_hotkey = os.getenv('BT_WALLET_NAME') or 'default'
            default_path = os.getenv('BT_WALLET_PATH') or '~/.bittensor/wallets/'
            parser.add_argument('--' + prefix_str + 'wallet.name', required=False, default=default_name,
                                help='The name of the wallet to unlock for running bittensor '
                                     '(name mock is reserved for mocking this wallet)')
            parser.add_argument('--' + prefix_str + 'wallet.hotkey', required=False, default=default_hotkey,
                                help="The name of the wallet's hotkey.")
            parser.add_argument('--' + prefix_str + 'wallet.path', required=False, default=default_path,
                                help='The path to your bittensor wallets')
        except argparse.ArgumentError as e:
            pass

    def __init__(
            self,
            name: str = None,
            hotkey: str = None,
            path: str = None,
            config: 'bittensor.config' = None,
    ):
        r"""
        Initialize the bittensor wallet object containing a hot and coldkey.
        
        Args:
            name (str, optional): The name of the wallet to unlock for running bittensor. Defaults to 'default'.
            hotkey (str, optional): The name of hotkey used to running the miner. Defaults to 'default'.
            path (str, optional): The path to your bittensor wallets. Defaults to '~/.bittensor/wallets/'.
            config (bittensor.config, optional): bittensor.wallet.config(). Defaults to None.
        """
        # Fill config from passed args using command line defaults.
        if config is None:
            config = wallet.config()
        self.config = copy.deepcopy(config)
        self.config.wallet.name = name or self.config.wallet.name
        self.config.wallet.hotkey = hotkey or self.config.wallet.hotkey
        self.config.wallet.path = path or self.config.wallet.path

        self.name = self.config.wallet.name
        self.path = self.config.wallet.path
        self.hotkey_str = self.config.wallet.hotkey

        self._hotkey = None
        self._coldkey = None
        self._coldkeypub = None


    def __str__(self):
        """
        Returns the string representation of the Wallet object.

        Returns:
            str: The string representation.
        """
        return "wallet({}, {}, {})".format(self.name, self.hotkey_str, self.path)

    def __repr__(self):
        """
        Returns the string representation of the Wallet object.

        Returns:
            str: The string representation.
        """
        return self.__str__()

    def create_if_non_existent(self, coldkey_use_password: bool = True, hotkey_use_password: bool = False) -> 'wallet':
        """
        Checks for existing coldkeypub and hotkeys and creates them if non-existent.

        Args:
            coldkey_use_password (bool, optional): Whether to use a password for coldkey. Defaults to True.
            hotkey_use_password (bool, optional): Whether to use a password for hotkey. Defaults to False.

        Returns:
            wallet: The Wallet object.
        """
        return self.create(coldkey_use_password, hotkey_use_password)

    def create(self, coldkey_use_password: bool = True, hotkey_use_password: bool = False) -> 'wallet':
        """
        Checks for existing coldkeypub and hotkeys and creates them if non-existent.

        Args:
            coldkey_use_password (bool, optional): Whether to use a password for coldkey. Defaults to True.
            hotkey_use_password (bool, optional): Whether to use a password for hotkey. Defaults to False.

        Returns:
            wallet: The Wallet object.
        """
        # ---- Setup Wallet. ----
        if not self.coldkey_file.exists_on_device() and not self.coldkeypub_file.exists_on_device():
            self.create_new_coldkey(n_words=12, use_password=coldkey_use_password)
        if not self.hotkey_file.exists_on_device():
            self.create_new_hotkey(n_words=12, use_password=hotkey_use_password)
        return self

    def recreate(self, coldkey_use_password: bool = True, hotkey_use_password: bool = False) -> 'wallet':
        """
        Checks for existing coldkeypub and hotkeys and creates them if non-existent.

        Args:
            coldkey_use_password (bool, optional): Whether to use a password for coldkey. Defaults to True.
            hotkey_use_password (bool, optional): Whether to use a password for hotkey. Defaults to False.

        Returns:
            wallet: The Wallet object.
        """
        # ---- Setup Wallet. ----
        self.create_new_coldkey(n_words=12, use_password=coldkey_use_password)
        self.create_new_hotkey(n_words=12, use_password=hotkey_use_password)
        return self

    @property
    def hotkey_file(self) -> 'bittensor.keyfile':
        """
        Property that returns the hotkey file.

        Returns:
            bittensor.keyfile: The hotkey file.
        """
        wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
        hotkey_path = os.path.join(wallet_path, "hotkeys", self.hotkey_str)
        return bittensor.keyfile(path=hotkey_path)

    @property
    def coldkey_file(self) -> 'bittensor.keyfile':
        """
        Property that returns the coldkey file.

        Returns:
            bittensor.keyfile: The coldkey file.
        """
        wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
        coldkey_path = os.path.join(wallet_path, "coldkey")
        return bittensor.keyfile(path=coldkey_path)

    @property
    def coldkeypub_file(self) -> 'bittensor.keyfile':
        """
        Property that returns the coldkeypub file.

        Returns:
            bittensor.keyfile: The coldkeypub file.
        """
        wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
        coldkeypub_path = os.path.join(wallet_path, "coldkeypub.txt")
        return bittensor.keyfile(path=coldkeypub_path)

    def set_hotkey(self, keypair: 'bittensor.Keypair', encrypt: bool = False, overwrite: bool = False) -> 'bittensor.keyfile':
        """
        Sets the hotkey for the wallet.

        Args:
            keypair (bittensor.Keypair): The hotkey keypair.
            encrypt (bool, optional): Whether to encrypt the hotkey. Defaults to False.
            overwrite (bool, optional): Whether to overwrite an existing hotkey. Defaults to False.

        Returns:
            bittensor.keyfile: The hotkey file.
        """
        self._hotkey = keypair
        self.hotkey_file.set_keypair(keypair, encrypt=encrypt, overwrite=overwrite)

    def set_coldkeypub(self, keypair: 'bittensor.Keypair', encrypt: bool = False, overwrite: bool = False) -> 'bittensor.keyfile':
        """
        Sets the coldkeypub for the wallet.

        Args:
            keypair (bittensor.Keypair): The coldkeypub keypair.
            encrypt (bool, optional): Whether to encrypt the coldkeypub. Defaults to False.
            overwrite (bool, optional): Whether to overwrite an existing coldkeypub. Defaults to False.

        Returns:
            bittensor.keyfile: The coldkeypub file.
        """
        self._coldkeypub = bittensor.Keypair(ss58_address=keypair.ss58_address)
        self.coldkeypub_file.set_keypair(self._coldkeypub, encrypt=encrypt, overwrite=overwrite)

    def set_coldkey(self, keypair: 'bittensor.Keypair', encrypt: bool = True, overwrite: bool = False) -> 'bittensor.keyfile':
        """
        Sets the coldkey for the wallet.

        Args:
            keypair (bittensor.Keypair): The coldkey keypair.
            encrypt (bool, optional): Whether to encrypt the coldkey. Defaults to True.
            overwrite (bool, optional): Whether to overwrite an existing coldkey. Defaults to False.

        Returns:
            bittensor.keyfile: The coldkey file.
        """
        self._coldkey = keypair
        self.coldkey_file.set_keypair(self._coldkey, encrypt=encrypt, overwrite=overwrite)

    def get_coldkey(self, password: str = None) -> 'bittensor.Keypair':
        """
        Gets the coldkey from the wallet.

        Args:
            password (str, optional): The password to decrypt the coldkey. Defaults to None.

        Returns:
            bittensor.Keypair: The coldkey keypair.
        """
        return self.coldkey_file.get_keypair(password=password)

    def get_hotkey(self, password: str = None) -> 'bittensor.Keypair':
        """
        Gets the hotkey from the wallet.

        Args:
            password (str, optional): The password to decrypt the hotkey. Defaults to None.

        Returns:
            bittensor.Keypair: The hotkey keypair.
        """
        return self.hotkey_file.get_keypair(password=password)

    def get_coldkeypub(self, password: str = None) -> 'bittensor.Keypair':
        """
        Gets the coldkeypub from the wallet.

        Args:
            password (str, optional): The password to decrypt the coldkeypub. Defaults to None.

        Returns:
            bittensor.Keypair: The coldkeypub keypair.
        """
        return self.coldkeypub_file.get_keypair(password=password)

    @property
    def hotkey(self) -> 'bittensor.Keypair':
        r""" Loads the hotkey from wallet.path/wallet.name/hotkeys/wallet.hotkey or raises an error.
            Returns:
                hotkey (Keypair):
                    hotkey loaded from config arguments.
            Raises:
                KeyFileError: Raised if the file is corrupt of non-existent.
                CryptoKeyError: Raised if the user enters an incorrec password for an encrypted keyfile.
        """
        if self._hotkey == None:
            self._hotkey = self.hotkey_file.keypair
        return self._hotkey

    @property
    def coldkey(self) -> 'bittensor.Keypair':
        r""" Loads the hotkey from wallet.path/wallet.name/coldkey or raises an error.
            Returns:
                coldkey (Keypair):
                    colkey loaded from config arguments.
            Raises:
                KeyFileError: Raised if the file is corrupt of non-existent.
                CryptoKeyError: Raised if the user enters an incorrec password for an encrypted keyfile.
        """
        if self._coldkey == None:
            self._coldkey = self.coldkey_file.keypair
        return self._coldkey

    @property
    def coldkeypub(self) -> 'bittensor.Keypair':
        r""" Loads the coldkeypub from wallet.path/wallet.name/coldkeypub.txt or raises an error.
            Returns:
                coldkeypub (Keypair):
                    colkeypub loaded from config arguments.
            Raises:
                KeyFileError: Raised if the file is corrupt of non-existent.
                CryptoKeyError: Raised if the user enters an incorrect password for an encrypted keyfile.
        """
        if self._coldkeypub == None:
            self._coldkeypub = self.coldkeypub_file.keypair
        return self._coldkeypub

    def create_coldkey_from_uri(self, uri:str, use_password: bool = True, overwrite:bool = False, suppress: bool = False) -> 'wallet':
        """ Creates coldkey from suri string, optionally encrypts it with the user's inputed password.
            Args:
                uri: (str, required):
                    URI string to use i.e. /Alice or /Bob
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional):
                    Will this operation overwrite the coldkey under the same path <wallet path>/<wallet name>/coldkey
            Returns:
                wallet (bittensor.wallet):
                    this object with newly created coldkey.
        """
        keypair = Keypair.create_from_uri( uri )
        if not suppress: display_mnemonic_msg( keypair, "coldkey" )
        self.set_coldkey( keypair, encrypt = use_password, overwrite = overwrite)
        self.set_coldkeypub( keypair, overwrite = overwrite)
        return self

    def create_hotkey_from_uri( self, uri:str, use_password: bool = False, overwrite:bool = False, suppress: bool = False ) -> 'wallet':
        """ Creates hotkey from suri string, optionally encrypts it with the user's inputed password.
            Args:
                uri: (str, required):
                    URI string to use i.e. /Alice or /Bob
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional):
                    Will this operation overwrite the hotkey under the same path <wallet path>/<wallet name>/hotkeys/<hotkey>
            Returns:
                wallet (bittensor.wallet):
                    this object with newly created hotkey.
        """
        keypair = Keypair.create_from_uri( uri )
        if not suppress: display_mnemonic_msg( keypair, "hotkey" )
        self.set_hotkey( keypair, encrypt=use_password, overwrite = overwrite)
        return self

    def new_coldkey( self, n_words:int = 12, use_password: bool = True, overwrite:bool = False, suppress: bool = False ) -> 'wallet':
        """ Creates a new coldkey, optionally encrypts it with the user's inputed password and saves to disk.
            Args:
                n_words: (int, optional):
                    Number of mnemonic words to use.
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional):
                    Will this operation overwrite the coldkey under the same path <wallet path>/<wallet name>/coldkey
            Returns:
                wallet (bittensor.wallet):
                    this object with newly created coldkey.
        """
        self.create_new_coldkey( n_words, use_password, overwrite, suppress)

    def create_new_coldkey( self, n_words:int = 12, use_password: bool = True, overwrite:bool = False, suppress: bool = False ) -> 'wallet':
        """ Creates a new coldkey, optionally encrypts it with the user's inputed password and saves to disk.
            Args:
                n_words: (int, optional):
                    Number of mnemonic words to use.
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional):
                    Will this operation overwrite the coldkey under the same path <wallet path>/<wallet name>/coldkey
            Returns:
                wallet (bittensor.wallet):
                    this object with newly created coldkey.
        """
        mnemonic = Keypair.generate_mnemonic( n_words)
        keypair = Keypair.create_from_mnemonic(mnemonic)
        if not suppress: display_mnemonic_msg( keypair, "coldkey" )
        self.set_coldkey( keypair, encrypt = use_password, overwrite = overwrite)
        self.set_coldkeypub( keypair, overwrite = overwrite)
        return self

    def new_hotkey( self, n_words:int = 12, use_password: bool = False, overwrite:bool = False, suppress: bool = False) -> 'wallet':
        """ Creates a new hotkey, optionally encrypts it with the user's inputed password and saves to disk.
            Args:
                n_words: (int, optional):
                    Number of mnemonic words to use.
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional):
                    Will this operation overwrite the hotkey under the same path <wallet path>/<wallet name>/hotkeys/<hotkey>
            Returns:
                wallet (bittensor.wallet):
                    this object with newly created hotkey.
        """
        self.create_new_hotkey( n_words, use_password, overwrite, suppress )

    def create_new_hotkey( self, n_words:int = 12, use_password: bool = False, overwrite:bool = False, suppress: bool = False ) -> 'wallet':
        """ Creates a new hotkey, optionally encrypts it with the user's inputed password and saves to disk.
            Args:
                n_words: (int, optional):
                    Number of mnemonic words to use.
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional):
                    Will this operation overwrite the hotkey under the same path <wallet path>/<wallet name>/hotkeys/<hotkey>
            Returns:
                wallet (bittensor.wallet):
                    this object with newly created hotkey.
        """
        mnemonic = Keypair.generate_mnemonic( n_words)
        keypair = Keypair.create_from_mnemonic(mnemonic)
        if not suppress: display_mnemonic_msg( keypair, "hotkey" )
        self.set_hotkey( keypair, encrypt=use_password, overwrite = overwrite)
        return self

    def regenerate_coldkeypub( self, ss58_address: Optional[str] = None, public_key: Optional[Union[str, bytes]] = None, overwrite: bool = False, suppress: bool = False ) -> 'wallet':
        """ Regenerates the coldkeypub from passed ss58_address or public_key and saves the file
               Requires either ss58_address or public_key to be passed.
            Args:
                ss58_address: (str, optional):
                    Address as ss58 string.
                public_key: (str | bytes, optional):
                    Public key as hex string or bytes.
                overwrite (bool, optional) (default: False):
                    Will this operation overwrite the coldkeypub (if exists) under the same path <wallet path>/<wallet name>/coldkeypub
            Returns:
                wallet (bittensor.wallet):
                    newly re-generated Wallet with coldkeypub.

        """
        if ss58_address is None and public_key is None:
            raise ValueError("Either ss58_address or public_key must be passed")

        if not is_valid_bittensor_address_or_public_key( ss58_address if ss58_address is not None else public_key ):
            raise ValueError(f"Invalid {'ss58_address' if ss58_address is not None else 'public_key'}")

        if ss58_address is not None:
            ss58_format = bittensor.utils.get_ss58_format( ss58_address )
            keypair = Keypair(ss58_address=ss58_address, public_key=public_key, ss58_format=ss58_format)
        else:
            keypair = Keypair(ss58_address=ss58_address, public_key=public_key, ss58_format=bittensor.__ss58_format__)

        # No need to encrypt the public key
        self.set_coldkeypub( keypair, overwrite = overwrite )

        return self

    # Short name for regenerate_coldkeypub
    regen_coldkeypub = regenerate_coldkeypub

    @overload
    def regenerate_coldkey(
            self,
            mnemonic: Optional[Union[list, str]] = None,
            use_password: bool = True,
            overwrite: bool = False,
            suppress: bool = False,
        ) -> 'wallet':
        ...

    @overload
    def regenerate_coldkey(
            self,
            seed: Optional[str] = None,
            use_password: bool = True,
            overwrite: bool = False,
            suppress: bool = False,
        ) -> 'wallet':
        ...

    @overload
    def regenerate_coldkey(
            self,
            json: Optional[Tuple[Union[str, Dict], str]] = None,
            use_password: bool = True,
            overwrite: bool = False,
            suppress: bool = False,
        ) -> 'wallet':
        ...


    def regenerate_coldkey(
            self,
            use_password: bool = True,
            overwrite: bool = False,
            suppress: bool = False,
            **kwargs
        ) -> 'wallet':
        """ Regenerates the coldkey from passed mnemonic, seed, or json encrypts it with the user's password and saves the file
            Args:
                mnemonic: (Union[list, str], optional):
                    Key mnemonic as list of words or string space separated words.
                seed: (str, optional):
                    Seed as hex string.
                json: (Tuple[Union[str, Dict], str], optional):
                    Restore from encrypted JSON backup as (json_data: Union[str, Dict], passphrase: str)
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional):
                    Will this operation overwrite the coldkey under the same path <wallet path>/<wallet name>/coldkey
            Returns:
                wallet (bittensor.wallet):
                    this object with newly created coldkey.

            Note: uses priority order: mnemonic > seed > json
        """
        if len(kwargs) == 0:
            raise ValueError("Must pass either mnemonic, seed, or json")

        # Get from kwargs
        mnemonic = kwargs.get('mnemonic', None)
        seed = kwargs.get('seed', None)
        json = kwargs.get('json', None)

        if mnemonic is None and seed is None and json is None:
            raise ValueError("Must pass either mnemonic, seed, or json")
        if mnemonic is not None:
            if isinstance( mnemonic, str): mnemonic = mnemonic.split()
            if len(mnemonic) not in [12,15,18,21,24]:
                raise ValueError("Mnemonic has invalid size. This should be 12,15,18,21 or 24 words")
            keypair = Keypair.create_from_mnemonic(" ".join(mnemonic), ss58_format=bittensor.__ss58_format__ )
            if not suppress: display_mnemonic_msg( keypair, "coldkey" )
        elif seed is not None:
            keypair = Keypair.create_from_seed(seed, ss58_format=bittensor.__ss58_format__ )
        else:
            # json is not None
            if not isinstance(json, tuple) or len(json) != 2 or not isinstance(json[0], (str, dict)) or not isinstance(json[1], str):
                raise ValueError("json must be a tuple of (json_data: str | Dict, passphrase: str)")

            json_data, passphrase = json
            keypair = Keypair.create_from_encrypted_json( json_data, passphrase, ss58_format=bittensor.__ss58_format__ )

        self.set_coldkey( keypair, encrypt = use_password, overwrite = overwrite)
        self.set_coldkeypub( keypair, overwrite = overwrite)
        return self

    # Short name for regenerate_coldkey
    regen_coldkey = regenerate_coldkey

    @overload
    def regenerate_hotkey(
            self,
            mnemonic: Optional[Union[list, str]] = None,
            use_password: bool = True,
            overwrite: bool = False,
            suppress: bool = False,
        ) -> 'wallet':
        ...

    @overload
    def regenerate_hotkey(
            self,
            seed: Optional[str] = None,
            use_password: bool = True,
            overwrite: bool = False,
            suppress: bool = False,
        ) -> 'wallet':
        ...

    @overload
    def regenerate_hotkey(
            self,
            json: Optional[Tuple[Union[str, Dict], str]] = None,
            use_password: bool = True,
            overwrite: bool = False,
            suppress: bool = False,
        ) -> 'wallet':
        ...

    def regenerate_hotkey(
            self,
            use_password: bool = True,
            overwrite: bool = False,
            suppress: bool = False,
            **kwargs
        ) -> 'wallet':
        """ Regenerates the hotkey from passed mnemonic, encrypts it with the user's password and save the file
            Args:
                mnemonic: (Union[list, str], optional):
                    Key mnemonic as list of words or string space separated words.
                seed: (str, optional):
                    Seed as hex string.
                json: (Tuple[Union[str, Dict], str], optional):
                    Restore from encrypted JSON backup as (json_data: Union[str, Dict], passphrase: str)
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional):
                    Will this operation overwrite the hotkey under the same path <wallet path>/<wallet name>/hotkeys/<hotkey>
            Returns:
                wallet (bittensor.wallet):
                    this object with newly created hotkey.
        """
        if len(kwargs) == 0:
            raise ValueError("Must pass either mnemonic, seed, or json")

        # Get from kwargs
        mnemonic = kwargs.get('mnemonic', None)
        seed = kwargs.get('seed', None)
        json = kwargs.get('json', None)

        if mnemonic is None and seed is None and json is None:
            raise ValueError("Must pass either mnemonic, seed, or json")
        if mnemonic is not None:
            if isinstance( mnemonic, str): mnemonic = mnemonic.split()
            if len(mnemonic) not in [12,15,18,21,24]:
                raise ValueError("Mnemonic has invalid size. This should be 12,15,18,21 or 24 words")
            keypair = Keypair.create_from_mnemonic(" ".join(mnemonic), ss58_format=bittensor.__ss58_format__ )
            if not suppress: display_mnemonic_msg( keypair, "hotkey")
        elif seed is not None:
            keypair = Keypair.create_from_seed(seed, ss58_format=bittensor.__ss58_format__ )
        else:
            # json is not None
            if not isinstance(json, tuple) or len(json) != 2 or not isinstance(json[0], (str, dict)) or not isinstance(json[1], str):
                raise ValueError("json must be a tuple of (json_data: str | Dict, passphrase: str)")

            json_data, passphrase = json
            keypair = Keypair.create_from_encrypted_json( json_data, passphrase, ss58_format=bittensor.__ss58_format__ )


        self.set_hotkey( keypair, encrypt=use_password, overwrite = overwrite)
        return self

    # Short name for regenerate_hotkey
    regen_hotkey = regenerate_hotkey


#########
# Tests #
########


import time
import pytest
import unittest
import bittensor
from unittest.mock import patch, MagicMock

class TestWallet(unittest.TestCase):
    def setUp(self):
        self.mock_wallet = bittensor.wallet( name = f'mock-{str(time.time())}', hotkey = f'mock-{str(time.time())}', path = '/tmp/tests_wallets/do_not_use' )
        self.mock_wallet.create_new_coldkey( use_password = False, overwrite = True, suppress=True ) 
        self.mock_wallet.create_new_hotkey( use_password = False, overwrite = True, suppress=True ) 

    def test_regen_coldkeypub_from_ss58_addr(self):
        """Test the `regenerate_coldkeypub` method of the wallet class, which regenerates the cold key pair from an SS58 address.
        It checks whether the `set_coldkeypub` method is called with the expected arguments, and verifies that the generated key pair's SS58 address matches the input SS58 address.
        It also tests the behavior when an invalid SS58 address is provided, raising a `ValueError` as expected.
        """
        ss58_address = "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        with patch.object(self.mock_wallet, 'set_coldkeypub') as mock_set_coldkeypub:
            self.mock_wallet.regenerate_coldkeypub( ss58_address=ss58_address, overwrite = True, suppress=True )

            mock_set_coldkeypub.assert_called_once()
            keypair: bittensor.Keypair = mock_set_coldkeypub.call_args_list[0][0][0]
            self.assertEqual(keypair.ss58_address, ss58_address)

        ss58_address_bad = "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zx" # 1 character short
        with pytest.raises(ValueError):
            self.mock_wallet.regenerate_coldkeypub(ss58_address=ss58_address_bad, overwrite = True, suppress=True )

    def test_regen_coldkeypub_from_hex_pubkey_str(self):
        """Test the `regenerate_coldkeypub` method of the wallet class, which regenerates the cold key pair from a hex public key string.
        It checks whether the `set_coldkeypub` method is called with the expected arguments, and verifies that the generated key pair's public key matches the input public key.
        It also tests the behavior when an invalid public key string is provided, raising a `ValueError` as expected.
        """
        pubkey_str = "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f"
        with patch.object(self.mock_wallet, 'set_coldkeypub') as mock_set_coldkeypub:
            self.mock_wallet.regenerate_coldkeypub(public_key=pubkey_str, overwrite = True, suppress=True )

            mock_set_coldkeypub.assert_called_once()
            keypair: bittensor.Keypair = mock_set_coldkeypub.call_args_list[0][0][0]
            self.assertEqual('0x' + keypair.public_key.hex(), pubkey_str)

        pubkey_str_bad = "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512" # 1 character short
        with pytest.raises(ValueError):
            self.mock_wallet.regenerate_coldkeypub(ss58_address=pubkey_str_bad, overwrite = True, suppress=True )

    def test_regen_coldkeypub_from_hex_pubkey_bytes(self):
        """Test the `regenerate_coldkeypub` method of the wallet class, which regenerates the cold key pair from a hex public key byte string.
        It checks whether the `set_coldkeypub` method is called with the expected arguments, and verifies that the generated key pair's public key matches the input public key.
        """
        pubkey_str = "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f"
        pubkey_bytes = bytes.fromhex(pubkey_str[2:]) # Remove 0x from beginning
        with patch.object(self.mock_wallet, 'set_coldkeypub') as mock_set_coldkeypub:
            self.mock_wallet.regenerate_coldkeypub(public_key=pubkey_bytes, overwrite = True, suppress=True )

            mock_set_coldkeypub.assert_called_once()
            keypair: bittensor.Keypair = mock_set_coldkeypub.call_args_list[0][0][0]
            self.assertEqual(keypair.public_key, pubkey_bytes)

    def test_regen_coldkeypub_no_pubkey(self):
        """Test the `regenerate_coldkeypub` method of the wallet class when no public key is provided.
        It verifies that a `ValueError` is raised when neither a public key nor an SS58 address is provided.
        """
        with pytest.raises(ValueError):
            # Must provide either public_key or ss58_address
            self.mock_wallet.regenerate_coldkeypub(ss58_address=None, public_key=None, overwrite = True, suppress=True )

    def test_regen_coldkey_from_hex_seed_str(self):
        """Test the `regenerate_coldkey` method of the wallet class, which regenerates the cold key pair from a hex seed string.
        It checks whether the `set_coldkey` method is called with the expected arguments, and verifies that the generated key pair's seed and SS58 address match the input seed and the expected SS58 address.
        It also tests the behavior when an invalid seed string is provided, raising a `ValueError` as expected.
        """
        ss58_addr = "5D5cwd8DX6ij7nouVcoxDuWtJfiR1BnzCkiBVTt7DU8ft5Ta"
        seed_str = "0x659c024d5be809000d0d93fe378cfde020846150b01c49a201fc2a02041f7636"
        with patch.object(self.mock_wallet, 'set_coldkey') as mock_set_coldkey:
            self.mock_wallet.regenerate_coldkey(seed=seed_str, overwrite = True, suppress=True)

            mock_set_coldkey.assert_called_once()
            keypair: bittensor.Keypair = mock_set_coldkey.call_args_list[0][0][0]
            self.assertRegex(keypair.seed_hex if isinstance(keypair.seed_hex, str) else keypair.seed_hex.hex(), rf'(0x|){seed_str[2:]}')
            self.assertEqual(keypair.ss58_address, ss58_addr) # Check that the ss58 address is correct

        seed_str_bad = "0x659c024d5be809000d0d93fe378cfde020846150b01c49a201fc2a02041f763" # 1 character short
        with pytest.raises(ValueError):
            self.mock_wallet.regenerate_coldkey(seed=seed_str_bad, overwrite = True, suppress=True )

    def test_regen_hotkey_from_hex_seed_str(self):
        """Test the `regenerate_coldkey` method of the wallet class, which regenerates the cold key pair from a hex seed string.
        It checks whether the `set_coldkey` method is called with the expected arguments, and verifies that the generated key pair's seed and SS58 address match the input seed and the expected SS58 address.
        It also tests the behavior when an invalid seed string is provided, raising a `ValueError` as expected.
        """
        ss58_addr = "5D5cwd8DX6ij7nouVcoxDuWtJfiR1BnzCkiBVTt7DU8ft5Ta"
        seed_str = "0x659c024d5be809000d0d93fe378cfde020846150b01c49a201fc2a02041f7636"
        with patch.object(self.mock_wallet, 'set_hotkey') as mock_set_hotkey:
            self.mock_wallet.regenerate_hotkey(seed=seed_str, overwrite = True, suppress=True )

            mock_set_hotkey.assert_called_once()
            keypair: bittensor.Keypair = mock_set_hotkey.call_args_list[0][0][0]
            self.assertRegex(keypair.seed_hex if isinstance(keypair.seed_hex, str) else keypair.seed_hex.hex(), rf'(0x|){seed_str[2:]}')
            self.assertEqual(keypair.ss58_address, ss58_addr) # Check that the ss58 address is correct

        seed_str_bad = "0x659c024d5be809000d0d93fe378cfde020846150b01c49a201fc2a02041f763" # 1 character short
        with pytest.raises(ValueError):
            self.mock_wallet.regenerate_hotkey(seed=seed_str_bad, overwrite = True, suppress=True )

if __name__ == '__main__':
    unittest.main()