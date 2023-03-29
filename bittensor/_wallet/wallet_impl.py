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
import sys
from types import SimpleNamespace
from typing import Optional, Union, List, Tuple, Dict, overload

import bittensor
from bittensor.utils import is_valid_bittensor_address_or_public_key
from substrateinterface import Keypair
from substrateinterface.base import is_valid_ss58_address
from termcolor import colored


def display_mnemonic_msg( keypair : Keypair, key_type : str ):
    """ Displaying the mnemonic and warning message to keep mnemonic safe
    """
    mnemonic = keypair.mnemonic
    mnemonic_green = colored(mnemonic, 'green')
    print (colored("\nIMPORTANT: Store this mnemonic in a secure (preferable offline place), as anyone " \
                "who has possesion of this mnemonic can use it to regenerate the key and access your tokens. \n", "red"))
    print ("The mnemonic to the new {} is:\n\n{}\n".format(key_type, mnemonic_green))
    print ("You can use the mnemonic to recreate the key in case it gets lost. The command to use to regenerate the key using this mnemonic is:")
    print("btcli regen_{} --mnemonic {}".format(key_type, mnemonic))
    print('')

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
        config: 'bittensor.Config' = None,
    ):
        r""" Init bittensor wallet object containing a hot and coldkey.
            Args:
                name (required=True, default='default):
                    The name of the wallet to unlock for running bittensor
                hotkey (required=True, default='default):
                    The name of hotkey used to running the miner.
                path (required=True, default='~/.bittensor/wallets/'):
                    The path to your bittensor wallets
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.wallet.config()
        """
        self.name = name
        self.path = path
        self.hotkey_str = hotkey
        self._hotkey = None
        self._coldkey = None
        self._coldkeypub = None
        self.config = config

    def __str__(self):
        return "Wallet ({}, {}, {})".format(self.name, self.hotkey_str, self.path)
    
    def __repr__(self):
        return self.__str__()

    def neuron(self, netuid: int) -> Optional['bittensor.NeuronInfo']:
        return self.get_neuron(netuid=netuid)

    def trust(self, netuid: int) -> Optional[float]:
        neuron = self.get_neuron(netuid=netuid)
        if neuron is None:
            return None
        return neuron.trust

    def validator_trust(self, netuid: int) -> Optional[float]:
        neuron = self.get_neuron(netuid=netuid)
        if neuron is None:
            return None
        return neuron.validator_trust

    def rank(self, netuid: int) -> Optional[float]:
        neuron = self.get_neuron(netuid=netuid)
        if neuron is None:
            return None
        return neuron.rank

    def incentive(self, netuid: int) -> Optional[float]:
        neuron = self.get_neuron(netuid=netuid)
        if neuron is None:
            return None
        return neuron.incentive

    def dividends(self, netuid: int) -> Optional[float]:
        neuron = self.get_neuron(netuid=netuid)
        if neuron is None:
            return None
        return neuron.dividends

    def consensus(self, netuid: int) -> Optional[float]:
        neuron = self.get_neuron(netuid=netuid)
        if neuron is None:
            return None
        return neuron.consensus

    def last_update(self, netuid: int) -> Optional[int]:
        neuron = self.get_neuron(netuid=netuid)
        if neuron is None:
            return None
        return neuron.last_update

    def validator_permit(self, netuid: int) -> Optional[bool]:
        neuron = self.get_neuron(netuid=netuid)
        if neuron is None:
            return None
        return neuron.validator_permit

    def weights(self, netuid: int) -> Optional[List[List[int]]]:
        neuron = self.get_neuron(netuid=netuid)
        if neuron is None:
            return None
        return neuron.weights

    def bonds(self, netuid: int) -> Optional[List[List[int]]]:
        neuron = self.get_neuron(netuid=netuid)
        if neuron is None:
            return None
        return neuron.bonds

    def uid(self, netuid: int) -> int:
        return self.get_uid(netuid=netuid)

    @property
    def stake(self) -> 'bittensor.Balance':
        return self.get_stake()

    @property
    def balance(self) -> 'bittensor.Balance':
        return self.get_balance()

    def is_registered( self, subtensor: Optional['bittensor.Subtensor'] = None, netuid: Optional[int] = None ) -> bool:
        """ Returns true if this wallet is registered.
            Args:
                subtensor( Optional['bittensor.Subtensor'] ):
                    Bittensor subtensor connection. Overrides with defaults if None.
                    Determines which network we check for registration.
                netuid ( Optional[int] ):
                    The network uid to check for registration.
                    Default is None, which checks any subnetwork.
            Return:
                is_registered (bool):
                    Is the wallet registered on the chain.
        """
        if subtensor == None: subtensor = bittensor.subtensor(self.config)
        if subtensor.network == 'nakamoto':
            neuron = subtensor.neuron_for_pubkey( ss58_hotkey = self.hotkey.ss58_address )
            return not neuron.is_null
        else:
            # default to finney
            if netuid == None:
                return subtensor.is_hotkey_registered_any( self.hotkey.ss58_address )
            else:
                return subtensor.is_hotkey_registered_on_subnet( self.hotkey.ss58_address, netuid = netuid )
        

    def get_neuron ( self, netuid: int, subtensor: Optional['bittensor.Subtensor'] = None ) -> Optional['bittensor.NeuronInfo'] :
        """ Returns this wallet's neuron information from subtensor.
            Args:
                netuid (int):
                    The network uid of the subnet to query.
                subtensor( Optional['bittensor.Subtensor'] ):
                    Bittensor subtensor connection. Overrides with defaults if None.
            Return:
                neuron (Union[ NeuronInfo, None ]):
                    neuron account on the chain or None if you are not registered.
        """
        if subtensor == None: subtensor = bittensor.subtensor()
        if not self.is_registered(netuid = netuid, subtensor=subtensor): 
            print(colored('This wallet is not registered. Call wallet.register() before this function.','red'))
            return None
        neuron = subtensor.neuron_for_wallet( self, netuid = netuid )
        return neuron

    def get_uid ( self, netuid: int, subtensor: Optional['bittensor.Subtensor'] = None ) -> int:
        """ Returns this wallet's hotkey uid or -1 if the hotkey is not subscribed.
            Args:
                netuid (int):
                    The network uid of the subnet to query.
                subtensor( Optional['bittensor.Subtensor'] ):
                    Bittensor subtensor connection. Overrides with defaults if None.
            Return:
                uid (int):
                    Network uid or -1 if you are not registered.
        """
        if subtensor == None: subtensor = bittensor.subtensor()
        if not self.is_registered(netuid = netuid, subtensor=subtensor): 
            print(colored('This wallet is not registered. Call wallet.register() before this function.','red'))
            return -1
        neuron = self.get_neuron(netuid = netuid, subtensor = subtensor)
        if neuron.is_null:
            return -1
        else:
            return neuron.uid

    def get_stake ( self, subtensor: Optional['bittensor.Subtensor'] = None ) -> 'bittensor.Balance':
        """ Returns this wallet's staking balance from passed subtensor connection.
            Args:
                subtensor( Optional['bittensor.Subtensor'] ):
                    Bittensor subtensor connection. Overrides with defaults if None.
            Return:
                balance (bittensor.utils.balance.Balance):
                    Stake account balance.
        """
        if subtensor == None: subtensor = bittensor.subtensor()
        stake = subtensor.get_stake_for_coldkey_and_hotkey( hotkey_ss58 = self.hotkey.ss58_address, coldkey_ss58 = self.coldkeypub.ss58_address )
        if not stake: # Not registered.
            print(colored('This wallet is not registered. Call wallet.register() before this function.','red'))
            return bittensor.Balance(0)
        
        return stake

    def get_balance( self, subtensor: Optional['bittensor.Subtensor'] = None ) -> 'bittensor.Balance':
        """ Returns this wallet's coldkey balance from passed subtensor connection.
            Args:
                subtensor( Optional['bittensor.Subtensor'] ):
                    Bittensor subtensor connection. Overrides with defaults if None.
            Return:
                balance (bittensor.utils.balance.Balance):
                    Coldkey balance.
        """
        if subtensor == None: subtensor = bittensor.subtensor()
        return subtensor.get_balance(address = self.coldkeypub.ss58_address)

    def reregister(
        self,
        netuid: int,
        subtensor: Optional['bittensor.Subtensor'] = None,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False
    ) -> Optional['bittensor.Wallet']:
        """ Re-register this wallet on the chain.
            Args:
                netuid (int):
                    The network uid of the subnet to register on.
                subtensor( Optional['bittensor.Subtensor'] ):
                    Bittensor subtensor connection. Overrides with defaults if None.
                wait_for_inclusion (bool):
                    if set, waits for the extrinsic to enter a block before returning true, 
                    or returns false if the extrinsic fails to enter the block within the timeout.   
                wait_for_finalization (bool):
                    if set, waits for the extrinsic to be finalized on the chain before returning true,
                    or returns false if the extrinsic fails to be finalized within the timeout.
                prompt (bool):
                    If true, the call waits for confirmation from the user before proceeding.
                
            Return:
                wallet (bittensor.Wallet):
                    This wallet.
        """
        if subtensor == None:
            subtensor = bittensor.subtensor()
        if not self.is_registered(netuid = netuid, subtensor=subtensor):
            # Check if the wallet should reregister
            if not self.config.wallet.get('reregister'):
                sys.exit(0)

            self.register(
                subtensor = subtensor,
                netuid = netuid,
                prompt = prompt,
                TPB = self.config.subtensor.register.cuda.get('TPB', None),
                update_interval = self.config.subtensor.register.cuda.get('update_interval', None),
                num_processes = self.config.subtensor.register.get('num_processes', None),
                cuda = self.config.subtensor.register.cuda.get('use_cuda', bittensor.defaults.subtensor.register.cuda.use_cuda),
                dev_id = self.config.subtensor.register.cuda.get('dev_id', None),
                wait_for_inclusion = wait_for_inclusion,
                wait_for_finalization = wait_for_finalization,
                output_in_place = self.config.subtensor.register.get('output_in_place', bittensor.defaults.subtensor.register.output_in_place),
                log_verbose = self.config.subtensor.register.get('verbose', bittensor.defaults.subtensor.register.verbose),
            )

        return self

    def register ( 
            self, 
            netuid: int,
            subtensor: Optional['bittensor.Subtensor'] = None, 
            wait_for_inclusion: bool = False,
            wait_for_finalization: bool = True,
            prompt: bool = False,
            max_allowed_attempts: int = 3,
            cuda: bool = False,
            dev_id: int = 0,
            TPB: int = 256,
            num_processes: Optional[int] = None,
            update_interval: Optional[int] = None,
            output_in_place: bool = True,
            log_verbose: bool = False,
        ) -> 'bittensor.Wallet':
        """ Registers the wallet to chain.
        Args:
            netuid (int):
                The network uid of the subnet to register on.
            subtensor( Optional['bittensor.Subtensor'] ):
                Bittensor subtensor connection. Overrides with defaults if None.
            wait_for_inclusion (bool):
                If set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                If set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
            max_allowed_attempts (int):
                Maximum number of attempts to register the wallet.
            cuda (bool):
                If true, the wallet should be registered on the cuda device.
            dev_id (int):
                The cuda device id.
            TPB (int):
                The number of threads per block (cuda).
            num_processes (int):
                The number of processes to use to register.
            update_interval (int):
                The number of nonces to solve between updates.
            output_in_place (bool):
                If true, the registration output is printed in-place.
            log_verbose (bool):
                If true, the registration output is more verbose.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """
        # Get chain connection.
        if subtensor == None: subtensor = bittensor.subtensor()
        subtensor.register(
            wallet = self,
            wait_for_inclusion = wait_for_inclusion,
            wait_for_finalization = wait_for_finalization,
            prompt=prompt, max_allowed_attempts=max_allowed_attempts,
            output_in_place = output_in_place,
            cuda=cuda,
            dev_id=dev_id,
            TPB=TPB,
            num_processes=num_processes,
            update_interval=update_interval,
            log_verbose=log_verbose,
            netuid = netuid,
        )
        
        return self

    def add_stake( self, 
        amount: Union[float, bittensor.Balance] = None, 
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        subtensor: Optional['bittensor.Subtensor'] = None,
        prompt: bool = False
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
                prompt (bool):
                    If true, the call waits for confirmation from the user before proceeding.
            Returns:
                success (bool):
                    flag is true if extrinsic was finalized or uncluded in the block. 
                    If we did not wait for finalization / inclusion, the response is true.
        """
        if subtensor == None: subtensor = bittensor.subtensor()
        return subtensor.add_stake( wallet = self, amount = amount, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization, prompt=prompt )

    def remove_stake( self, 
        amount: Union[float, bittensor.Balance] = None, 
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        subtensor: Optional['bittensor.Subtensor'] = None,
        prompt: bool = False,
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
                prompt (bool):
                    If true, the call waits for confirmation from the user before proceeding.
            Returns:
                success (bool):
                    flag is true if extrinsic was finalized or uncluded in the block. 
                    If we did not wait for finalization / inclusion, the response is true.
        """
        if subtensor == None: subtensor = bittensor.subtensor()
        return subtensor.unstake( wallet = self, amount = amount, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization, prompt=prompt )

    def transfer( 
        self, 
        dest:str,
        amount: Union[float, bittensor.Balance] , 
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        subtensor: Optional['bittensor.Subtensor'] = None,
        prompt: bool = False,
    ) -> bool:
        """ Transfers Tao from this wallet's coldkey to the destination address.
            Args:
                dest (`type`:str, required):
                    The destination address either encoded as a ss58 or ed255 public-key string of 
                    secondary account.
                amount (float, required):
                    amount of tao to transfer or a bittensor balance object.
                wait_for_inclusion (bool):
                    if set, waits for the extrinsic to enter a block before returning true, 
                    or returns false if the extrinsic fails to enter the block within the timeout.   
                wait_for_finalization (bool):
                    if set, waits for the extrinsic to be finalized on the chain before returning true,
                    or returns false if the extrinsic fails to be finalized within the timeout.
                subtensor( `bittensor.Subtensor` ):
                    Bittensor subtensor connection. Overrides with defaults if None.
                prompt (bool):
                    If true, the call waits for confirmation from the user before proceeding.
            Returns:
                success (bool):
                    flag is true if extrinsic was finalized or uncluded in the block. 
                    If we did not wait for finalization / inclusion, the response is true.
        """
        if subtensor == None: subtensor = bittensor.subtensor() 
        return subtensor.transfer( wallet = self, dest = dest, amount = amount, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization, prompt=prompt )

    def create_if_non_existent( self, coldkey_use_password:bool = True, hotkey_use_password:bool = False) -> 'Wallet':
        """ Checks for existing coldkeypub and hotkeys and creates them if non-existent.
        """
        return self.create(coldkey_use_password, hotkey_use_password)

    def create (self, coldkey_use_password:bool = True, hotkey_use_password:bool = False ) -> 'Wallet':
        """ Checks for existing coldkeypub and hotkeys and creates them if non-existent.
        """
        # ---- Setup Wallet. ----
        if not self.coldkey_file.exists_on_device() and not self.coldkeypub_file.exists_on_device():
            self.create_new_coldkey( n_words = 12, use_password = coldkey_use_password )
        if not self.hotkey_file.exists_on_device():
            self.create_new_hotkey( n_words = 12, use_password = hotkey_use_password )
        return self

    def recreate (self, coldkey_use_password:bool = True, hotkey_use_password:bool = False ) -> 'Wallet':
        """ Checks for existing coldkeypub and hotkeys and creates them if non-existent.
        """
        # ---- Setup Wallet. ----
        self.create_new_coldkey( n_words = 12, use_password = coldkey_use_password )
        self.create_new_hotkey( n_words = 12, use_password = hotkey_use_password )
        return self

    @property
    def hotkey_file(self) -> 'bittensor.Keyfile':

        wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
        hotkey_path = os.path.join(wallet_path, "hotkeys", self.hotkey_str)
        return bittensor.keyfile( path = hotkey_path )

    @property
    def coldkey_file(self) -> 'bittensor.Keyfile':
        wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
        coldkey_path = os.path.join(wallet_path, "coldkey")
        return bittensor.keyfile( path = coldkey_path )

    @property
    def coldkeypub_file(self) -> 'bittensor.Keyfile':
        wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
        coldkeypub_path = os.path.join(wallet_path, "coldkeypub.txt")
        return bittensor.Keyfile( path = coldkeypub_path )

    def set_hotkey(self, keypair: 'bittensor.Keypair', encrypt: bool = False, overwrite: bool = False) -> 'bittensor.Keyfile':
        self._hotkey = keypair
        self.hotkey_file.set_keypair( keypair, encrypt = encrypt, overwrite = overwrite )

    def set_coldkeypub(self, keypair: 'bittensor.Keypair', encrypt: bool = False, overwrite: bool = False) -> 'bittensor.Keyfile':
        self._coldkeypub = Keypair(ss58_address=keypair.ss58_address)
        self.coldkeypub_file.set_keypair( self._coldkeypub, encrypt = encrypt, overwrite = overwrite  )

    def set_coldkey(self, keypair: 'bittensor.Keypair', encrypt: bool = True, overwrite: bool = False) -> 'bittensor.Keyfile':
        self._coldkey = keypair
        self.coldkey_file.set_keypair( self._coldkey, encrypt = encrypt, overwrite = overwrite )

    def get_coldkey(self, password: str = None ) -> 'bittensor.Keypair':
        self.coldkey_file.get_keypair( password = password )

    def get_hotkey(self, password: str = None ) -> 'bittensor.Keypair':
        self.hotkey_file.get_keypair( password = password )

    def get_coldkeypub(self, password: str = None ) -> 'bittensor.Keypair':
        self.coldkeypub_file.get_keypair( password = password )

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
        keypair = Keypair.create_from_uri( uri )
        display_mnemonic_msg( keypair, "coldkey" )
        self.set_coldkey( keypair, encrypt = use_password, overwrite = overwrite)
        self.set_coldkeypub( keypair, overwrite = overwrite)
        return self

    def create_hotkey_from_uri( self, uri:str, use_password: bool = False, overwrite:bool = False) -> 'Wallet':  
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
        keypair = Keypair.create_from_uri( uri )
        display_mnemonic_msg( keypair, "hotkey" )
        self.set_hotkey( keypair, encrypt=use_password, overwrite = overwrite)
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
        mnemonic = Keypair.generate_mnemonic( n_words)
        keypair = Keypair.create_from_mnemonic(mnemonic)
        display_mnemonic_msg( keypair, "coldkey" )
        self.set_coldkey( keypair, encrypt = use_password, overwrite = overwrite)
        self.set_coldkeypub( keypair, overwrite = overwrite)
        return self

    def new_hotkey( self, n_words:int = 12, use_password: bool = False, overwrite:bool = False) -> 'Wallet':  
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

    def create_new_hotkey( self, n_words:int = 12, use_password: bool = False, overwrite:bool = False) -> 'Wallet':  
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
        mnemonic = Keypair.generate_mnemonic( n_words)
        keypair = Keypair.create_from_mnemonic(mnemonic)
        display_mnemonic_msg( keypair, "hotkey" )
        self.set_hotkey( keypair, encrypt=use_password, overwrite = overwrite)
        return self

    def regenerate_coldkeypub( self, ss58_address: Optional[str] = None, public_key: Optional[Union[str, bytes]] = None, overwrite: bool = False ) -> 'Wallet':
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
                wallet (bittensor.Wallet):
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
        self.set_coldkeypub( keypair, overwrite = overwrite)

        return self

    # Short name for regenerate_coldkeypub
    regen_coldkeypub = regenerate_coldkeypub

    @overload
    def regenerate_coldkey(
            self,
            mnemonic: Optional[Union[list, str]] = None,
            use_password: bool = True,
            overwrite: bool = False
        ) -> 'Wallet':
        ...

    @overload
    def regenerate_coldkey(
            self,
            seed: Optional[str] = None,
            use_password: bool = True,
            overwrite: bool = False
        ) -> 'Wallet':
        ...

    @overload
    def regenerate_coldkey(
            self,
            json: Optional[Tuple[Union[str, Dict], str]] = None,
            use_password: bool = True,
            overwrite: bool = False
        ) -> 'Wallet':
        ...


    def regenerate_coldkey(
            self,
            use_password: bool = True,
            overwrite: bool = False,
            **kwargs
        ) -> 'Wallet':
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
                wallet (bittensor.Wallet):
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
            display_mnemonic_msg( keypair, "coldkey" )
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
            overwrite: bool = False
        ) -> 'Wallet':
        ...

    @overload
    def regenerate_hotkey(
            self,
            seed: Optional[str] = None,
            use_password: bool = True,
            overwrite: bool = False
        ) -> 'Wallet':
        ...

    @overload
    def regenerate_hotkey(
            self,
            json: Optional[Tuple[Union[str, Dict], str]] = None,
            use_password: bool = True,
            overwrite: bool = False
        ) -> 'Wallet':
        ...

    def regenerate_hotkey(
            self,
            use_password: bool = True,
            overwrite: bool = False,
            **kwargs
        ) -> 'Wallet':
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
                wallet (bittensor.Wallet):
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
            display_mnemonic_msg( keypair, "hotkey" )
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
