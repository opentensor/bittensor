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
import random
import time

from munch import Munch
from loguru import logger
from typing import List, Tuple
from termcolor import colored

import bittensor
import bittensor.utils.networking as net
from bittensor.substrate.base import SubstrateInterface, Keypair
from bittensor.substrate.exceptions import SubstrateRequestException
from bittensor.utils.neurons import Neuron, Neurons
from bittensor.utils.balance import Balance

class Subtensor:
    """
    Handles interactions with the subtensor chain.
    """
    custom_type_registry = {
        "runtime_id": 2,
        "types": {
            "NeuronMetadataOf": {
                "type": "struct",
                "type_mapping": [["ip", "u128"], ["port", "u16"], ["ip_type", "u8"], ["uid", "u64"], ["modality", "u8"], ["hotkey", "AccountId"], ["coldkey", "AccountId"]]
            }
        }
    }

    def __init__(self, config: 'Munch' = None, wallet: 'bittensor.wallet.Wallet' = None):
        r""" Initializes a subtensor chain interface.
            Args:
                config (:obj:`Munch`, `optional`): 
                    metagraph.Metagraph.config()
                wallet (:obj:`bittensor.wallet.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
        """
        if config == None:
            config = Subtensor.build_config()
        self.config = config

        if wallet == None:
            wallet = bittensor.wallet.Wallet( self.config )
        self.wallet = wallet

        self.substrate = SubstrateInterface(
            address_type = 42,
            type_registry_preset='substrate-node-template',
            type_registry=self.custom_type_registry,
        )

    @staticmethod
    def build_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Subtensor.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        Subtensor.check_config(config)
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.wallet.Wallet.add_args( parser )
        try:
            parser.add_argument('--subtensor.network', default='akira', type=str, 
                                help='''The subtensor network flag. The likely choices are:
                                        -- akira (testing network)
                                        -- kusanagi (main network)
                                    If this option is set it overloads subtensor.chain_endpoint with 
                                    an entry point node from that network.
                                    ''')
            parser.add_argument('--subtensor.chain_endpoint', default=None, type=str, 
                                help='''The subtensor endpoint flag. If set, overrides the --network flag.
                                    ''')
        except:
            pass
        
    @staticmethod   
    def check_config(config: Munch):
        bittensor.wallet.Wallet.check_config( config )

    def endpoint_for_network( self, blacklist: List[str] = [] ) -> str:
        r""" Returns a chain endpoint based on config with black list. 
            Returns None if there are no available endpoints.
        Raises:
            endpoint (str):
                Websocket endpoint or None if there are none available.
        """

        # Chain endpoint overrides the --network flag.
        if self.config.subtensor.chain_endpoint != None:
            if self.config.subtensor.chain_endpoint in blacklist:
                return None
            else:
                return self.config.subtensor.chain_endpoint

        # Else defaults to networks.
        # TODO(const): this should probably make a DNS lookup.
        all_networks = ['akira', 'boltzmann', 'kusanagi', 'local']
        if self.config.subtensor.network == "akira":
            akira_available = [item for item in bittensor.__akira_entrypoints__ if item not in blacklist ]
            if len(akira_available) == 0:
                return None
            return random.choice( akira_available )

        elif self.config.subtensor.network == "boltzmann":
            boltzmann_available = [item for item in bittensor.__boltzmann_entrypoints__ if item not in blacklist ]
            if len(boltzmann_available) == 0:
                return None
            return random.choice( boltzmann_available )

        elif self.config.subtensor.network == "kusanagi":
            kusanagi_available = [item for item in bittensor.__kusanagi_entrypoints__ if item not in blacklist ]
            if len(kusanagi_available) == 0:
                return None
            return random.choice( kusanagi_available )

        elif self.config.subtensor.network == "local":
            local_available = [item for item in bittensor.__local_entrypoints__ if item not in blacklist ]
            if len(local_available) == 0:
                return None
            return random.choice( local_available )
            
        else:
            akira_available = [item for item in bittensor.__akira_entrypoints__ if item not in blacklist ]
            if len(akira_available) == 0:
                return None
            return random.choice( akira_available )

    def is_connected(self):
        r""" Returns the connection state as a boolean.
        Raises:
            success (bool):
                True is the websocket is connected to the chain endpoint.
        """
        return self.substrate.is_connected()

    def connect(self, timeout: int = 10, failure = True ) -> bool:
        r""" Attempts to connect the substrate interface backend. 
        If the connection fails, attemps another endpoint until a timeout.
        Args:
            timeout (int):
                Time to wait before subscription times out.
            failure (bool):
                This connection attempt raises an error an a failed attempt.
        Returns:
            success (bool):
                True on success. 
        """
        start_time = time.time()
        attempted_endpoints = []
        while True:
            def connection_error_message():
                logger.error('Check that your internet connection is working and the chain endpoints are available: {} ', attempted_endpoints)
                logger.error( '''   The subtensor chain endpoint should likely be one of the following choices:
                                        -- local -- (your locally running node)
                                        -- akira (testnet)
                                        -- kusanagi (mainnet)
                                    Or you may set the endpoint manually using the --subtensor.chain_endpoint flag 
                                    To connect run a local node (See: docs/running_a_validator.md)
                              ''')

            # ---- Get next endpoint ----
            ws_chain_endpoint = self.endpoint_for_network( blacklist = attempted_endpoints )
            if ws_chain_endpoint == None:
                logger.error('No more available endpoint for connection with subtensor.network: {} attempted: {}', self.config.subtensor.network, attempted_endpoints)
                connection_error_message()
                if failure:
                    raise RuntimeError('Unable to connect to network {}. Make sure your internet connection is stable and the network is properly set.'.format(self.config.subtensor.network))
                else:
                    return False
            attempted_endpoints.append(ws_chain_endpoint)

            # --- Attempt connection ----
            self.substrate.connect( ws_chain_endpoint )

            # ---- Success ----
            if self.substrate.is_connected():
                print(colored("Successfully connected to endpoint: {}".format(ws_chain_endpoint), 'green'))
                return True

            # ---- Timeout ----
            elif (time.time() - start_time) > timeout:
                print(colored("Error while subscribing to the chain endpoint {}".format(ws_chain_endpoint), 'red'))
                connection_error_message()
                if failure:
                    raise RuntimeError('Unable to connect to network {}. Make sure your internet connection is stable and the network is properly set.'.format(self.config.subtensor.network))
                else:
                    return False

            # ---- Loop ----
            else:
                time.sleep(1)

    def _check_connection(self) -> bool:
        if not self.is_connected():
            return self.connect()
        return True

    def is_subscribed(self, ip: str, port: int, modality: int, coldkey: str):
        uid = self.get_uid_for_pubkey(self.wallet.hotkey.public_key)
        if uid != None:
            neuron = self.get_neuron_for_uid( uid )
            if neuron['ip'] == net.ip_to_int(ip) and neuron['port'] == port:
                return True
            else:
                return False
        else:
            return False



    def subscribe(self, ip: str, port: int, modality: int, coldkeypub: str, wait_for_finalization=True) -> bool:
        r""" Subscribes the passed metadata to the substensor chain.
        """
        if not self._check_connection():
            return False

        if self.is_subscribed( ip, port, modality, coldkeypub ):
            print(colored('Subscribed with [ip: {}, port: {}, modality: {}, coldkey: {}]'.format(ip, port, modality, coldkeypub), 'green'))
            return True

        ip_as_int  = net.ip_to_int(ip)
        params = {
            'ip': ip_as_int,
            'port': port, 
            'ip_type': 4,
            'modality': modality,
            'coldkey': coldkeypub,
        }
        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='subscribe',
            call_params=params
        )

        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.wallet.hotkey)
        if wait_for_finalization:
            try:
                receipt = self.substrate.submit_extrinsic( extrinsic, wait_for_finalization = wait_for_finalization ) # Waiting for inclusion and other does not work
                
                if receipt.is_success:
                    print(colored('Subscribed with [ip: {}, port: {}, modality: {}, coldkey: {}]'.format(ip, port, modality, coldkeypub), 'green'))
                    return True
                else:
                    print(colored('Subscription failure with: error {}'.format(receipt.error_message), 'red'))
                    return False

            except SubstrateRequestException as e:
                print(colored("Failed to send subscribe extrinsic with error: {}".format(e), 'red'))
        else:
            return True

    def get_balance(self, address):
        r""" Returns the token balance for the passed ss58_address address
        Args:
            address (Substrate address format, default = 42):
                ss58 chain address.
        """
        self._check_connection()
        logger.debug("Getting balance for: {}", address)
        result = self.substrate.get_runtime_state(
            module='System',
            storage_function='Account',
            params=[address],
            block_hash=None
        )
        balance_info = result.get('result')
        if not balance_info:
            logger.debug("{} has no balance", address)
            return Balance(0)
        balance = balance_info['data']['free']
        logger.debug("{} has {} rao", address, balance)
        return Balance(balance)

    def add_stake(self, amount : Balance, hotkey_id: int):
        r""" Adds the specified amount of stake to passed hotkey uid.
        Args:
            amount (bittensor.utils.balance.Balance):
                amount to stake as bittensor balance
            hotkey_id (int):
                uid of hotkey to stake into.
        """
        self._check_connection()
        logger.info("Adding stake of {} rao from coldkey {} to hotkey {}", amount.rao, self.wallet.coldkey.public_key, hotkey_id)
        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='add_stake',
            call_params={
                'hotkey': hotkey_id,
                'ammount_staked': amount.rao
            }
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.wallet.coldkey)
        result = self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        return result

    def transfer(self, dest:str, amount: Balance):
        r""" Transfers funds from this wallet to the destination public key address
        Args:
            dest (str):
                destination public key address of reciever. 
            amount (bittensor.utils.balance.Balance):
                amount to stake as bittensor balance
        """
        self._check_connection()
        logger.debug("Requesting transfer of {}, from coldkey: {} to dest: {}", amount.rao, self.wallet.coldkey.public_key, dest)
        call = self.substrate.compose_call(
            call_module='Balances',
            call_function='transfer',
            call_params={
                'dest': dest, 
                'value': amount.rao
            }
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair = self.wallet.coldkey)
        result = self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        return result

    def unstake(self, amount : Balance, hotkey_id: int):
        r""" Removes stake into the wallet coldket from the specified hotkey uid.
        Args:
            amount (bittensor.utils.balance.Balance):
                amount to stake as bittensor balance
            hotkey_id (int):
                uid of hotkey to unstake from.
        """
        self._check_connection()
        logger.debug("Requesting unstake of {} rao for hotkey: {} to coldkey: {}", amount.rao, hotkey_id, self.wallet.coldkey.public_key)
        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='remove_stake',
            call_params={'ammount_unstaked': amount.rao, 'hotkey': hotkey_id}
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.wallet.coldkey)
        result = self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        return result

    def set_weights(self, destinations, values, wait_for_inclusion=False):
        r""" Sets the given weights and values on chain for wallet hotkey account.
        Args:
            destinations (List[int]):
                uint64 uids of destination neurons.
            values (List[int]):
                u32 max encoded floating point weights.
        """
        self._check_connection()
        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='set_weights',
            call_params = {'dests': destinations, 'weights': values}
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair = self.wallet.hotkey)
        result = self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=wait_for_inclusion)
        return result

    def get_current_block(self) -> int:
        r""" Returns the block number from the chain.
        Returns:
            block_number (int):
                Current chain blocknumber.
        """
        self._check_connection()
        return self.substrate.get_block_number(None)

    def get_active(self) -> List[int]:
        r""" Returns a list of uids one for each active peer on chain.
        Returns:
            active (List[int}):
                List of active peers.
        """
        self._check_connection()
        result =self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='Active',
        )
        return result

    def get_uid_for_pubkey(self, pubkey = str) -> int:
        r""" Returns the uid of the peer given passed public key string.
        Args:
            pubkey (str):
                String encoded public key.
        Returns:
            uid (int):
                uid of peer with hotkey equal to passed public key.
        """
        self._check_connection()
        result = self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='Active',
            params=[pubkey]
        )
        if result['result'] is None:
            return None
        return int(result['result'])

    def get_neuron_for_uid(self, uid:int):
        r""" Returns the neuron metadata of the peer with the passed uid.
        Args:
            uid (int):
                Uid to query for metadata.
        Returns:
            metadata (Dict):
                Dict in list form containing metadata of associated uid.
        """
        self._check_connection()
        result = self.substrate.get_runtime_state(
                module='SubtensorModule',
                storage_function='Neurons',
                params=[uid]
        )
        return result['result']
    
    def neurons(self, uid=None, decorator=False):
        r""" Returns a list of neuron from the chain. 
            Returns neuron objects if decorator is set to true.
        Args:
            uid (int):
                Uid to query for metadata.
            decorator(bool):
                specifiying if the returned data should be as neuron objects.
        Returns:
            neuron (List[Tuple[int, Dict]]):
                List of neuron objects.
        """
        self._check_connection()
        # Todo (parall4x, 23-12-2020) Get rid of this decorator flag. This should be refactored into something that returns Neuron objects only
        if uid:
            result = self.substrate.get_runtime_state(
                module='SubtensorModule',
                storage_function='Neurons',
                params=[uid]
            )
            return Neurons.from_list(result['result']) if decorator else result['result']

        else:
            neurons = self.substrate.iterate_map(
                module='SubtensorModule',
                storage_function='Neurons'
            )
            return Neurons.from_list(neurons) if decorator else neurons

    def get_stake_for_uid(self, uid) -> Balance:
        r""" Returns the staked token amount of the peer with the passed uid.
        Args:
            uid (int):
                Uid to query for metadata.
        Returns:
            stake (int):
                Amount of staked token.
        """
        self._check_connection()
        stake = self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='Stake',
            params = [uid]
        )
        result = stake['result']
        if not result:
            return Balance(0)
        return Balance(result)

    def weight_uids_for_uid(self, uid):
        r""" Returns the weight uids of the peer with the passed uid.
        Args:
            uid (int):
                Uid to query for metadata.
        Returns:
            weight_uids (List[int]):
                Weight uids for passed uid.
        """
        self._check_connection()
        logger.info('uid {}', uid)
        result = self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='WeightUids',
            params = [uid]
        )
        return result['result']

    def weight_vals_for_uid(self, uid):
        r""" Returns the weight vals of the peer with the passed uid.
        Args:
            uid (int):
                Uid to query for metadata.
        Returns:
            weight_vals (List[int]):
                Weight vals for passed uid.
        """
        self._check_connection()
        result = self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='WeightVals',
            params = [uid]
        )
        return result['result']

    def get_last_emit_data_for_uid(self, uid):
        r""" Returns the last emit of the peer with the passed uid.
        Args:
            uid (int):
                Uid to query for metadata.
        Returns:
            last_emit (int):
                Last emit block numebr
        """
        self._check_connection()
        result = self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='LastEmit',
            params = [uid]
        )
        return result['result']

    def get_last_emit_data(self):
        r""" Returns the last emit for all active peers.
        Returns:
            last_emits (List[int]):
                Last emit for peers on the chain.
        """
        self._check_connection()
        result = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='LastEmit'
        )
        return result

    def get_full_state(self) -> Tuple[List[int], List[Tuple[int, dict]], List[int], List[Tuple[int, List[int]]],  List[Tuple[int, List[int]]], List[Tuple[int, str]]]:
        self._check_connection()
        last_emit = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='LastEmit'
        )
        neurons = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='Neurons'
        )
        stake = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='Stake'
        )
        weight_vals = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='WeightVals'
        )
        weight_uids = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='WeightUids'
        )
        active = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='Active'
        )
        return last_emit, neurons, stake, weight_vals, weight_uids, active