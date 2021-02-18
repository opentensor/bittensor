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

from munch import Munch
from loguru import logger
from typing import List, Tuple

import bittensor
import bittensor.utils.networking as net
from bittensor.substrate.base import SubstrateInterface, Keypair
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
            url = self.config.subtensor.chain_endpoint,
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
            parser.add_argument('--subtensor.chain_endpoint', default=None, type=str, 
                                help='''The subtensor chain endpoint. The likely choices are:
                                        -- localhost:9944 -- (your locally running node)
                                        -- feynman.akira.bittensor.com:9944 (testnet)
                                        -- feynman.kusanagi.bittensor.com:12345 (mainnet)
                                    If subtensor.network is set it is overloaded by subtensor.network.
                                    ''')
            parser.add_argument('--subtensor.network', default=None, type=str, 
                                help='''The subtensor network flag. The likely choices are:
                                        -- akira (testing network)
                                        -- kusanagi (main network)
                                    If this option is set it overloads subtensor.chain_endpoint with 
                                    an entry point node from that network.
                                    ''')
        except:
            pass
        
    @staticmethod   
    def check_config(config: Munch):
        bittensor.wallet.Wallet.check_config( config )

        # Neither are set, default to akira.
        if config.subtensor.network == None and config.subtensor.chain_endpoint == None:
            logger.info('Defaulting to network: akira')
            config.subtensor.network = 'akira'

        # Switch based on network config item. 
        if config.subtensor.network != None:
            all_networks = ['akira', 'boltzmann', 'kusanagi']
            assert config.subtensor.network in all_networks, 'metagraph.network == {} not one of {}'.format(config.subtensor.network, all_networks)
            if config.subtensor.network == "akira":
                config.subtensor.chain_endpoint = random.choice(bittensor.__akira_entrypoints__)
            elif config.subtensor.network == "boltzmann":
                config.subtensor.chain_endpoint = random.choice(bittensor.__boltzmann_entrypoints__)
            elif config.subtensor.network == "kusanagi":
                config.subtensor.chain_endpoint = random.choice(bittensor.__kusanagi_entrypoints__)
            else:
                raise ValueError('metagraph.network == {} not one of {}'.format(config.subtensor.network, all_networks))

        # The chain endpoint it set.
        elif config.subtensor.chain_endpoint != None:
            all_entrypoints = bittensor.__akira_entrypoints__ + bittensor.__boltzmann_entrypoints__ + bittensor.__kusanagi_entrypoints__
            if not config.subtensor.chain_endpoint in all_entrypoints:
                logger.info('metagraph.chain_endpoint == {}, NOTE: not one of {}', config.subtensor.chain_endpoint, all_entrypoints)

    def is_connected(self):
        r""" Returns the connection state as a boolean.
        Raises:
            success (bool):
                True is the websocket is connected to the chain endpoint.
        """
        return self.substrate.is_connected()

    def connect(self, timeout: int = 10) -> bool:
        r""" Attempts to connect the backend websocket to the subtensor chain.
        Args:
            timeout (int):
                Time to wait before subscription times out.
        Returns:
            success (bool):
                True on success. 
        """
        self.substrate.connect()
        return self.substrate.is_connected()

    def subscribe(self, ip: str, port: int, modality: int, coldkey: str):
        r""" Subscribes the passed metadata to the substensor chain.
        """
        ip_as_int  = net.ip_to_int(ip)
        params = {
            'ip': ip_as_int,
            'port': port, 
            'ip_type': 4,
            'modality': modality,
            'coldkey': coldkey
        }
        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='subscribe',
            call_params=params
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.wallet.hotkey.public_key)
        self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)  # Waiting for inclusion and other does not work

    def get_balance(self, address):
        r""" Returns the token balance for the passed ss58_address address
        Args:
            address (Substrate address format, default = 42):
                ss58 chain address.
        """
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
        return self.substrate.get_block_number(None)

    def get_active(self) -> List[int]:
        r""" Returns a list of uids one for each active peer on chain.
        Returns:
            active (List[int}):
                List of active peers.
        """
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
        result = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='LastEmit'
        )
        return result

    def get_full_state(self) -> Tuple[List[int], List[Tuple[int, dict]], List[int], List[Tuple[int, List[int]]],  List[Tuple[int, List[int]]], List[Tuple[int, str]]]:
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



    # ConnectSuccess = 1
    # ConnectUnknownError = 2
    # ConnectTimeout = 3
    # def connect(self, timeout:int) -> Tuple[int, str]:
    #     r""" Synchronous: Connects to the chain.
    #     Args:
    #         timeout (int):
    #             Time to wait before connecting times out.
    #     Returns:
    #         see: _try_async_connect
    #     """
    #     # ---- Try Connection ----
    #     try:
    #         code, message = self._try_async_connect(timeout)

    #         if code == Metagraph.ConnectSuccess:
    #             logger.info('Successfully connected to chain endpoint: {}', self.config.subtensor.chain_endpoint)
    #             return code, message

    #         elif code == Metagraph.ConnectUnknownError:
    #             logger.error('Connection threw an unknown error: {}', message)
    #             return code, message

    #         elif code == Metagraph.ConnectTimeout:
    #             logger.error('Connection timeout {}', message)
    #             return code, message

    #     except Exception as e:
    #         logger.error('Connection threw an uncaught error {}', e)
    #         return Metagraph.ConnectUnknownError, e

    # def connect(self, timeout: int) -> Tuple[int, str]:
    #     r""" Makes connection attempts to the chain, continuing to attempt until timeout.

    #     Args:
    #         timeout (int):
    #             Time to wait before connecting times out.
    #     Raises:
    #         code (ENUM) {}
    #             ConnectSuccess:
    #                 Raised when the connection is a success before the timeout

    #             ConnectUnknownError:
    #                 UnknownError during connecting to chain.

    #             Connectimeout:
    #                 Raised when the attempted connection fails after timeout.
    #         }
    #         message:
    #             Message associated with code. 
    #     """
    #     # ---- Make Chain connection attempt  ----
    #     start_time = time.time()
    #     while True:
    #         # ---- Make connection call.
    #         try:
    #             self.subtensor.connect()
    #         except Exception as e:
    #             return Metagraph.ConnectUnknownError, e
            
    #         # ---- Wait for connection future to reture, or timeout.
    #         is_connected = self.subtensor.is_connected()
    #         try:
    #             asyncio.wait_for(is_connected, timeout=timeout)
    #         except asyncio.TimeoutError:
    #             return Metagraph.ConnectTimeout, "Timeout"

    #         # ---- Return on success.
    #         if is_connected:
    #             return Metagraph.ConnectSuccess, "Success"

    #         # ---- Retry or timeout.
    #         elif (time.time() - start_time) > timeout:
    #             return Metagraph.ConnectTimeout, "Timeout"
    #         else:
    #             asyncio.sleep(1)
    #             continue

    # SubscribeSuccess = 1
    # SubscribeUnknownError = 2
    # SubscribeTimeout = 3
    # SubscribeNotConnected = 4
    # def subscribe(self, timeout) -> Tuple[int, str]:
    #     r""" Syncronous: Makes a subscribe request to the chain. Waits for subscription inclusion or returns False
    #     Returns:
    #         see: _try_async_subscribe
    #     """
    #     # ---- Try Subscription ----
    #     try:
    #         code, message = self._try_subscribe(timeout)

    #         if code == Metagraph.SubscribeSuccess:
    #             logger.info('Successfully subcribed with: {}', self.metadata)
    #             return code, message

    #         elif code == Metagraph.SubscribeNotConnected:
    #             logger.error('Subscription failed because you are not connected to a chain endpoint, call metagraph.connect() first')
    #             return code, message

    #         elif code == Metagraph.SubscribeUnknownError:
    #             logger.error('Subscription threw an unknown error: {}', message)
    #             return code, message

    #         elif code == Metagraph.SubscribeTimeout:
    #             logger.error('Subscription timeout {}', message)
    #             return code, message

    #     except Exception as e:
    #         logger.error('Subscription threw an uncaught error {}', e)
    #         return Metagraph.SubscribeUnknownError, e
        
    # def _try_subscribe(self, timeout: int):
    #     r""" Makes subscription attempts to the chain, continuing to attempt until timeout and finally waiting for inclusion.

    #     Args:
    #         timeout (int):
    #             Time to wait before subscription times out.

    #     Raises:
    #         code (ENUM) {}
    #             SubscribeSuccess:
    #                 Raised when the subscription is a success before the timeout

    #             SubscribeUnknownError:
    #                 UnknownError during subscription.

    #             SubscribeTimeout:
    #                 Raised when the attempted subscription fails after timeout.

    #             SubscribeNotConnected:
    #                 Raised if a subscription is attempted while before metagraph.connect is called.
    #                 Mush call metagraph.connect() before metagraph.subscribe()
    #         }
    #         message:
    #             Message associated with code. 
    #     """
    #     # --- Check that we are already connected to the chain.
    #     is_connected = self.subtensor.is_connected()
    #     try:
    #         asyncio.wait_for(is_connected, timeout = 10)
    #     except asyncio.TimeoutError:
    #         return Metagraph.SubscribeNotConnected, "Not connected"
    #     if not is_connected:
    #         return Metagraph.SubscribeNotConnected, "Not connected"

    #     # ---- Make Subscription transaction ----
    #     logger.info("Subscribing to subtensor")
    #     main_start_time = time.time()
    #     while True:

    #         subscribe_start_time = time.time()
    #         try:
    #             self.subtensor.subscribe(self.config.axon.external_ip, self.config.axon.external_port, bittensor.proto.Modality.TEXT, self.wallet.coldkey)

    #         except Exception as e:
    #             if (time.time() - subscribe_start_time) > 8:
    #                 # --- Timeout during emit call ----
    #                 message = "Timed-out with Unknown Error while trying to make the subscription call. With last exception {}".format(e)
    #                 return Metagraph.SubscribeUnknownError, message

    #             else:
    #                 # --- Wait for inclusion, no error.
    #                 logger.trace('Error while attempting subscription {}', e)
    #                 continue

    #         # ---- Wait for inclusion ----
    #         check_start_time = time.time()
    #         while True:
    #             try:
    #                 # ---- Request info from chain ----
    #                 self.uid = self.subtensor.get_uid_for_pubkey(self.wallet.hotkey.public_key)
    #             except Exception as e:
    #                 # ---- Catch errors in request ----
    #                 message = "Subscription threw an unknown exception {}".format(e)
    #                 return Metagraph.SubscribeUnknownError, message

    #             if self.uid != None:
    #                 # ---- Request info from chain ----
    #                 self.metadata = self.subtensor.neurons(self.uid)
    #                 if not self.metadata:
    #                     return Metagraph.SubscribeUnknownError, "Critical error: There no metadata returned"

    #                 # ---- Subscription was a success ----
    #                 return Metagraph.SubscribeSuccess, "Subscription success"

    #             elif time.time() - check_start_time > 8:
    #                 break

    #             else:
    #                 # ---- wait -----
    #                 asyncio.sleep(1)
            
    #         if time.time() - main_start_time > 90:
    #             return Metagraph.SubscribeTimeout, "Timeout occured while trying to subscribe to the chain, potentially the chain is recieving too many subsribe requests at this time."

                 

    #     # ---- ?! WaT ?! ----
    #     logger.critical('Should not get here')
    #     return Metagraph.SubscribeUnknownError, 'Should not get here'


    # EmitSuccess = 1
    # EmitValueError = 2
    # EmitUnknownError = 3
    # EmitTimeoutError = 4
    # EmitTimeoutError = 5
    # EmitResultUnknown = 6
    # EmitNoOp = 7
    # def async_emit(self, weights: torch.FloatTensor, wait_for_inclusion = False, timeout = 12) -> Tuple[int, str]:
    #     r""" Calls _try_async_emit, logs and returns results based on code. Only fails on an uncaught Exception.
        
    #     Args:
    #         weights: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
    #             Weights to set on chain.
    #         wait_for_inclusion: (bool):
    #             If true, the call waits for block-inclusion before continuing or throws error after timeout.
    #         timeout: (int, default = 12 sec):
    #             Time to wait for inclusion before raising a caught error.

    #     Returns:
    #         see: _try_async_emit
    #     """
    #     # --- Try emit, optionally wait ----
    #     try:
    #         code, message = self._try_async_emit(weights, wait_for_inclusion, timeout)
    #         if code == Metagraph.EmitSuccess:
    #             # ---- Emit was a success. ----
    #             logger.info("Successful emission.")

    #         elif code == Metagraph.EmitValueError:
    #             # ---- Passed weights were incorrect ----
    #             logger.warning("Value error during emission: {}", message)

    #         elif code == Metagraph.EmitUnknownError:
    #             # ---- Unknown error ----
    #             logger.error("Unknown error during emission: {}", message)

    #         elif code == Metagraph.EmitTimeoutError:
    #             # ---- Timeout while waiting for inclusion ----
    #             logger.warning("Emission timeout after {} seconds with error {}", timeout, message)

    #         elif code == Metagraph.EmitResultUnknown:
    #             # ---- Did not wait, result unknown ----
    #             logger.trace("Emit results unknown.")

    #         elif code == Metagraph.EmitNoOp:
    #             # ---- Emit is a NoOp ----
    #             logger.info("When trying to set weights on chain. Weights are unchanged, nothing to emit.")

    #     except Exception as e:
    #         # ---- Unknown error, raises error again. Should never get here ----
    #         logger.error("Unknown Error during emission {}", e)
    #         raise e

    #     return code, message


    # def _try_emit(self, weights: torch.FloatTensor, wait_for_inclusion = False, timeout = 12) -> Tuple[int, str]:
    #     r""" Makes emit checks, emits to chain, and raises one of the following errors.
    #         Args:
    #             weights: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
    #                 Weights to set on chain.

    #             wait_for_inclusion: (:obj:`bool`):
    #                 If true, the call waits for block-inclusion before continuing or throws error after timeout.

    #             timeout: (:obj:`int`, default = 12 sec):
    #                 Time to wait for inclusion before raising a caught error.

    #         Returns:
    #             code (:obj:`ENUM`) {
    #                 EmitSuccess (:obj:`ENUM`):
    #                     Raised when try_async_emit emits weights successfully with known result.

    #                 EmitNoOp (:obj:`ENUM`):
    #                     Raised when calling emit does not change weights on chain.

    #                 EmitUnknownError (:obj:`ENUM`):
    #                     UnknownError during emit.

    #                 EmitValueError (:obj:`ENUM`):
    #                     Raised during emission when passed weights are not properly set.

    #                 EmitTimeoutError (:obj:`ENUM`):
    #                     Raised during emission during a timeout.

    #                 EmitResultUnknown (:obj:`ENUM`):
    #                     Called when an emit step end without a known result, for instance, 
    #                     if the user has wait_for_inclusion = False.
    #             }
    #             message:
    #                 Message associated with code.

    #     """
    #     # --- Check type ----
    #     if not isinstance(weights, torch.Tensor):
    #         message = "Error trying to set weights on chain. Got weights type {}, but weights must be of type {}".format(type(weights), torch.Tensor)
    #         return Metagraph.EmitValueError, message
        
    #     # --- Check nan ---
    #     if torch.any(weights.isnan()).item():
    #         message = "Error trying to set weight on chain. Got nan values {}".format(weights)
    #         return Metagraph.EmitValueError, message

    #     # ---- Convert weights to list ----
    #     weights = [float(w) for w in weights.tolist()]

    #     # ---- Check length > 0 ----
    #     if len(weights) == 0:
    #         message = "Error tyring to set weight on china. Got a length 0 set of values, must be at least length 1."
    #         return Metagraph.EmitValueError, message

    #     # ---- Check length ----
    #     if len(weights) != self.state.n:
    #         message = "Error trying to set weights on chain. Got length {}, but the length must match the number of neurons in metagraph.neurons {}".format(len(weights), self.state.n)
    #         return Metagraph.EmitValueError, message

    #     # ---- Check approximate sum ----
    #     sum_weights = sum(weights)
    #     epsilon = 0.001
    #     if abs(1.0 - sum_weights) > epsilon:
    #         message = "Error trying to set weights on chain. Got {} for sum, but passed weights must sum to 1 ".format(len(sum_weights), self.state.n)
    #         return Metagraph.EmitValueError, message

    #     # ---- Check min ----
    #     min_weights = min(weights)
    #     if min_weights < 0.0:
    #         message = "Error trying to set weights on chain. Got min value {} but values must be in range [0,1]".format(min_weights)
    #         return Metagraph.EmitValueError, message

    #     # ---- Check max ----
    #     max_weights = max(weights)
    #     if max_weights > 1.0:
    #         message = "Error trying to set weights on chain. Got max value {} but values must be in range [0,1]".format(max_weights)
    #         return Metagraph.EmitValueError, message

    #     # ---- Convert Weights to int-vals and pubkeys ----
    #     try:
    #         weight_uids, weight_vals = self.convert_weights_to_emit(weights)
    #     except Exception as e:
    #         message = "Unknown error when converting weights to ints with weights {} and error {}".format(weights, e)
    #         return Metagraph.EmitUnknownError, message

    #     # ---- Check sum ----
    #     weight_sum = sum(weight_vals)
    #     if weight_sum != MAX_INT_WEIGHT:
    #         message = "Error trying to set weights on chain. Converted weights do not sum to {} with weights_vals {}".format(MAX_INT_WEIGHT, weight_vals)
    #         return Metagraph.EmitValueError, message

    #     # ---- Check NO-OP ----
    #     if self._are_set_on_chain(weight_vals, weight_uids):
    #         message = "When trying to set weights on chain. Weights are unchanged, nothing to emit."
    #         return Metagraph.EmitNoOp, message

    #     # ---- Emit ----
    #     start_time = time.time()
    #     while True:
    #         try:
    #             # --- Make emission call ----
    #             logger.debug('Emit -> {} {}', weight_uids, weight_vals)
    #             self.subtensor.set_weights(weight_uids, weight_vals)
    #             break

    #         except Exception as e:
    #             logger.trace('Emit error {}', e)

    #             if not wait_for_inclusion:
    #                 # --- No wait, and error during emit call ----
    #                 message = "Error raised during call to emit {}".format( e )
    #                 return Metagraph.EmitUnknownError, message
                
    #             elif (time.time() - start_time) > timeout:
    #                 # --- Timeout during emit call ----
    #                 message = "Timed-out with unknown Error while trying to make the emit call. With last exception {}".format(e)
    #                 return Metagraph.EmitUnknownError, message

    #             else:
    #                 # --- Wait for inclusion, no error.
    #                 logger.info('retry emit...')
    #                 asyncio.sleep(3) # To avoid ddos-ing the chain.
    #                 continue

    #     # --- Wait for inclusion ----
    #     if not wait_for_inclusion:
    #         message = "Emit ended but we don't know if weights were set on chain"
    #         return Metagraph.EmitResultUnknown, message

    #     else:
    #         while True:
    #             did_emit = self._are_set_on_chain(weight_uids, weight_vals)

    #             if not did_emit and (time.time() - start_time) > timeout:
    #                 # ---- Emit caused timeout  -----
    #                 message = "Timed-out while waiting for inclusion."
    #                 return Metagraph.EmitTimeoutError, message

    #             elif not did_emit:
    #                 # ---- Did not emit, but waiting for inclusion -----
    #                 asyncio.sleep(3)
    #                 continue

    #             else:
    #                 # --- Did emit, return latest chain weights ----
    #                 messages = "Successful emission"
    #                 return Metagraph.EmitSuccess, messages

    #     message = 'Should never get here'
    #     logger.critical(message)
    #     return Metagraph.EmitUnknownError, message

    # def _are_set_on_chain(self, weight_uids, weight_vals) -> bool:
    #     r""" Returns true if the passed key and vals are set on chain.
    #     """
    #     cmap = {}
    #     chain_uids = self.subtensor.weight_uids_for_uid(self.uid)
    #     chain_vals = self.subtensor.weight_vals_for_uid(self.uid)
    #     if chain_uids != None and chain_vals != None:
    #         n_same = 0
    #         for uid, val in list(zip(chain_uids, chain_vals)):
    #             cmap[uid] = val
    #         for uid, val in list(zip(weight_uids, weight_vals)):
    #             if uid in cmap:
    #                 if cmap[uid] == val:
    #                     n_same += 1
    #         if n_same == len(weight_vals):
    #             return True
    #         else:
    #             return False
    #     else:
    #         return False 


    # def convert_weights_to_emit(self, weights: List[float]) -> Tuple[List[str], List[int]]:
    #     r""" Converts weights into integer u32 representation that sum to MAX_INT_WEIGHT.

    #          Returns:
    #             keys (:obj:`List[str]`):
    #                 List of pubkeys associated with each weight from vals.
    #             vals (:obj:`List[int]`):
    #             List of u32 integer representations of floating point weights.

    #     """
    #     remainder = MAX_INT_WEIGHT
    #     weight_vals = []
    #     weight_uids = []
    #     pos_self_uid = -1
    #     for i, val in enumerate(weights):
    #         int_val = int(float(val) * int(MAX_INT_WEIGHT)) # convert to int representation.
    #         remainder -= int_val
    #         uid_i = self.state.uids.tolist()[i]

    #         # ---- Fix remainders and overflows ----
    #         if remainder < 0:
    #             int_val = int_val + remainder
    #             remainder = 0

    #         if i == (len(weights) -1) and remainder > 0: # last item.
    #             int_val += remainder
    #             remainder = 0

    #         # Do not add zero values. 
    #         if int_val != 0:
    #             weight_vals.append( int_val ) # int weights sum to MAX_INT_WEIGHT.
    #             weight_uids.append( uid_i ) # Gets the uid at this index

    #         if uid_i == self.uid:
    #             pos_self_uid = i

    #     # Places the self weight in the first position if it exists
    #     if pos_self_uid != -1 and len(weight_uids) > 1:
    #         weight_uids.insert(0, weight_uids.pop(pos_self_uid))
    #         weight_vals.insert(0, weight_vals.pop(pos_self_uid))
    #     return weight_uids, weight_vals
