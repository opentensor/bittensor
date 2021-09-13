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
import random
import time
from substrateinterface.base import Keypair
import torch
from multiprocessing import Process

from typing import List, Tuple, Dict

import bittensor
import bittensor.utils.networking as net
import bittensor.utils.weight_utils as weight_utils
from substrateinterface import SubstrateInterface
from substrateinterface.exceptions import SubstrateRequestException
from bittensor.utils.balance import Balance

from loguru import logger
logger = logger.opt(colors=True)

class Subtensor:
    """
    Handles interactions with the subtensor chain.
    """
    def __init__( 
        self, 
        substrate: 'SubstrateInterface',
        network: str,
        chain_endpoint: str
    ):
        r""" Initializes a subtensor chain interface.
            Args:
                substrate (:obj:`SubstrateInterface`, `required`): 
                    substrate websocket client.
                network (default='akatsuki', type=str)
                    The subtensor network flag. The likely choices are:
                            -- kusanagi (test network)
                            -- akatsuki (main network)
                    If this option is set it overloads subtensor.chain_endpoint with 
                    an entry point node from that network.
                chain_endpoint (default=None, type=str)
                    The subtensor endpoint flag. If set, overrides the network argument.
        """
        self.network = network
        self.chain_endpoint = chain_endpoint
        self.substrate = substrate

    def __str__(self) -> str:
        if self.network == self.chain_endpoint:
            return "Subtensor({})".format( self.chain_endpoint )
        else:
            return "Subtensor({}, {})".format( self.network, self.chain_endpoint )

    def __repr__(self) -> str:
        return self.__str__()
  
    def endpoint_for_network( 
            self,
            blacklist: List[str] = [] 
        ) -> str:
        r""" Returns a chain endpoint based on self.network.
            Returns None if there are no available endpoints.
        """

        # Chain endpoint overrides the --network flag.
        if self.chain_endpoint != None:
            if self.chain_endpoint in blacklist:
                return None
            else:
                return self.chain_endpoint

        # Else defaults to networks.
        # TODO(const): this should probably make a DNS lookup.
        if self.network == "akatsuki":
            akatsuki_available = [item for item in bittensor.__akatsuki_entrypoints__ if item not in blacklist ]
            if len(akatsuki_available) == 0:
                return None
            return random.choice (akatsuki_available)

        elif self.network == "kusanagi":
            kusanagi_available = [item for item in bittensor.__kusanagi_entrypoints__ if item not in blacklist ]
            if len(kusanagi_available) == 0:
                return None
            return random.choice( kusanagi_available )

        elif self.network == "local":
            local_available = [item for item in bittensor.__local_entrypoints__ if item not in blacklist ]
            if len(local_available) == 0:
                return None
            return random.choice( local_available )
            
        else:
            kusanagi_available = [item for item in bittensor.__kusanagi_entrypoints__ if item not in blacklist ]
            if len(kusanagi_available) == 0:
                return None
            return random.choice( kusanagi_available )


    def connect( self, timeout: int = 10, failure = True ) -> bool:
        attempted_endpoints = []
        while True:
            def connection_error_message():
                print('''
Check that your internet connection is working and the chain endpoints are available: <blue>{}</blue>
The subtensor.network should likely be one of the following choices:
    -- local - (your locally running node)
    -- kusanagi - (testnet)
    -- akatsuki - (mainnet)
Or you may set the endpoint manually using the --subtensor.chain_endpoint flag 
To run a local node (See: docs/running_a_validator.md) \n
                              '''.format( attempted_endpoints) )

            # ---- Get next endpoint ----
            ws_chain_endpoint = self.endpoint_for_network( blacklist = attempted_endpoints )
            if ws_chain_endpoint == None:
                logger.error("No more endpoints available for subtensor.network: <blue>{}</blue>, attempted: <blue>{}</blue>".format(self.network, attempted_endpoints))
                connection_error_message()
                if failure:
                    logger.critical('Unable to connect to network:<blue>{}</blue>.\nMake sure your internet connection is stable and the network is properly set.'.format(self.network))
                else:
                    return False
            attempted_endpoints.append(ws_chain_endpoint)

            # --- Attempt connection ----
            try:
                with self.substrate:
                    logger.success("Network:".ljust(20) + "<blue>{}</blue>", self.network)
                    logger.success("Endpoint:".ljust(20) + "<blue>{}</blue>", ws_chain_endpoint)
                    return True
            
            except Exception:
                logger.error( "Error while connecting to network:<blue>{}</blue> at endpoint: <blue>{}</blue>".format(self.network, ws_chain_endpoint))
                connection_error_message()
                if failure:
                    raise RuntimeError('Unable to connect to network:<blue>{}</blue>.\nMake sure your internet connection is stable and the network is properly set.'.format(self.network))
                else:
                    return False

    def is_subscribed(self, wallet: 'bittensor.wallet', ip: str, port: int) -> bool:
        r""" Returns true if the bittensor endpoint is already subscribed with the wallet and metadata.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            ip (str):
                endpoint host port i.e. 192.122.31.4
            port (int):
                endpoint port number i.e. 9221
        """
        uid = self.get_uid_for_pubkey( wallet.hotkey.public_key )
        if uid != None:
            neuron = self.get_neuron_for_uid( uid )
            if neuron['ip'] == net.ip_to_int(ip) and neuron['port'] == port:
                return True
            else:
                logger.warning('IP does not match, resubscribing with new IP')
                return False
        else:
            logger.warning('No previous subscription found with this hotkey. Initalizing subscription..')
            return False

    def subscribe(
            self, 
            wallet: 'bittensor.wallet',
            ip: str, 
            port: int, 
            modality: int, 
            wait_for_inclusion: bool = False,
            wait_for_finalization = True,
        ) -> bool:
        r""" Subscribes an bittensor endpoint to the substensor chain.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            ip (str):
                endpoint host port i.e. 192.122.31.4
            port (int):
                endpoint port number i.e. 9221
            modality (int):
                int encoded endpoint modality i.e 0 for TEXT
            wait_for_inclusion (bool):
                if set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                if set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """

        if self.is_subscribed( wallet, ip, port ):
            logger.success( "Subscribed with".ljust(20) + '<blue>ip: {}, port: {}, modality: {}, hotkey: {}, coldkey: {}</blue>'.format(ip, port, modality, wallet.hotkey.public_key, wallet.coldkeypub ))
            return True

        ip_as_int  = net.ip_to_int(ip)
        ip_version = net.ip_version(ip)

        # TODO(const): subscribe with version too.
        params = {
            'ip': ip_as_int,
            'port': port, 
            'ip_type': ip_version,
            'modality': modality,
            'coldkey': wallet.coldkeypub,
        }
        
        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module='SubtensorModule',
                call_function='subscribe',
                call_params=params
            )
            # TODO (const): hotkey should be an argument here not assumed. Either that or the coldkey pub should also be assumed.
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.hotkey)
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
            if wait_for_inclusion or wait_for_finalization:
                response.process_events()
                if response.is_success:
                    bittensor.logging.success( 'Subscribed', '<blue>ip: {}, port: {}, modality: {}, hotkey: {}, coldkey: {}</blue>'.format(ip, port, modality, wallet.hotkey.public_key, wallet.coldkeypub ))
                    return True
                else:
                    bittensor.logging.warning( 'Failed to Subscribe:', str(response.error_message) )
                    return False
            else:
                return True

    def add_stake(
            self, 
            wallet: 'bittensor.wallet',
            amount: Balance, 
            wait_for_inclusion: bool = False,
            wait_for_finalization: bool = False,
        ) -> bool:
        r""" Adds the specified amount of stake to passed hotkey uid.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            amount (bittensor.utils.balance.Balance):
                amount to stake as bittensor balance
            wait_for_inclusion (bool):
                if set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                if set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """
        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module='SubtensorModule', 
                call_function='add_stake',
                call_params={
                    'hotkey': wallet.hotkey.public_key,
                    'ammount_staked': amount.rao
                }
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
            if wait_for_inclusion or wait_for_finalization:
                response.process_events()
                if response.is_success:
                    bittensor.logging.success( 'Staked', str(amount.tao) )
                    return True
                else:
                    bittensor.logging.warning( 'Failed to stake', str(response.error_message) )
                    return False
            else:
                return True

    def transfer(
            self, 
            wallet: 'bittensor.wallet',
            dest:str, 
            amount: Balance, 
            wait_for_inclusion: bool = False,
            wait_for_finalization: bool = False,
        ) -> bool:
        r""" Transfers funds from this wallet to the destination public key address
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            dest (str):
                destination public key address of reciever. 
            amount (bittensor.utils.balance.Balance):
                amount to stake as bittensor balance
            wait_for_inclusion (bool):
                if set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                if set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """
        
        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module='Balances',
                call_function='transfer',
                call_params={
                    'dest': dest, 
                    'value': amount.rao
                }
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
            if wait_for_inclusion or wait_for_finalization:
                response.process_events()
                if response.is_success:
                    bittensor.logging.success( 'Transfered', str(amount) + '->' + str(dest) )
                    return True
                else:
                    bittensor.logging.warning( 'Failed to transfer:', str(response.error_message) )
                    return False
            else:
                return True

    def unstake(
            self, 
            wallet: 'bittensor.wallet',
            amount: Balance, 
            wait_for_inclusion:bool = False, 
            wait_for_finalization:bool = False,
            timeout: int = 3 * bittensor.__blocktime__,
        ) -> bool:
        r""" Removes stake into the wallet coldkey from the specified hotkey uid.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            amount (bittensor.utils.balance.Balance):
                amount to stake as bittensor balance
            wait_for_inclusion (bool):
                if set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                if set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """
    
        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module='SubtensorModule', 
                call_function='remove_stake',
                call_params={
                    'hotkey': wallet.hotkey.public_key,
                    'ammount_unstaked': amount.rao
                }
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
            if wait_for_inclusion or wait_for_finalization:
                response.process_events()
                if response.is_success:
                    bittensor.logging.success( 'Unstaked', str(amount.tao) )
                    return True
                else:
                    bittensor.logging.warning( 'Failed to unstake:', str(response.error_message) )
                    return False
            else:
                return True

    def set_weights(
            self, 
            wallet: 'bittensor.wallet',
            uids: torch.LongTensor,
            weights: torch.FloatTensor,
            wait_for_inclusion:bool = False,
            wait_for_finalization:bool = False,
        ) -> bool:
        r""" Sets the given weights and values on chain for wallet hotkey account.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            uids (torch.LongTensor):
                uint64 uids of destination neurons.
            weights (torch.FloatTensor):
                weights to set which must floats and correspond to the passed uids.
            wait_for_inclusion (bool):
                if set, waits for the extrinsic to enter a block before returning true,
                or returns false if the extrinsic fails to enter the block within the timeout.
            wait_for_finalization (bool):
                if set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            timeout (int):
                time that this call waits for either finalization of inclusion.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block.
                If we did not wait for finalization / inclusion, the response is true.
        """
        bittensor.logging.success( 'Setting weights', str(list(zip(uids.tolist(), weights.tolist()))))
        weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit( uids, weights )
        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module='SubtensorModule',
                call_function='set_weights',
                call_params = {'dests': weight_uids, 'weights': weight_vals}
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.hotkey )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
            if wait_for_inclusion or wait_for_finalization:
                response.process_events()
                if response.is_success:
                    bittensor.logging.success( prefix = 'Weights Set', sufix = '<green>True</green>')
                    return True
                else:
                    bittensor.logging.warning(  prefix = 'Weights Set', sufix = '<green>False: </green>' + str(response.error_message) )
                    return False
            else:
                bittensor.logging.warning( prefix = 'Weights Set', sufix = '<yellow>Unknown (non-waiting)</yellow>')
                return True

    def get_balance(self, address: str, block: int = None) -> Balance:
        r""" Returns the token balance for the passed ss58_address address
        Args:
            address (Substrate address format, default = 42):
                ss58 chain address.
        Return:
            balance (bittensor.utils.balance.Balance):
                account balance
        """
        with self.substrate as substrate:
            result = substrate.get_runtime_state(
                module='System',
                storage_function='Account',
                params=[address],
                block_hash = None if block == None else substrate.get_block_hash( block )
            )
            balance_info = result.get('result')
            if not balance_info:
                return Balance(0)
            balance = balance_info['data']['free']
            return Balance(balance)

    def get_current_block(self) -> int:
        r""" Returns the current block number on the chain.
        Returns:
            block_number (int):
                Current chain blocknumber.
        """
        
        with self.substrate as substrate:
            return substrate.get_block_number(None)


    def get_balances(self, block: int = None) -> Dict[str, float]:
        with self.substrate as substrate:
            result = substrate.iterate_map (
                module='System',
                storage_function='Account',
                block_hash = None if block == None else substrate.get_block_hash( block )
            )
            return_dict = {}
            for r in result:
                balance = float( r[1]['data']['free'] ) / float(1000000000)
                return_dict[r[0]] = balance
            return return_dict

    def get_active(self, block: int = None) -> List[Tuple[str, int]]:
        r""" Returns a list of (public key, uid) pairs one for each active peer on chain.
        Args:
            blcok (int, default = None):
                Retrieve data at this block.
        Returns:
            active (List[Tuple[str, int]]):
                List of active peers.
        """
        with self.substrate as substrate:
            result =  substrate.iterate_map (
                module='SubtensorModule',
                storage_function='Active',
                block_hash = None if block == None else substrate.get_block_hash( block )
            )
            return result

    def get_stake(self, block: int = None) -> List[Tuple[int, int]]:
        r""" Returns a list of (uid, stake) pairs one for each active peer on chain.
        Returns:
            stake (List[Tuple[int, int]]):
                List of stake values.
        """
        
        with self.substrate as substrate:
            result = substrate.iterate_map (
                module='SubtensorModule',
                storage_function='Stake',
                block_hash = None if block == None else substrate.get_block_hash( block )
            )
            return result

    def get_last_emit(self, block: int = None) -> List[Tuple[int, int]]:
        r""" Returns a list of (uid, last emit) pairs for each active peer on chain.
        Returns:
            last_emit (List[Tuple[int, int]]):
                List of last emit values.
        """
        with self.substrate as substrate:
            result = substrate.iterate_map (
                module='SubtensorModule',
                storage_function='LastEmit',
                block_hash = None if block == None else substrate.get_block_hash( block )
            )
            return result

    def get_weight_vals(self, block: int = None) -> List[Tuple[int, List[int]]]:
        r""" Returns a list of (uid, weight vals) pairs for each active peer on chain.
        Returns:
            weight_vals (List[Tuple[int, List[int]]]):
                List of weight val pairs.
        """
        with self.substrate as substrate:
            result = substrate.iterate_map (
                module='SubtensorModule',
                storage_function='WeightVals',
                block_hash = None if block == None else substrate.get_block_hash( block )
            )
            return result

    def get_weight_uids(self, block: int = None) -> List[Tuple[int, int]]:
        r""" Returns a list of (uid, weight uids) pairs for each active peer on chain.
        Returns:
            weight_uids (List[Tuple[int, List[int]]]):
                List of weight uid pairs
        """
        with self.substrate as substrate:
            result = substrate.iterate_map (
                module='SubtensorModule',
                storage_function='WeightUids',
                block_hash = None if block == None else substrate.get_block_hash( block )
            )
            return result

    def neurons(self, block: int = None) -> List[Tuple[int, dict]]: 
        r""" Returns a list of neuron from the chain. 
        Returns:
            neuron (List[Tuple[int, dict]]):
                List of neuron objects.
        """
        with self.substrate as substrate:
            neurons = substrate.iterate_map (
                module='SubtensorModule',
                storage_function='Neurons',
                block_hash = None if block == None else substrate.get_block_hash( block )
            )
            return neurons

    def get_uid_for_pubkey(self, pubkey = str, block: int = None) -> int: 
        r""" Returns the uid of the peer given passed public key string.
        Args:
            pubkey (str):
                String encoded public key.
        Returns:
            uid (int):
                uid of peer with hotkey equal to passed public key.
        """
        with self.substrate as substrate:
            result = substrate.get_runtime_state(
                module='SubtensorModule',
                storage_function='Active',
                params=[pubkey],
                block_hash = None if block == None else substrate.get_block_hash( block )
            )
            if result['result'] is None:
                return 0
            return int(result['result'])

    def get_neuron_for_uid(self, uid, block: int = None) -> dict:
        r""" Returns the neuron metadata of the peer with the passed uid.
        Args:
            uid (int):
                Uid to query for metadata.
        Returns:
            metadata (Dict):
                Dict in list form containing metadata of associated uid.
        """
        
        with self.substrate as substrate:
            result = substrate.get_runtime_state(
                    module='SubtensorModule',
                    storage_function='Neurons',
                    params=[uid],
                    block_hash = None if block == None else substrate.get_block_hash( block )
            )
            return result['result']

    def get_stake_for_uid(self, uid, block: int = None) -> Balance:
        r""" Returns the staked token amount of the peer with the passed uid.
        Args:
            uid (int):
                Uid to query for metadata.
        Returns:
            stake (int):
                Amount of staked token.
        """
        
        with self.substrate as substrate:
            stake = substrate.get_runtime_state(
                module='SubtensorModule',
                storage_function='Stake',
                params = [uid],
                block_hash = None if block == None else substrate.get_block_hash( block )
            )
            result = stake['result']
            if not result:
                return Balance(0)
            return Balance(result)

    def weight_uids_for_uid(self, uid, block: int = None) -> List[int]:
        r""" Returns the weight uids of the peer with the passed uid.
        Args:
            uid (int):
                Uid to query for metadata.
        Returns:
            weight_uids (List[int]):
                Weight uids for passed uid.
        """
        with self.substrate as substrate:
            result = substrate.get_runtime_state(
                module='SubtensorModule',
                storage_function='WeightUids',
                params = [uid],
                block_hash = None if block == None else substrate.get_block_hash( block )
            )
            return result['result']

    def weight_vals_for_uid(self, uid, block: int = None) -> List[int]:
        r""" Returns the weight vals of the peer with the passed uid.
        Args:
            uid (int):
                Uid to query for metadata.
        Returns:
            weight_vals (List[int]):
                Weight vals for passed uid.
        """
        with self.substrate as substrate:
            result = substrate.get_runtime_state(
                module='SubtensorModule',
                storage_function='WeightVals',
                params = [uid],
                block_hash = None if block == None else substrate.get_block_hash( block )
            )
            return result['result']

    def get_last_emit_data_for_uid(self, uid, block: int = None) -> int:
        r""" Returns the last emit of the peer with the passed uid.
        Args:
            uid (int):
                Uid to query for metadata.
        Returns:
            last_emit (int):
                Last emit block numebr
        """
        with self.substrate as substrate:
            result = substrate.get_runtime_state(
                module='SubtensorModule',
                storage_function='LastEmit',
                params = [uid],
                block_hash = None if block == None else substrate.get_block_hash( block )
            )
        return result['result']

    def timeout_set_weights(
            self, 
            timeout,
            wallet: 'bittensor.wallet',
            uids: torch.LongTensor,
            weights: torch.FloatTensor,
            wait_for_inclusion:bool = False,
        ) -> bool:
        r""" wrapper for set weights function that includes a timeout component
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            uids (torch.LongTensor):
                uint64 uids of destination neurons.
            weights (torch.FloatTensor):
                weights to set which must floats and correspond to the passed uids.
            wait_for_inclusion (bool):
                if set, waits for the extrinsic to enter a block before returning true,
                or returns false if the extrinsic fails to enter the block within the timeout.
            timeout (int):
                time that this call waits for either finalization of inclusion.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or included in the block.
        """
        set_weights = Process(target= self.set_weights, kwargs={
                                                            'uids':uids,
                                                            'weights': weights,
                                                            'wait_for_inclusion': wait_for_inclusion,
                                                            'wallet' : wallet,
                                                            })
        set_weights.start()
        set_weights.join(timeout=timeout)
        set_weights.terminate()

        if set_weights.exitcode == 0:
            return True
        
        raise Exception('Timeout')
