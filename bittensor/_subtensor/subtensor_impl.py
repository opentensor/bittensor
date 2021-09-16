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
import torch
from multiprocessing import Process

from typing import List, Tuple, Dict, Union

import bittensor
import bittensor.utils.networking as net
import bittensor.utils.weight_utils as weight_utils
from substrateinterface import SubstrateInterface
from bittensor.utils.balance import Balance
from types import SimpleNamespace

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

        neuron = self.neuron_for_pubkey( wallet.hotkey.ss58_address )
        if not neuron.is_null and neuron.ip == ip and neuron.port == port:
            logger.success( "Already subscribed".ljust(20) + '<blue>ip: {}, port: {}, modality: {}, hotkey: {}, coldkey: {}</blue>'.format(ip, port, modality, wallet.hotkey.public_key, wallet.coldkeypub ))
            return True

        ip_as_int  = net.ip_to_int(ip)
        ip_version = net.ip_version(ip)

        # TODO(const): subscribe with version too.
        params = {
            'version': bittensor.__version_as_int__,
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
            result = substrate.query(
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

    def neurons(self, block: int = None) -> List[Tuple[int, dict]]: 
        r""" Returns a list of neuron from the chain. 
        Returns:
            neuron (List[Tuple[int, dict]]):
                List of neuron objects.
        """
        with self.substrate as substrate:
            neurons = substrate.query_map (
                module='SubtensorModule',
                storage_function='Neurons',
                block_hash = None if block == None else substrate.get_block_hash( block )
            )
            result = []
            for n in neurons:
                n = SimpleNamespace( **dict(n[1].value) )
                if n.hotkey == "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM":
                    n.is_null = True
                else:
                    n.is_null = False
                result.append( n )
            return result

    def neuron_for_uid( self, uid: int, block: int = None ) -> Union[ dict, None]: 
        r""" Returns a list of neuron from the chain. 
        Args:
            uid ( int ):
                The uid of the neuron to query for.

        Returns:
            neuron (dict(NeuronMetadata)):
                neuron object associated with uid or None if it does not exist.
        """
        with self.substrate as substrate:
            neuron = dict( substrate.query( module='SubtensorModule',  storage_function='Neurons', params = [ uid ]).value )
            neuron = SimpleNamespace( **neuron )
            neuron.ip = net.int_to_ip(neuron.ip)
            if neuron.hotkey == "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM":
                neuron.is_null = True
            else:
                neuron.is_null = False
            return neuron

    def neuron_for_pubkey( self, ss58_hotkey: str, block: int = None ) -> SimpleNamespace: 
        r""" Returns a list of neuron from the chain. 
        Args:
            ss58_hotkey ( str ):
                The hotkey to query for a neuron.

        Returns:
            neuron ( dict(NeuronMetadata) ):
                neuron object associated with uid or None if it does not exist.
        """
        with self.substrate as substrate:
            result = substrate.query (
                module='SubtensorModule',
                storage_function='Hotkeys',
                params = [ ss58_hotkey ],
                block_hash = None if block == None else substrate.get_block_hash( block )
            )
            
            # Get response uid. This will be zero if it doesn't exist.
            uid = int(result.value)
            neuron = self.neuron_for_uid( uid, block)
            if neuron.hotkey != ss58_hotkey:
                neuron.is_null = True
            else:
                neuron.is_null = False
            return neuron

    def get_n( self, block: int = None ) -> int: 
        r""" Returns the number of neurons on the chain at block.
        Args:
            block ( int ):
                The block number to get the neuron count from.

        Returns:
            n ( int ):
                the number of neurons subscribed to the chain.
        """
        with self.substrate as substrate:
            return int(substrate.query(  module='SubtensorModule', storage_function = 'N' ).value)

    def neuron_for_wallet( self, wallet: 'bittensor.Wallet', block: int = None ) -> SimpleNamespace: 
        r""" Returns a list of neuron from the chain. 
        Args:
            wallet ( `bittensor.Wallet` ):
                Checks to ensure that the passed wallet is subscribed.
        Returns:
            neuron ( dict(NeuronMetadata) ):
                neuron object associated with uid or None if it does not exist.
        """
        return self.neuron_for_pubkey ( wallet.hotkey.ss58_address )

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
