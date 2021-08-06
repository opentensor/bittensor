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
import torch

from typing import List, Tuple

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
                network (default='kusanagi', type=str)
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
        start_time = time.time()
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
                with self.substrate as substrate:
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

    def _submit_and_check_extrinsic(
            self, 
            extrinsic, 
            wait_for_inclusion:bool = False, 
            wait_for_finalization: bool = False, 
            timeout: int = bittensor.__blocktime__ * 3
        ) -> bool:
        r""" Makes an extrinsic call to the chain, returns true if the extrinsic send was a success.
        If wait_for_inclusion or wait_for_finalization are true, the call will return true iff the 
        extrinsic enters or finalizes in a block.
        Args:
            extrinsic (substrate extrinsic):
                Extrinsic to send to the chain.
            wait_for_inclusion (bool):
                If set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                If set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            timeout (int):
                Time that this call waits for either finalization of inclusion.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """
        # Send extrinsic
        try:
            with self.substrate as substrate:
                response = substrate.submit_extrinsic( 
                                        extrinsic, 
                                        wait_for_inclusion = wait_for_inclusion,
                                        wait_for_finalization = wait_for_finalization, 
                                )
                
        except SubstrateRequestException as e:
            logger.error('Extrinsic exception with error {}', e)
            return False
        except Exception as e:
            logger.error('Error submitting extrinsic with error {}', e)
            return False

        # Check timeout.
        if response == None:
            logger.error('Error in extrinsic: No response within timeout')
            return False

        # Check result.
        if not wait_for_inclusion and not wait_for_finalization:
            return True
        else:
            if response.is_success:
                return True
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
                return False
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
            timeout: int = 3 * bittensor.__blocktime__,
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
            timeout (int):
                time that this call waits for either finalization of inclusion.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """

        if self.is_subscribed( wallet, ip, port ):
            logger.success( "Subscribed with".ljust(20) + '<blue>ip: {}, port: {}, modality: {}, hotkey: {}, coldkey: {}</blue>'.format(ip, port, modality, wallet.hotkey.public_key, wallet.coldkeypub ))
            return True

        ip_as_int  = net.ip_to_int(ip)

        # TODO(const): subscribe with version too.
        params = {
            'ip': ip_as_int,
            'port': port, 
            'ip_type': 4,
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
            result = self._submit_and_check_extrinsic (extrinsic, wait_for_inclusion, wait_for_finalization, timeout)

            if result:
                logger.success( "Subscribed with".ljust(20) + '<blue>ip: {}, port: {}, modality: {}, hotkey: {}, coldkey: {}</blue>'.format(ip, port, modality, wallet.hotkey.public_key, wallet.coldkeypub ))
            else:
                logger.error( "Failed to subscribe")
            return result

    def add_stake(
            self, 
            wallet: 'bittensor.wallet',
            amount: Balance, 
            hotkey_id: int, 
            wait_for_inclusion: bool = False,
            wait_for_finalization: bool = False,
            timeout: int = 3 * bittensor.__blocktime__,
        ) -> bool:
        r""" Adds the specified amount of stake to passed hotkey uid.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            amount (bittensor.utils.balance.Balance):
                amount to stake as bittensor balance
            hotkey_id (int):
                uid of hotkey to stake into.
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
        
        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module='SubtensorModule',
                call_function='add_stake',
                call_params={
                    'hotkey': hotkey_id,
                    'ammount_staked': amount.rao
                }
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
            return self._submit_and_check_extrinsic ( extrinsic, wait_for_inclusion, wait_for_finalization, timeout )

    def transfer(
            self, 
            wallet: 'bittensor.wallet',
            dest:str, 
            amount: Balance, 
            wait_for_inclusion: bool = False,
            wait_for_finalization: bool = False,
            timeout: int = 3 * bittensor.__blocktime__,
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
            timeout (int):
                time that this call waits for either finalization of inclusion.
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
            return self._submit_and_check_extrinsic ( extrinsic, wait_for_inclusion, wait_for_finalization, timeout )

    def unstake(
            self, 
            wallet: 'bittensor.wallet',
            amount: Balance, 
            hotkey_id: int, 
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
            hotkey_id (int):
                uid of hotkey to unstake from.
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
        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module='SubtensorModule',
                call_function='remove_stake',
                call_params={'ammount_unstaked': amount.rao, 'hotkey': hotkey_id}
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
            return self._submit_and_check_extrinsic ( extrinsic, wait_for_inclusion, wait_for_finalization, timeout )

    def set_weights(
            self, 
            wallet: 'bittensor.wallet',
            uids: torch.LongTensor,
            weights: torch.FloatTensor,
            wait_for_inclusion:bool = False,
            wait_for_finalization:bool = False,
            timeout: int = 3 * bittensor.__blocktime__
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
        weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit( uids, weights )
        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module='SubtensorModule',
                call_function='set_weights',
                call_params = {'dests': weight_uids, 'weights': weight_vals}
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.hotkey )
            return self._submit_and_check_extrinsic ( extrinsic, wait_for_inclusion, wait_for_finalization, timeout )


    def set_weights_v1_1_0(self,
        wallet: 'bittensor.wallet',
        destinations,
        values,
        transaction_fee,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        timeout: int = 3 * bittensor.__blocktime__
    ) -> bool:
        """ Sets the given weights and values on chain for wallet hotkey account.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            destinations (List[int]):
                uint64 uids of destination neurons.
            values (List[int]):
                u32 max encoded floating point weights.
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
        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module='SubtensorModule',
                call_function='set_weights',
                call_params = {'dests': destinations, 'weights': values, 'fee': transaction_fee}
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.hotkey )
            return self._submit_and_check_extrinsic ( extrinsic, wait_for_inclusion, wait_for_finalization, timeout )


    def get_balance(self, address: str) -> Balance:
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
                block_hash=None
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

    def get_active(self) -> List[Tuple[str, int]]:
        r""" Returns a list of (public key, uid) pairs one for each active peer on chain.
        Returns:
            active (List[Tuple[str, int]]):
                List of active peers.
        """
        with self.substrate as substrate:
            result =  substrate.iterate_map(
                module='SubtensorModule',
                storage_function='Active',
            )
            return result

    def stake(self) -> List[Tuple[int, int]]:
        r""" Returns a list of (uid, stake) pairs one for each active peer on chain.
        Returns:
            stake (List[Tuple[int, int]]):
                List of stake values.
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_get_stake())

    def get_stake(self) -> List[Tuple[int, int]]:
        r""" Returns a list of (uid, stake) pairs one for each active peer on chain.
        Returns:
            stake (List[Tuple[int, int]]):
                List of stake values.
        """
        
        with self.substrate as substrate:
            result = substrate.iterate_map(
                module='SubtensorModule',
                storage_function='Stake',
            )
            return result

    def get_last_emit(self) -> List[Tuple[int, int]]:
        r""" Returns a list of (uid, last emit) pairs for each active peer on chain.
        Returns:
            last_emit (List[Tuple[int, int]]):
                List of last emit values.
        """
        with self.substrate as substrate:
            result = substrate.iterate_map(
                module='SubtensorModule',
                storage_function='LastEmit'
            )
            return result

    def get_weight_vals(self) -> List[Tuple[int, List[int]]]:
        r""" Returns a list of (uid, weight vals) pairs for each active peer on chain.
        Returns:
            weight_vals (List[Tuple[int, List[int]]]):
                List of weight val pairs.
        """
        with self.substrate as substrate:
            result = substrate.iterate_map(
                module='SubtensorModule',
                storage_function='WeightVals'
            )
            return result

    def get_weight_uids(self) -> List[Tuple[int, int]]:
        r""" Returns a list of (uid, weight uids) pairs for each active peer on chain.
        Returns:
            weight_uids (List[Tuple[int, List[int]]]):
                List of weight uid pairs
        """
        with self.substrate as substrate:
            result = substrate.iterate_map(
                module='SubtensorModule',
                storage_function='WeightUids'
            )
            return result

    def neurons(self) -> List[Tuple[int, dict]]: 
        r""" Returns a list of neuron from the chain. 
        Returns:
            neuron (List[Tuple[int, dict]]):
                List of neuron objects.
        """
        with self.substrate as substrate:
            neurons = substrate.iterate_map(
                module='SubtensorModule',
                storage_function='Neurons'
            )
            return neurons

    def get_uid_for_pubkey(self, pubkey = str) -> int: 
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
                params=[pubkey]
            )
            if result['result'] is None:
                return None
            return int(result['result'])

    def get_neuron_for_uid(self, uid) -> dict:
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
                    params=[uid]
            )
            return result['result']

    def get_stake_for_uid(self, uid) -> Balance:
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
                params = [uid]
            )
            result = stake['result']
            if not result:
                return Balance(0)
            return Balance(result)

    def weight_uids_for_uid(self, uid) -> List[int]:
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
                params = [uid]
            )
            return result['result']

    def weight_vals_for_uid(self, uid) -> List[int]:
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
                params = [uid]
            )
            return result['result']

    def get_last_emit_data_for_uid(self, uid) -> int:
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
                params = [uid]
            )
        return result['result']