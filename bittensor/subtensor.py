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
import asyncio
import random
import time
import copy

from munch import Munch
from typing import List, Tuple
from termcolor import colored

import bittensor
import bittensor.utils.networking as net
from bittensor.substrate import SubstrateWSInterface, Keypair
from bittensor.substrate.exceptions import SubstrateRequestException
from bittensor.utils.neurons import Neuron, Neurons
from bittensor.utils.balance import Balance

from loguru import logger
logger = logger.opt(colors=True)

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

    def __init__(
            self, 
            config: 'Munch' = None,
            network: str = None,
            chain_endpoint: str = None
        ):
        r""" Initializes a subtensor chain interface.
            Args:
                config (:obj:`Munch`, `optional`): 
                    metagraph.Metagraph.config()
                network (default='akira', type=str)
                    The subtensor network flag. The likely choices are:
                            -- akira (testing network)
                            -- kusanagi (main network)
                    If this option is set it overloads subtensor.chain_endpoint with 
                    an entry point node from that network.
                chain_endpoint (default=None, type=str)
                    The subtensor endpoint flag. If set, overrides the --network flag.
        """
        if config == None:
            config = Subtensor.default_config()
        config.subtensor.network = network if network != None else config.subtensor.network
        config.subtensor.chain_endpoint = chain_endpoint if chain_endpoint != None else config.subtensor.chain_endpoint
        Subtensor.check_config(config)
        self.config = copy.deepcopy(config)

        self.substrate = SubstrateWSInterface(
            address_type = 42,
            type_registry_preset='substrate-node-template',
            type_registry=self.custom_type_registry,
        )

    @staticmethod
    def default_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Subtensor.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config
    
    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        try:
            parser.add_argument('--subtensor.network', default='kusanagi', type=str, 
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
        pass

    def endpoint_for_network( self, blacklist: List[str] = [] ) -> str:
        r""" Returns a chain endpoint based on config.subtensor.network.
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

    def is_connected(self) -> bool:
        r""" Returns true if the connection state as a boolean.
        Raises:
            success (bool):
                True is the websocket is connected to the chain endpoint.
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_is_connected())

    async def async_is_connected(self) -> bool:
        r""" Returns the connection state as a boolean.
        Raises:
            success (bool):
                True is the websocket is connected to the chain endpoint.
        """
        return await self.substrate.async_is_connected()

    def check_connection(self) -> bool:
        r""" Checks if substrate websocket backend is connected, connects if it is not. 
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_check_connection())

    async def async_check_connection(self) -> bool:
        r""" Checks if substrate websocket backend is connected, connects if it is not. 
        """
        if not await self.async_is_connected():
            return await self.async_connect()
        return True

    def connect( self, timeout: int = 10, failure = True ) -> bool:
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_connect(timeout, failure))

    async def async_connect( self, timeout: int = 10, failure = True ) -> bool:
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
                print('''
Check that your internet connection is working and the chain endpoints are available: <cyan>{}</cyan>
The subtensor.network should likely be one of the following choices:
    -- local - (your locally running node)
    -- akira - (testnet)
    -- kusanagi - (mainnet)
Or you may set the endpoint manually using the --subtensor.chain_endpoint flag 
To run a local node (See: docs/running_a_validator.md) \n
                              '''.format( attempted_endpoints) )

            # ---- Get next endpoint ----
            ws_chain_endpoint = self.endpoint_for_network( blacklist = attempted_endpoints )
            if ws_chain_endpoint == None:
                logger.error("No more endpoints available for subtensor.network: <cyan>{}</cyan>, attempted: <cyan>{}</cyan>".format(self.config.subtensor.network, attempted_endpoints))
                connection_error_message()
                if failure:
                    logger.critical('Unable to connect to network:<cyan>{}</cyan>.\nMake sure your internet connection is stable and the network is properly set.'.format(self.config.subtensor.network))
                else:
                    return False
            attempted_endpoints.append(ws_chain_endpoint)

            # --- Attempt connection ----
            if await self.substrate.async_connect( ws_chain_endpoint, timeout = 5 ):
                logger.success("Connected to network:<cyan>{}</cyan> at endpoint:<cyan>{}</cyan>".format(self.config.subtensor.network, ws_chain_endpoint))
                return True
            
            # ---- Timeout ----
            elif (time.time() - start_time) > timeout:
                logger.error( "Error while connecting to network:<cyan>{}</cyan> at endpoint: <cyan>{}</cyan>".format(self.config.subtensor.network, ws_chain_endpoint))
                connection_error_message()
                if failure:
                    raise RuntimeError('Unable to connect to network:<cyan>{}</cyan>.\nMake sure your internet connection is stable and the network is properly set.'.format(self.config.subtensor.network))
                else:
                    return False

    async def _submit_and_check_extrinsic(
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
            response = await self.substrate.submit_extrinsic( 
                                    extrinsic, 
                                    wait_for_inclusion = wait_for_inclusion,
                                    wait_for_finalization = wait_for_finalization, 
                                    timeout = timeout
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
            if 'error' in response:
                logger.error('Error in extrinsic: {}', response['error'])
            elif 'finalized' in response and response['finalized'] == True:
                return True
            elif 'inBlock' in response and response['inBlock'] == True:
                return True
            else:
                return False

    def is_subscribed(self, wallet: 'bittensor.wallet.Wallet', ip: str, port: int, modality: int ) -> bool:
        r""" Returns true if the bittensor endpoint is already subscribed with the wallet and metadata.
        Args:
            wallet (bittensor.wallet.Wallet):
                bittensor wallet object.
            ip (str):
                endpoint host port i.e. 192.122.31.4
            port (int):
                endpoint port number i.e. 9221
            modality (int):
                int encoded endpoint modality i.e 0 for TEXT
            coldkeypub (str):
                string encoded coldekey pub.
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_is_subscribed( wallet, ip, port, modality ))
            
    async def async_is_subscribed( self, wallet: 'bittensor.wallet.Wallet', ip: str, port: int, modality: int ) -> bool:
        r""" Returns true if the bittensor endpoint is already subscribed with the wallet and metadata.
        Args:
            wallet (bittensor.wallet.Wallet):
                bittensor wallet object.
            ip (str):
                endpoint host port i.e. 192.122.31.4
            port (int):
                endpoint port number i.e. 9221
            modality (int):
                int encoded endpoint modality i.e 0 for TEXT
        """
        uid = await self.async_get_uid_for_pubkey( wallet.hotkey.public_key )
        if uid != None:
            neuron = await self.async_get_neuron_for_uid( uid )
            if neuron['ip'] == net.ip_to_int(ip) and neuron['port'] == port:
                return True
            else:
                return False
        else:
            return False

    def subscribe(
            self, 
            wallet: 'bittensor.wallet.Wallet',
            ip: str, 
            port: int, 
            modality: int, 
            wait_for_inclusion: bool = False,
            wait_for_finalization = True,
            timeout: int = 3 * bittensor.__blocktime__,
        ) -> bool:
        r""" Subscribes an bittensor endpoint to the substensor chain.
        Args:
            wallet (bittensor.wallet.Wallet):
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_subscribe( wallet, ip, port, modality, wait_for_inclusion, wait_for_finalization, timeout ))

    async def async_subscribe(
            self, 
            wallet: 'bittensor.wallet.Wallet',
            ip: str, 
            port: int, 
            modality: int, 
            wait_for_inclusion: bool = False,
            wait_for_finalization:bool = True,
            timeout: int = 3 * bittensor.__blocktime__,
        ) -> bool:
        r""" Subscribes an bittensor endpoint to the substensor chain.
        Args:
            wallet (bittensor.wallet.Wallet):
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
        if not await self.async_check_connection():
            return False

        if await self.async_is_subscribed( wallet, ip, port, modality ):
            logger.success( "Already subscribed with:\n<cyan>[\n  ip: {},\n  port: {},\n  modality: {},\n  hotkey: {},\n  coldkey: {}\n]</cyan>".format(ip, port, modality, wallet.hotkey.public_key, wallet.coldkeypub ))
            return True

        ip_as_int  = net.ip_to_int(ip)
        params = {
            'ip': ip_as_int,
            'port': port, 
            'ip_type': 4,
            'modality': modality,
            'coldkey': wallet.coldkeypub,
        }
        call = await self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='subscribe',
            call_params=params
        )
        # TODO (const): hotkey should be an argument here not assumed. Either that or the coldkey pub should also be assumed.
        extrinsic = await self.substrate.create_signed_extrinsic( call = call, keypair = wallet.hotkey)
        result = await self._submit_and_check_extrinsic (extrinsic, wait_for_inclusion, wait_for_finalization, timeout)
        if result:
            logger.success( "Successfully subscribed with:\n<cyan>[\n  ip: {},\n  port: {},\n  modality: {},\n  hotkey: {},\n  coldkey: {}\n]</cyan>".format(ip, port, modality, wallet.hotkey.public_key, wallet.coldkeypub ))
        else:
            logger.error( "Failed to subscribe")
        return result
            
    def add_stake(
            self, 
            wallet: 'bittensor.wallet.Wallet',
            amount: Balance, 
            hotkey_id: int, 
            wait_for_inclusion: bool = False,
            wait_for_finalization: bool = False,
            timeout: int = 3 * bittensor.__blocktime__,
        ) -> bool:
        r""" Adds the specified amount of stake to passed hotkey uid.
        Args:
            wallet (bittensor.wallet.Wallet):
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_add_stake( wallet, amount, hotkey_id, wait_for_inclusion, wait_for_finalization, timeout ))

    async def async_add_stake(
            self, 
            wallet: 'bittensor.wallet.Wallet',
            amount: Balance, 
            hotkey_id: int, 
            wait_for_inclusion: bool = False,
            wait_for_finalization: bool = False,
            timeout: int = 3 * bittensor.__blocktime__,
        ) -> bool:
        r""" Adds the specified amount of stake to passed hotkey uid.
        Args:
            wallet (bittensor.wallet.Wallet):
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
        await self.async_check_connection()
        call = await self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='add_stake',
            call_params={
                'hotkey': hotkey_id,
                'ammount_staked': amount.rao
            }
        )
        extrinsic = await self.substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
        return await self._submit_and_check_extrinsic ( extrinsic, wait_for_inclusion, wait_for_finalization, timeout )

    def transfer(
            self, 
            wallet: 'bittensor.wallet.Wallet',
            dest:str, 
            amount: Balance, 
            wait_for_inclusion: bool = False,
            wait_for_finalization: bool = False,
            timeout: int = 3 * bittensor.__blocktime__,
        ) -> bool:
        r""" Transfers funds from this wallet to the destination public key address
        Args:
            wallet (bittensor.wallet.Wallet):
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_transfer(wallet, dest, amount, wait_for_inclusion, wait_for_finalization, timeout))

    async def async_transfer(
            self, 
            wallet: 'bittensor.wallet.Wallet',
            dest:str, 
            amount: Balance, 
            wait_for_inclusion: bool = False,
            wait_for_finalization: bool = False,
            timeout: int = 3 * bittensor.__blocktime__,
        ) -> bool:
        r""" Transfers funds from this wallet to the destination public key address
        Args:
            wallet (bittensor.wallet.Wallet):
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
        await self.async_check_connection()
        call = await self.substrate.compose_call(
            call_module='Balances',
            call_function='transfer',
            call_params={
                'dest': dest, 
                'value': amount.rao
            }
        )
        extrinsic = await self.substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
        return await self._submit_and_check_extrinsic ( extrinsic, wait_for_inclusion, wait_for_finalization, timeout )

    def unstake(
            self, 
            wallet: 'bittensor.wallet.Wallet',
            amount: Balance, 
            hotkey_id: int, 
            wait_for_inclusion:bool = False, 
            wait_for_finalization:bool = False,
            timeout: int = 3 * bittensor.__blocktime__,
        ) -> bool:
        r""" Removes stake into the wallet coldkey from the specified hotkey uid.
        Args:
            wallet (bittensor.wallet.Wallet):
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_unstake( wallet, amount, hotkey_id, wait_for_inclusion, wait_for_finalization, timeout ))

    async def async_unstake(
            self, 
            wallet: 'bittensor.wallet.Wallet',
            amount: Balance, 
            hotkey_id: int, 
            wait_for_inclusion:bool = False, 
            wait_for_finalization:bool = False,
            timeout: int = 3 * bittensor.__blocktime__
        ) -> bool:
        r""" Removes stake into the wallet coldkey from the specified hotkey uid.
        Args:
            wallet (bittensor.wallet.Wallet):
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
        await self.async_check_connection()
        call = await self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='remove_stake',
            call_params={'ammount_unstaked': amount.rao, 'hotkey': hotkey_id}
        )
        extrinsic = await self.substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
        return await self._submit_and_check_extrinsic ( extrinsic, wait_for_inclusion, wait_for_finalization, timeout )

    def set_weights(
            self, 
            wallet: 'bittensor.wallet.Wallet',
            destinations, 
            values, 
            wait_for_inclusion:bool = False, 
            wait_for_finalization:bool = False,
            timeout: int = 3 * bittensor.__blocktime__
        ) -> bool:
        r""" Sets the given weights and values on chain for wallet hotkey account.
        Args:
            wallet (bittensor.wallet.Wallet):
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_set_weights( wallet, destinations, values, wait_for_inclusion, wait_for_finalization, timeout ))

    async def async_set_weights(
            self, 
            wallet: 'bittensor.wallet.Wallet',
            destinations, 
            values, 
            wait_for_inclusion:bool = False, 
            wait_for_finalization:bool = False,
            timeout: int = 3 * bittensor.__blocktime__
        ) -> bool:
        r""" Sets the given weights and values on chain for wallet hotkey account.
        Args:
            wallet (bittensor.wallet.Wallet):
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
        await self.async_check_connection()
        call = await self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='set_weights',
            call_params = {'dests': destinations, 'weights': values}
        )
        extrinsic = await self.substrate.create_signed_extrinsic( call = call, keypair = wallet.hotkey )
        return await self._submit_and_check_extrinsic ( extrinsic, wait_for_inclusion, wait_for_finalization, timeout )

    def get_balance(self, address: str) -> Balance:
        r""" Returns the token balance for the passed ss58_address address
        Args:
            address (Substrate address format, default = 42):
                ss58 chain address.
        Return:
            balance (bittensor.utils.balance.Balance):
                account balance
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_get_balance(address))

    async def async_get_balance(self, address) -> Balance:
        r""" Returns the token balance for the passed ss58_address address
        Args:
            address (Substrate address format, default = 42):
                ss58 chain address.
        Return:
            balance (bittensor.utils.balance.Balance):
                account balance
        """
        await self.async_check_connection()
        result = await self.substrate.get_runtime_state(
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_get_current_block())

    async def async_get_current_block(self) -> int:
        r""" Returns the current block number on the chain.
        Returns:
            block_number (int):
                Current chain blocknumber.
        """
        await self.async_check_connection()
        return await self.substrate.async_get_block_number(None)

    def get_active(self) -> List[Tuple[str, int]]:
        r""" Returns a list of (public key, uid) pairs one for each active peer on chain.
        Returns:
            active (List[Tuple[str, int]]):
                List of active peers.
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_get_active())

    async def async_get_active(self) -> List[Tuple[str, int]]:
        r""" Returns a list of (public key, uid) pairs one for each active peer on chain.
        Returns:
            active (List[Tuple[str, int]]):
                List of active peers.
        """
        await self.async_check_connection()
        result = await self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='Active',
        )
        return result

    def get_stake(self) -> List[Tuple[int, int]]:
        r""" Returns a list of (uid, stake) pairs one for each active peer on chain.
        Returns:
            stake (List[Tuple[int, int]]):
                List of stake values.
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_get_stake())

    async def async_get_stake(self) -> List[Tuple[int, int]]:
        r""" Returns a list of (uid, stake) pairs one for each active peer on chain.
        Returns:
            stake (List[Tuple[int, int]]):
                List of stake values.
        """
        await self.async_check_connection()
        result = await self.substrate.iterate_map(
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_get_last_emit())

    async def async_get_last_emit(self) -> List[Tuple[int, int]]:
        r""" Returns a list of (uid, last emit) pairs for each active peer on chain.
        Returns:
            last_emit (List[Tuple[int, int]]):
                List of last emit values.
        """
        await self.async_check_connection()
        result = await self.substrate.iterate_map(
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_get_weight_vals())

    async def async_get_weight_vals(self) -> List[Tuple[int, List[int]]]:
        r""" Returns a list of (uid, weight vals) pairs for each active peer on chain.
        Returns:
            weight_vals (List[Tuple[int, List[int]]]):
                List of weight val pairs.
        """
        await self.async_check_connection()
        result = await self.substrate.iterate_map(
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_get_weight_uids())

    async def async_get_weight_uids(self) -> List[Tuple[int, List[int]]]:
        r""" Returns a list of (uid, weight uids) pairs for each active peer on chain.
        Returns:
            weight_uids (List[Tuple[int, List[int]]]):
                List of weight uid pairs
        """
        await self.async_check_connection()
        result = await self.substrate.iterate_map(
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_neurons( ))

    async def async_neurons(self) -> List[Tuple[int, dict]]:
        r""" Returns a list of neuron from the chain. 
        Returns:
            neuron (List[Tuple[int, dict]]):
                List of neuron objects.
        """
        await self.async_check_connection()
        neurons = await self.substrate.iterate_map(
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_get_uid_for_pubkey( pubkey ))

    async def async_get_uid_for_pubkey(self, pubkey = str) -> int:
        r""" Returns the uid of the peer given passed public key string.
        Args:
            pubkey (str):
                String encoded public key.
        Returns:
            uid (int):
                uid of peer with hotkey equal to passed public key.
        """
        await self.async_check_connection()
        result = await self.substrate.get_runtime_state(
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_get_neuron_for_uid( uid ))

    async def async_get_neuron_for_uid(self, uid:int) -> dict:
        r""" Returns the neuron metadata of the peer with the passed uid.
        Args:
            uid (int):
                Uid to query for metadata.
        Returns:
            metadata (Dict):
                Dict in list form containing metadata of associated uid.
        """
        await self.async_check_connection()
        result = await self.substrate.get_runtime_state(
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_get_stake_for_uid( uid ))

    async def async_get_stake_for_uid(self, uid) -> Balance:
        r""" Returns the staked token amount of the peer with the passed uid.
        Args:
            uid (int):
                Uid to query for metadata.
        Returns:
            stake (int):
                Amount of staked token.
        """
        await self.async_check_connection()
        stake = await self.substrate.get_runtime_state(
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_weight_uids_for_uid( uid ))

    async def async_weight_uids_for_uid(self, uid) -> List[int]:
        r""" Returns the weight uids of the peer with the passed uid.
        Args:
            uid (int):
                Uid to query for metadata.
        Returns:
            weight_uids (List[int]):
                Weight uids for passed uid.
        """
        await self.async_check_connection()
        result = await self.substrate.get_runtime_state(
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_weight_vals_for_uid( uid ))

    async def async_weight_vals_for_uid(self, uid) -> List[int]:
        r""" Returns the weight vals of the peer with the passed uid.
        Args:
            uid (int):
                Uid to query for metadata.
        Returns:
            weight_vals (List[int]):
                Weight vals for passed uid.
        """
        await self.async_check_connection()
        result = await self.substrate.get_runtime_state(
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
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_get_last_emit_data_for_uid( uid ))

    async def async_get_last_emit_data_for_uid(self, uid) -> int:
        r""" Returns the last emit of the peer with the passed uid.
        Args:
            uid (int):
                Uid to query for metadata.
        Returns:
            last_emit (int):
                Last emit block numebr
        """
        await self.async_check_connection()
        result = await self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='LastEmit',
            params = [uid]
        )
        return result['result']
