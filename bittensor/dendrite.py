# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

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

import asyncio
import uuid
import time
import torch
import httpx
import bittensor as bt
from IPython import get_ipython
from typing import Union, Optional, List

def am_i_in_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

class dendrite(torch.nn.Module):
    """
    The Dendrite class, inheriting from PyTorch's Module class, represents the abstracted 
    implementation of a network client module. In the brain analogy, dendrites receive signals
    from other neurons (in this case, network servers or axons), and the Dendrite class here is designed 
    to send requests to those endpoint to recieve inputs.

    This class includes a wallet or keypair used for signing messages, and methods for making 
    HTTP requests to the network servers. It also provides functionalities such as logging 
    network requests and processing server responses. 

    Attributes:
        keypair: The wallet or keypair used for signing messages.

    Methods:
        __str__(): Returns a string representation of the Dendrite object.
        __repr__(): Returns a string representation of the Dendrite object, acting as a fallback 
                    for __str__().

    Example:
        >>> dendrite_obj = dendrite(wallet = bt.wallet() )
        >>> print(dendrite_obj)
        >>> d( <axon> ) # ping axon
        >>> d( [<axons>] ) # ping multiple
        >>> d( bt.axon(), bt.Synapse )
    """

    def __init__(
            self, 
            wallet: Optional[Union['bt.wallet', 'bt.keypair']] = None
        ):
        """
        Initializes the Dendrite object, setting up essential properties.

        Args:
            wallet (Optional[Union['bt.wallet', 'bt.keypair']], optional): 
                The user's wallet or keypair used for signing messages. Defaults to None, 
                in which case a new bt.wallet().hotkey is generated and used.
        """
        # Initialize the parent class
        super(dendrite, self).__init__()

        # Unique identifier for the instance
        self.uuid = str(uuid.uuid1())

        # HTTP client for making requests
        self.client = httpx.AsyncClient()

        # Get the external IP
        self.external_ip = bt.utils.networking.get_external_ip()

        # If a wallet or keypair is provided, use its hotkey. If not, generate a new one.
        self.keypair = (wallet.hotkey if isinstance(wallet, bt.Wallet) else wallet) or bt.wallet().hotkey

    def query( self, *args, **kwargs ):
        """
        Makes a synchronous request to multiple target Axons and returns the server responses.

        Args:
            axons (Union[List[Union['bt.AxonInfo', 'bt.axon']], Union['bt.AxonInfo', 'bt.axon']]):
                The list of target Axon information.
            synapse (bt.Synapse, optional): The Synapse object. Defaults to bt.Synapse().
            timeout (float, optional): The request timeout duration in seconds. 
                Defaults to 12.0 seconds.
        Returns:
            Union[bt.Synapse, List[bt.Synapse]]: If a single target axon is provided, 
                returns the response from that axon. If multiple target axons are provided, 
                returns a list of responses from all target axons.
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete( self.forward( *args, **kwargs ) )
        except:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            result = loop.run_until_complete( self.forward( *args, **kwargs ) )
            new_loop.close()
            return result

    async def forward(
            self, 
            axons: Union[List[Union['bt.AxonInfo', 'bt.axon']], Union['bt.AxonInfo', 'bt.axon']],
            synapse: bt.Synapse = bt.Synapse(), 
            timeout: float = 12,
            deserialize: bool = True,
        ) -> bt.Synapse:
        """
        Makes asynchronous requests to multiple target Axons and returns the server responses.

        Args:
            axons (Union[List[Union['bt.AxonInfo', 'bt.axon']], Union['bt.AxonInfo', 'bt.axon']]):
                The list of target Axon information.
            synapse (bt.Synapse, optional): The Synapse object. Defaults to bt.Synapse().
            timeout (float, optional): The request timeout duration in seconds. 
                Defaults to 12.0 seconds.

        Returns:
            Union[bt.Synapse, List[bt.Synapse]]: If a single target axon is provided, 
                returns the response from that axon. If multiple target axons are provided, 
                returns a list of responses from all target axons.
        """
        
        # If a single axon is provided, wrap it in a list for uniform processing
        if not isinstance(axons, list):
            axons = [axons]

        # Build coroutines for all axons
        async def query_all_axons():
            coroutines = [ self.call( target_axon = target_axon, synapse = synapse, timeout = timeout, deserialize = deserialize ) for target_axon in axons]
            all_responses = await asyncio.gather(*coroutines)
            return all_responses
        
        # Run all requests concurrently and get the responses
        responses = await query_all_axons()

        # Return the single response if only one axon was targeted, else return all responses
        if len(responses) == 1:
            return responses[0]
        else:
            return responses

    async def call(
        self,
        target_axon: Union['bt.AxonInfo', 'bt.axon'],
        synapse: bt.Synapse = bt.Synapse(),
        timeout: float = 12.0,
        deserialize: bool = True,
    ) -> bt.Synapse:
        """
        Makes an asynchronous request to the target Axon, processes the server 
        response and returns the updated Synapse.

        Args:
            target_axon (Union['bt.AxonInfo', 'bt.axon']): The target Axon information.
            synapse (bt.Synapse, optional): The Synapse object. Defaults to bt.Synapse().
            timeout (float, optional): The request timeout duration in seconds. 
                Defaults to 12.0 seconds.
            deserialize (bool, optional): Whether to deserialize the returned Synapse.
                Defaults to True.

        Returns:
            bt.Synapse: The updated Synapse object after processing server response.
        """
        
        # Record start time
        start_time = time.time()
        target_axon = target_axon.info() if isinstance(target_axon, bt.axon) else target_axon

        # Build request endpoint from the synapse class
        request_name = synapse.__class__.__name__
        endpoint = f"localhost:{str(target_axon.port)}" if target_axon.ip == str(self.external_ip) else f"{target_axon.ip}:{str(target_axon.port)}"
        url = f"http://{endpoint}/{request_name}"

        # Preprocess synapse for making a request
        synapse = self.preprocess_synapse_for_request(target_axon, synapse, timeout)

        try:
            # Log outgoing request
            bt.logging.debug(f"dendrite | --> | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | 0 | Success")
            
            # Make the HTTP POST request
            json_response = await self.client.post( url, headers=synapse.to_headers(), json=synapse.dict(), timeout = timeout )
            
            # Process the server response
            self.process_server_response(json_response, synapse)

            # Set process time and log the response
            synapse.dendrite.process_time = str(time.time() - start_time)
            bt.logging.debug(f"dendrite | <-- | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | {synapse.axon.status_code} | {synapse.axon.status_message}")

        except httpx.TimeoutException as e:
            # Set the status code of the synapse to "406" which indicates a timeout error.
            synapse.dendrite.status_code = '406'
            synapse.dendrite.status_message = f"Timedout after {timeout} seconds."
            bt.logging.debug(f"dendrite | <-- | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | {synapse.dendrite.status_code} | {synapse.dendrite.status_message}")

        except Exception as e:    
            # Handle failure to parse response and log the error
            synapse.dendrite.status_code = '406'
            synapse.dendrite.status_message = f"Failed to parse response object with error: {str(e)}"
            bt.logging.debug(f"dendrite | <-- | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | {synapse.dendrite.status_code} | {synapse.dendrite.status_message}")

        finally:
            # Return the updated synapse object after deserializing if requested
            if deserialize:
                return synapse.deserialize()
            else:
                return synapse


    def preprocess_synapse_for_request(
        self,
        target_axon_info: 'bt.AxonInfo',
        synapse: bt.Synapse, 
        timeout: float = 12.0,    
    ) -> bt.Synapse: 
        """
        Preprocesses the synapse for making a request. This includes building 
        headers for Dendrite and Axon and signing the request.

        Args:
            target_axon_info (bt.AxonInfo): The target axon information.
            synapse (bt.Synapse): The synapse object to be preprocessed.
            timeout (float, optional): The request timeout duration in seconds. 
                Defaults to 12.0 seconds.

        Returns:
            bt.Synapse: The preprocessed synapse.
        """

        # Set the timeout for the synapse
        synapse.timeout = str(timeout)

        # Build the Dendrite headers using the local system's details
        synapse.dendrite = bt.TerminalInfo(
            **{
                "ip": str(self.external_ip),
                "version": str(bt.__version_as_int__),
                "nonce": f"{time.monotonic_ns()}",
                "uuid": str(self.uuid),
                "hotkey": str(self.keypair.ss58_address)
            }
        )

        # Build the Axon headers using the target axon's details
        synapse.axon = bt.TerminalInfo(
            **{
                "ip": str(target_axon_info.ip),
                "port": str(target_axon_info.port),
                "hotkey": str(target_axon_info.hotkey),
            }
        )

        # Sign the request using the dendrite and axon information
        message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}.{synapse.axon.hotkey}.{synapse.dendrite.uuid}"
        synapse.dendrite.signature = f"0x{self.keypair.sign(message).hex()}"

        return synapse
    
    def process_server_response(
        self,
        server_response,
        local_synapse: bt.Synapse,
    ):
        """
        Processes the server response, updates the local synapse state with the 
        server's state and merges headers set by the server.

        Args:
            server_response (object): The response object from the server.
            local_synapse (bt.Synapse): The local synapse object to be updated.

        Raises:
            None, but errors in attribute setting are silently ignored.
        """
        # Check if the server responded with a successful status code
        if server_response.status_code == 200:

            # If the response is successful, overwrite local synapse state with 
            # server's state only if the protocol allows mutation. To prevent overwrites, 
            # the protocol must set allow_mutation = False
            server_synapse = local_synapse.__class__(**server_response.json())
            for key in local_synapse.dict().keys(): 
                try: 
                    # Set the attribute in the local synapse from the corresponding 
                    # attribute in the server synapse
                    setattr(local_synapse, key, getattr(server_synapse, key)) 
                except: 
                    # Ignore errors during attribute setting
                    pass
    
        # Extract server headers and overwrite None values in local synapse headers
        server_headers = bt.Synapse.from_headers(server_response.headers)

        # Merge dendrite headers
        local_synapse.dendrite.__dict__.update(**(local_synapse.dendrite.dict(exclude_none=True) | server_headers.dendrite.dict(exclude_none=True)))
        
        # Merge axon headers
        local_synapse.axon.__dict__.update(**(local_synapse.axon.dict(exclude_none=True) | server_headers.axon.dict(exclude_none=True)))

        # Update the status code and status message of the dendrite to match the axon
        local_synapse.dendrite.status_code = local_synapse.axon.status_code
        local_synapse.dendrite.status_message = local_synapse.axon.status_message


    def __str__(self) -> str:
        """
        Returns a string representation of the Dendrite object.
        
        Returns:
            str: The string representation of the Dendrite object in the format "dendrite(<user_wallet_address>)".
        """
        return "dendrite({})".format(self.keypair.ss58_address)

    def __repr__(self) -> str: 
        """
        Returns a string representation of the Dendrite object, acting as a fallback for __str__().
        
        Returns:
            str: The string representation of the Dendrite object in the format "dendrite(<user_wallet_address>)".
        """
        return self.__str__()
