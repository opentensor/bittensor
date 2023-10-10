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

from __future__ import annotations

import asyncio
import uuid
import time
import torch
import aiohttp
import bittensor
from fastapi import Response
from typing import Union, Optional, List, Union


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
        >>> dendrite_obj = dendrite(wallet = bittensor.wallet() )
        >>> print(dendrite_obj)
        >>> d( <axon> ) # ping axon
        >>> d( [<axons>] ) # ping multiple
        >>> d( bittensor.axon(), bittensor.Synapse )
    """

    def __init__(
        self, wallet: Optional[Union[bittensor.wallet, bittensor.keypair]] = None
    ):
        """
        Initializes the Dendrite object, setting up essential properties.

        Args:
            wallet (Optional[Union['bittensor.wallet', 'bittensor.keypair']], optional):
                The user's wallet or keypair used for signing messages. Defaults to None,
                in which case a new bittensor.wallet().hotkey is generated and used.
        """
        # Initialize the parent class
        super(dendrite, self).__init__()

        # Unique identifier for the instance
        self.uuid = str(uuid.uuid1())

        # Get the external IP
        self.external_ip = bittensor.utils.networking.get_external_ip()

        # If a wallet or keypair is provided, use its hotkey. If not, generate a new one.
        self.keypair = (
            wallet.hotkey if isinstance(wallet, bittensor.wallet) else wallet
        ) or bittensor.wallet().hotkey

        self.synapse_history: list = []

        self._session: aiohttp.ClientSession = None

    @property
    async def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close_session(self):
        if self._session:
            await self._session.close()
            self._session = None

    def query(
        self, *args, **kwargs
    ) -> Union[bittensor.Synapse, List[bittensor.Synapse]]:
        """
        Makes a synchronous request to multiple target Axons and returns the server responses.

        Args:
            axons (Union[List[Union['bittensor.AxonInfo', 'bittensor.axon']], Union['bittensor.AxonInfo', 'bittensor.axon']]):
                The list of target Axon information.
            synapse (bittensor.Synapse, optional): The Synapse object. Defaults to bittensor.Synapse().
            timeout (float, optional): The request timeout duration in seconds.
                Defaults to 12.0 seconds.
        Returns:
            Union[bittensor.Synapse, List[bittensor.Synapse]]: If a single target axon is provided,
                returns the response from that axon. If multiple target axons are provided,
                returns a list of responses from all target axons.
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.forward(*args, **kwargs))
        except:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            result = loop.run_until_complete(self.forward(*args, **kwargs))
            new_loop.close()
            return result

    async def forward(
        self,
        axons: Union[
            List[Union[bittensor.AxonInfo, bittensor.axon]],
            Union[bittensor.AxonInfo, bittensor.axon],
        ],
        synapse: bittensor.Synapse = bittensor.Synapse(),
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
    ) -> bittensor.Synapse:
        """
        Makes asynchronous requests to multiple target Axons and returns the server responses.

        Args:
            axons (Union[List[Union['bittensor.AxonInfo', 'bittensor.axon']], Union['bittensor.AxonInfo', 'bittensor.axon']]):
                The list of target Axon information.
            synapse (bittensor.Synapse, optional): The Synapse object. Defaults to bittensor.Synapse().
            timeout (float, optional): The request timeout duration in seconds.
                Defaults to 12.0 seconds.

        Returns:
            Union[bittensor.Synapse, List[bittensor.Synapse]]: If a single target axon is provided,
                returns the response from that axon. If multiple target axons are provided,
                returns a list of responses from all target axons.
        """
        is_list = True
        # If a single axon is provided, wrap it in a list for uniform processing
        if not isinstance(axons, list):
            is_list = False
            axons = [axons]

        # This asynchronous function is used to send queries to all axons.
        async def query_all_axons() -> List[bittensor.Synapse]:
            # If the 'run_async' flag is not set, the code runs synchronously.
            if not run_async:
                # Create an empty list to hold the responses from all axons.
                all_responses = []
                # Loop through each axon in the 'axons' list.
                for target_axon in axons:
                    # The response from each axon is then appended to the 'all_responses' list.
                    all_responses.append(
                        await self.call(
                            target_axon=target_axon,
                            synapse=synapse.copy(),
                            timeout=timeout,
                            deserialize=deserialize,
                        )
                    )
                # The function then returns a list of responses from all axons.
                return all_responses
            else:
                # Here we build a list of coroutines without awaiting them.
                coroutines = [
                    self.call(
                        target_axon=target_axon,
                        synapse=synapse.copy(),
                        timeout=timeout,
                        deserialize=deserialize,
                    )
                    for target_axon in axons
                ]
                # 'asyncio.gather' is a method which takes multiple coroutines and runs them in parallel.
                all_responses = await asyncio.gather(*coroutines)
                # The function then returns a list of responses from all axons.
                return all_responses

        # Run all requests concurrently and get the responses
        responses = await query_all_axons()

        # Return the single response if only one axon was targeted, else return all responses
        if len(responses) == 1 and not is_list:
            return responses[0]
        else:
            return responses

    async def call(
        self,
        target_axon: Union[bittensor.AxonInfo, bittensor.axon],
        synapse: bittensor.Synapse = bittensor.Synapse(),
        timeout: float = 12.0,
        deserialize: bool = True,
    ) -> bittensor.Synapse:
        """
        Makes an asynchronous request to the target Axon, processes the server
        response and returns the updated Synapse.

        Args:
            target_axon (Union['bittensor.AxonInfo', 'bittensor.axon']): The target Axon information.
            synapse (bittensor.Synapse, optional): The Synapse object. Defaults to bittensor.Synapse().
            timeout (float, optional): The request timeout duration in seconds.
                Defaults to 12.0 seconds.
            deserialize (bool, optional): Whether to deserialize the returned Synapse.
                Defaults to True.

        Returns:
            bittensor.Synapse: The updated Synapse object after processing server response.
        """

        # Record start time
        start_time = time.time()
        target_axon = (
            target_axon.info()
            if isinstance(target_axon, bittensor.axon)
            else target_axon
        )

        # Build request endpoint from the synapse class
        request_name = synapse.__class__.__name__
        endpoint = (
            f"0.0.0.0:{str(target_axon.port)}"
            if target_axon.ip == str(self.external_ip)
            else f"{target_axon.ip}:{str(target_axon.port)}"
        )
        url = f"http://{endpoint}/{request_name}"

        # Preprocess synapse for making a request
        synapse = self.preprocess_synapse_for_request(target_axon, synapse, timeout)

        try:
            # Log outgoing request
            bittensor.logging.debug(
                f"dendrite | --> | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | 0 | Success"
            )

            # Make the HTTP POST request
            async with (await self.session).post(
                url,
                headers=synapse.to_headers(),
                json=synapse.dict(),
                timeout=timeout,
            ) as response:
                if (
                    response.headers.get("Content-Type", "").lower()
                    == "text/event-stream".lower()
                ):  # identify streaming response
                    await synapse.process_streaming_response(
                        response
                    )  # process the entire streaming response
                    json_response = synapse.extract_response_json(response)
                else:
                    json_response = await response.json()

                # Process the server response
                self.process_server_response(response, json_response, synapse)

            # Set process time and log the response
            synapse.dendrite.process_time = str(time.time() - start_time)

        except aiohttp.ClientConnectorError as e:
            synapse.dendrite.status_code = "503"
            synapse.dendrite.status_message = f"Service at {synapse.axon.ip}:{str(synapse.axon.port)}/{request_name} unavailable."

        except asyncio.TimeoutError as e:
            synapse.dendrite.status_code = "408"
            synapse.dendrite.status_message = f"Timedout after {timeout} seconds."

        except Exception as e:
            synapse.dendrite.status_code = "422"
            synapse.dendrite.status_message = (
                f"Failed to parse response object with error: {str(e)}"
            )

        finally:
            bittensor.logging.debug(
                f"dendrite | <-- | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | {synapse.dendrite.status_code} | {synapse.dendrite.status_message}"
            )

            # Log synapse event history
            self.synapse_history.append(
                bittensor.Synapse.from_headers(synapse.to_headers())
            )

            # Return the updated synapse object after deserializing if requested
            if deserialize:
                return synapse.deserialize()
            else:
                return synapse

    def preprocess_synapse_for_request(
        self,
        target_axon_info: bittensor.AxonInfo,
        synapse: bittensor.Synapse,
        timeout: float = 12.0,
    ) -> bittensor.Synapse:
        """
        Preprocesses the synapse for making a request. This includes building
        headers for Dendrite and Axon and signing the request.

        Args:
            target_axon_info (bittensor.AxonInfo): The target axon information.
            synapse (bittensor.Synapse): The synapse object to be preprocessed.
            timeout (float, optional): The request timeout duration in seconds.
                Defaults to 12.0 seconds.

        Returns:
            bittensor.Synapse: The preprocessed synapse.
        """
        # Set the timeout for the synapse
        synapse.timeout = str(timeout)

        # Build the Dendrite headers using the local system's details
        synapse.dendrite = bittensor.TerminalInfo(
            **{
                "ip": str(self.external_ip),
                "version": str(bittensor.__version_as_int__),
                "nonce": f"{time.monotonic_ns()}",
                "uuid": str(self.uuid),
                "hotkey": str(self.keypair.ss58_address),
            }
        )

        # Build the Axon headers using the target axon's details
        synapse.axon = bittensor.TerminalInfo(
            **{
                "ip": str(target_axon_info.ip),
                "port": str(target_axon_info.port),
                "hotkey": str(target_axon_info.hotkey),
            }
        )

        # Sign the request using the dendrite, axon info, and the synapse body hash
        message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}.{synapse.axon.hotkey}.{synapse.dendrite.uuid}.{synapse.body_hash}"
        synapse.dendrite.signature = f"0x{self.keypair.sign(message).hex()}"

        return synapse

    def process_server_response(
        self,
        server_response: Response,
        json_response: dict,
        local_synapse: bittensor.Synapse,
    ):
        """
        Processes the server response, updates the local synapse state with the
        server's state and merges headers set by the server.

        Args:
            server_response (object): The aiohttp response object from the server.
            json_response (dict): The parsed JSON response from the server.
            local_synapse (bittensor.Synapse): The local synapse object to be updated.

        Raises:
            None, but errors in attribute setting are silently ignored.
        """
        # Check if the server responded with a successful status code
        if server_response.status == 200:
            # If the response is successful, overwrite local synapse state with
            # server's state only if the protocol allows mutation. To prevent overwrites,
            # the protocol must set allow_mutation = False
            server_synapse = local_synapse.__class__(**json_response)
            for key in local_synapse.dict().keys():
                try:
                    # Set the attribute in the local synapse from the corresponding
                    # attribute in the server synapse
                    setattr(local_synapse, key, getattr(server_synapse, key))
                except:
                    # Ignore errors during attribute setting
                    pass

        # Extract server headers and overwrite None values in local synapse headers
        server_headers = bittensor.Synapse.from_headers(server_response.headers)

        # Merge dendrite headers
        local_synapse.dendrite.__dict__.update(
            {
                **local_synapse.dendrite.dict(exclude_none=True),
                **server_headers.dendrite.dict(exclude_none=True),
            }
        )

        # Merge axon headers
        local_synapse.axon.__dict__.update(
            {
                **local_synapse.axon.dict(exclude_none=True),
                **server_headers.axon.dict(exclude_none=True),
            }
        )

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
