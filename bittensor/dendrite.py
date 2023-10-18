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
from typing import Union, Optional, List, Union, AsyncGenerator


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
        streaming: bool = False,
    ) -> bittensor.Synapse:
        """
        Asynchronously sends requests to one or multiple Axons and collates their responses.

        This function acts as a bridge for sending multiple requests concurrently or sequentially 
        based on the provided parameters. It checks the type of the target Axons, preprocesses 
        the requests, and then sends them off. After getting the responses, it processes and 
        collates them into a unified format.

        Args:
            axons (Union[List[Union['bittensor.AxonInfo', 'bittensor.axon']], Union['bittensor.AxonInfo', 'bittensor.axon']]):
                The target Axons to send requests to. Can be a single Axon or a list of Axons.
            synapse (bittensor.Synapse, optional): The Synapse object encapsulating the data. Defaults to a new bittensor.Synapse instance.
            timeout (float, optional): Maximum duration to wait for a response from an Axon in seconds. Defaults to 12.0.
            deserialize (bool, optional): Determines if the received response should be deserialized. Defaults to True.
            run_async (bool, optional): If True, sends requests concurrently. Otherwise, sends requests sequentially. Defaults to True.
            streaming (bool, optional): Indicates if the response is expected to be in streaming format. Defaults to False.

        Returns:
            Union[bittensor.Synapse, List[bittensor.Synapse]]: If a single Axon is targeted, returns its response. 
            If multiple Axons are targeted, returns a list of their responses.
        """
        is_list = True
        # If a single axon is provided, wrap it in a list for uniform processing
        if not isinstance(axons, list):
            is_list = False
            axons = [axons]

        # Check if synapse is an instance of the StreamingSynapse class or if streaming flag is set.
        streaming = (
            issubclass(synapse.__class__, bittensor.StreamingSynapse) or streaming
        )

        async def query_all_axons(is_stream: bool) -> List[bt.Synapse]:
            """
            Handles requests for all axons, either in streaming or non-streaming mode.

            Args:
                is_stream: If True, handles the axons in streaming mode.

            Returns:
                List of Synapse objects with responses.
            """

            async def single_axon_response(target_axon) -> bt.Synapse:
                """
                Retrieve response for a single axon, either in streaming or non-streaming mode.

                Args:
                    target_axon: The target axon to send request to.

                Returns:
                    A Synapse object with the response.
                """
                if is_stream:
                    # If in streaming mode, we iterate over the streaming content
                    # and take the last item as the final result.
                    final_result = None
                    async for result in self.call_stream(
                        target_axon=target_axon,
                        synapse=synapse.copy(),
                        timeout=timeout,
                        deserialize=deserialize,
                    ):
                        final_result = result
                    return final_result
                else:
                    # If not in streaming mode, simply call the axon and get the response.
                    return await self.call(
                        target_axon=target_axon,
                        synapse=synapse.copy(),
                        timeout=timeout,
                        deserialize=deserialize,
                    )

            # If run_async flag is False, get responses one by one.
            if not run_async:
                return [
                    await single_axon_response(target_axon) for target_axon in axons
                ]
            # If run_async flag is True, get responses concurrently using asyncio.gather().
            return await asyncio.gather(
                *(single_axon_response(target_axon) for target_axon in axons)
            )

        # Get responses for all axons.
        responses = await query_all_axons(streaming)
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
        Asynchronously sends a request to a specified Axon and processes the response.

        This function establishes a connection with a specified Axon, sends the encapsulated 
        data through the Synapse object, waits for a response, processes it, and then 
        returns the updated Synapse object.

        Args:
            target_axon (Union['bittensor.AxonInfo', 'bittensor.axon']): The target Axon to send the request to.
            synapse (bittensor.Synapse, optional): The Synapse object encapsulating the data. Defaults to a new bittensor.Synapse instance.
            timeout (float, optional): Maximum duration to wait for a response from the Axon in seconds. Defaults to 12.0.
            deserialize (bool, optional): Determines if the received response should be deserialized. Defaults to True.

        Returns:
            bittensor.Synapse: The Synapse object, updated with the response data from the Axon.
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
                # Extract the JSON response from the server
                json_response = await response.json()
                # Process the server response and fill synapse
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

    async def call_stream(
        self,
        target_axon: Union[bittensor.AxonInfo, bittensor.axon],
        synapse: bittensor.Synapse = bittensor.Synapse(),
        timeout: float = 12.0,
        deserialize: bool = True,
    ) -> AsyncGenerator[bittensor.Synapse, None]:
        """
        Sends a request to a specified Axon and yields streaming responses.

        Similar to `call`, but designed for scenarios where the Axon sends back data in 
        multiple chunks or streams. The function yields each chunk as it is received. This is 
        useful for processing large responses piece by piece without waiting for the entire 
        data to be transmitted.

        Args:
            target_axon (Union['bittensor.AxonInfo', 'bittensor.axon']): The target Axon to send the request to.
            synapse (bittensor.Synapse, optional): The Synapse object encapsulating the data. Defaults to a new bittensor.Synapse instance.
            timeout (float, optional): Maximum duration to wait for a response (or a chunk of the response) from the Axon in seconds. Defaults to 12.0.
            deserialize (bool, optional): Determines if each received chunk should be deserialized. Defaults to True.

        Yields:
            bittensor.Synapse: Each yielded Synapse object contains a chunk of the response data from the Axon.
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
                f"stream dendrite | --> | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | 0 | Success"
            )

            # Make the HTTP POST request
            async with (await self.session).post(
                url,
                headers=synapse.to_headers(),
                json=synapse.dict(),
                timeout=timeout,
            ) as response:
                # Use synapse subclass' process_streaming_response method to yield the response chunks
                async for chunk in synapse.process_streaming_response(response):
                    yield chunk  # Yield each chunk as it's processed
                json_response = synapse.extract_response_json(response)

                # Process the server response
                self.process_server_response(response, json_response, synapse)

            # Set process time and log the response
            synapse.dendrite.process_time = str(time.time() - start_time)
            bittensor.logging.debug(
                f"stream dendrite | <-- | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | {synapse.axon.status_code} | {synapse.axon.status_message}"
            )

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
                yield synapse.deserialize()
            else:
                yield synapse

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
