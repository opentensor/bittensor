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
import aiohttp
import bittensor
from typing import Optional, List, Union, AsyncGenerator, Any
from bittensor.utils.registration import torch, use_torch


class DendriteMixin:
    """
    The Dendrite class represents the abstracted implementation of a network client module.

    In the brain analogy, dendrites receive signals
    from other neurons (in this case, network servers or axons), and the Dendrite class here is designed
    to send requests to those endpoint to recieve inputs.

    This class includes a wallet or keypair used for signing messages, and methods for making
    HTTP requests to the network servers. It also provides functionalities such as logging
    network requests and processing server responses.

    Args:
        keypair: The wallet or keypair used for signing messages.
        external_ip (str): The external IP address of the local system.
        synapse_history (list): A list of Synapse objects representing the historical responses.

    Methods:
        __str__(): Returns a string representation of the Dendrite object.
        __repr__(): Returns a string representation of the Dendrite object, acting as a fallback for __str__().
        query(self, *args, **kwargs) -> Union[bittensor.Synapse, List[bittensor.Synapse]]:
            Makes synchronous requests to one or multiple target Axons and returns responses.

        forward(self, axons, synapse=bittensor.Synapse(), timeout=12, deserialize=True, run_async=True, streaming=False) -> bittensor.Synapse:
            Asynchronously sends requests to one or multiple Axons and collates their responses.

        call(self, target_axon, synapse=bittensor.Synapse(), timeout=12.0, deserialize=True) -> bittensor.Synapse:
            Asynchronously sends a request to a specified Axon and processes the response.

        call_stream(self, target_axon, synapse=bittensor.Synapse(), timeout=12.0, deserialize=True) -> AsyncGenerator[bittensor.Synapse, None]:
            Sends a request to a specified Axon and yields an AsyncGenerator that contains streaming response chunks before finally yielding the filled Synapse as the final element.

        preprocess_synapse_for_request(self, target_axon_info, synapse, timeout=12.0) -> bittensor.Synapse:
            Preprocesses the synapse for making a request, including building headers and signing.

        process_server_response(self, server_response, json_response, local_synapse):
            Processes the server response, updates the local synapse state, and merges headers.

        close_session(self):
            Synchronously closes the internal aiohttp client session.

        aclose_session(self):
            Asynchronously closes the internal aiohttp client session.

    NOTE:
        When working with async `aiohttp <https://github.com/aio-libs/aiohttp>`_ client sessions, it is recommended to use a context manager.

    Example with a context manager::

        >>> aysnc with dendrite(wallet = bittensor.wallet()) as d:
        >>>     print(d)
        >>>     d( <axon> ) # ping axon
        >>>     d( [<axons>] ) # ping multiple
        >>>     d( bittensor.axon(), bittensor.Synapse )

    However, you are able to safely call :func:`dendrite.query()` without a context manager in a synchronous setting.

    Example without a context manager::

        >>> d = dendrite(wallet = bittensor.wallet() )
        >>> print(d)
        >>> d( <axon> ) # ping axon
        >>> d( [<axons>] ) # ping multiple
        >>> d( bittensor.axon(), bittensor.Synapse )
    """

    def __init__(
        self, wallet: Optional[Union[bittensor.wallet, bittensor.Keypair]] = None
    ):
        """
        Initializes the Dendrite object, setting up essential properties.

        Args:
            wallet (Optional[Union['bittensor.wallet', 'bittensor.keypair']], optional):
                The user's wallet or keypair used for signing messages. Defaults to ``None``, in which case a new :func:`bittensor.wallet().hotkey` is generated and used.
        """
        # Initialize the parent class
        super(DendriteMixin, self).__init__()

        # Unique identifier for the instance
        self.uuid = str(uuid.uuid1())

        # Get the external IP
        self.external_ip = bittensor.utils.networking.get_external_ip()

        # If a wallet or keypair is provided, use its hotkey. If not, generate a new one.
        self.keypair = (
            wallet.hotkey if isinstance(wallet, bittensor.wallet) else wallet
        ) or bittensor.wallet().hotkey

        self.synapse_history: list = []

        self._session: Optional[aiohttp.ClientSession] = None

    @property
    async def session(self) -> aiohttp.ClientSession:
        """
        An asynchronous property that provides access to the internal `aiohttp <https://github.com/aio-libs/aiohttp>`_ client session.

        This property ensures the management of HTTP connections in an efficient way. It lazily
        initializes the `aiohttp.ClientSession <https://docs.aiohttp.org/en/stable/client_reference.html#aiohttp.ClientSession>`_ on its first use. The session is then reused for subsequent
        HTTP requests, offering performance benefits by reusing underlying connections.

        This is used internally by the dendrite when querying axons, and should not be used directly
        unless absolutely necessary for your application.

        Returns:
            aiohttp.ClientSession: The active `aiohttp <https://github.com/aio-libs/aiohttp>`_ client session instance. If no session exists, a
            new one is created and returned. This session is used for asynchronous HTTP requests within
            the dendrite, adhering to the async nature of the network interactions in the Bittensor framework.

        Example usage::

            import bittensor as bt                    # Import bittensor
            wallet = bt.wallet( ... )                 # Initialize a wallet
            dendrite = bt.dendrite( wallet )          # Initialize a dendrite instance with the wallet

            async with (await dendrite.session).post( # Use the session to make an HTTP POST request
                url,                                  # URL to send the request to
                headers={...},                        # Headers dict to be sent with the request
                json={...},                           # JSON body data to be sent with the request
                timeout=10,                           # Timeout duration in seconds
            ) as response:
                json_response = await response.json() # Extract the JSON response from the server

        """
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    def close_session(self):
        """
        Closes the internal `aiohttp <https://github.com/aio-libs/aiohttp>`_ client session synchronously.

        This method ensures the proper closure and cleanup of the aiohttp client session, releasing any
        resources like open connections and internal buffers. It is crucial for preventing resource leakage
        and should be called when the dendrite instance is no longer in use, especially in synchronous contexts.

        Note:
            This method utilizes asyncio's event loop to close the session asynchronously from a synchronous context. It is advisable to use this method only when asynchronous context management is not feasible.

        Usage:
            When finished with dendrite in a synchronous context
            :func:`dendrite_instance.close_session()`.
        """
        if self._session:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._session.close())
            self._session = None

    async def aclose_session(self):
        """
        Asynchronously closes the internal `aiohttp <https://github.com/aio-libs/aiohttp>`_ client session.

        This method is the asynchronous counterpart to the :func:`close_session` method. It should be used in
        asynchronous contexts to ensure that the aiohttp client session is closed properly. The method
        releases resources associated with the session, such as open connections and internal buffers,
        which is essential for resource management in asynchronous applications.

        Usage:
            When finished with dendrite in an asynchronous context
            await :func:`dendrite_instance.aclose_session()`.

        Example::

            async with dendrite_instance:
                # Operations using dendrite
                pass
            # The session will be closed automatically after the above block
        """
        if self._session:
            await self._session.close()
            self._session = None

    def _get_endpoint_url(self, target_axon, request_name):
        """
        Constructs the endpoint URL for a network request to a target axon.

        This internal method generates the full HTTP URL for sending a request to the specified axon. The
        URL includes the IP address and port of the target axon, along with the specific request name. It
        differentiates between requests to the local system (using '0.0.0.0') and external systems.

        Args:
            target_axon: The target axon object containing IP and port information.
            request_name: The specific name of the request being made.

        Returns:
            str: A string representing the complete HTTP URL for the request.
        """
        endpoint = (
            f"0.0.0.0:{str(target_axon.port)}"
            if target_axon.ip == str(self.external_ip)
            else f"{target_axon.ip}:{str(target_axon.port)}"
        )
        return f"http://{endpoint}/{request_name}"

    def _handle_request_errors(self, synapse, request_name, exception):
        """
        Handles exceptions that occur during network requests, updating the synapse with appropriate status codes and messages.

        This method interprets different types of exceptions and sets the corresponding status code and
        message in the synapse object. It covers common network errors such as connection issues and timeouts.

        Args:
            synapse: The synapse object associated with the request.
            request_name: The name of the request during which the exception occurred.
            exception: The exception object caught during the request.

        Note:
            This method updates the synapse object in-place.
        """
        if isinstance(exception, aiohttp.ClientConnectorError):
            synapse.dendrite.status_code = "503"
            synapse.dendrite.status_message = f"Service at {synapse.axon.ip}:{str(synapse.axon.port)}/{request_name} unavailable."
        elif isinstance(exception, asyncio.TimeoutError):
            synapse.dendrite.status_code = "408"
            synapse.dendrite.status_message = (
                f"Timedout after {synapse.timeout} seconds."
            )
        else:
            synapse.dendrite.status_code = "422"
            synapse.dendrite.status_message = (
                f"Failed to parse response object with error: {str(exception)}"
            )

    def _log_outgoing_request(self, synapse):
        """
        Logs information about outgoing requests for debugging purposes.

        This internal method logs key details about each outgoing request, including the size of the
        request, the name of the synapse, the axon's details, and a success indicator. This information
        is crucial for monitoring and debugging network activity within the Bittensor network.

        To turn on debug messages, set the environment variable BITTENSOR_DEBUG to ``1``, or call the bittensor debug method like so::

            import bittensor
            bittensor.debug()

        Args:
            synapse: The synapse object representing the request being sent.
        """
        bittensor.logging.trace(
            f"dendrite | --> | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | 0 | Success"
        )

    def _log_incoming_response(self, synapse):
        """
        Logs information about incoming responses for debugging and monitoring.

        Similar to :func:`_log_outgoing_request`, this method logs essential details of the incoming responses,
        including the size of the response, synapse name, axon details, status code, and status message.
        This logging is vital for troubleshooting and understanding the network interactions in Bittensor.

        Args:
            synapse: The synapse object representing the received response.
        """
        bittensor.logging.trace(
            f"dendrite | <-- | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | {synapse.dendrite.status_code} | {synapse.dendrite.status_message}"
        )

    def query(
        self, *args, **kwargs
    ) -> List[
        Union[AsyncGenerator[Any, Any], bittensor.Synapse, bittensor.StreamingSynapse]
    ]:
        """
        Makes a synchronous request to multiple target Axons and returns the server responses.

        Cleanup is automatically handled and sessions are closed upon completed requests.

        Args:
            axons (Union[List[Union['bittensor.AxonInfo', 'bittensor.axon']], Union['bittensor.AxonInfo', 'bittensor.axon']]):
                The list of target Axon information.
            synapse (bittensor.Synapse, optional): The Synapse object. Defaults to :func:`bittensor.Synapse()`.
            timeout (float, optional): The request timeout duration in seconds.
                Defaults to ``12.0`` seconds.
        Returns:
            Union[bittensor.Synapse, List[bittensor.Synapse]]: If a single target axon is provided, returns the response from that axon. If multiple target axons are provided, returns a list of responses from all target axons.
        """
        result = None
        try:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self.forward(*args, **kwargs))
        except:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            result = loop.run_until_complete(self.forward(*args, **kwargs))
            new_loop.close()
        finally:
            self.close_session()
            return result  # type: ignore

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
    ) -> List[
        Union[AsyncGenerator[Any, Any], bittensor.Synapse, bittensor.StreamingSynapse]
    ]:
        """
        Asynchronously sends requests to one or multiple Axons and collates their responses.

        This function acts as a bridge for sending multiple requests concurrently or sequentially
        based on the provided parameters. It checks the type of the target Axons, preprocesses
        the requests, and then sends them off. After getting the responses, it processes and
        collates them into a unified format.

        When querying an Axon that sends a single response, this function returns a Synapse object
        containing the response data. If multiple Axons are queried, a list of Synapse objects is
        returned, each containing the response from the corresponding Axon.

        For example::

            >>> ...
            >>> wallet = bittensor.wallet()                   # Initialize a wallet
            >>> synapse = bittensor.Synapse(...)              # Create a synapse object that contains query data
            >>> dendrte = bittensor.dendrite(wallet = wallet) # Initialize a dendrite instance
            >>> axons = metagraph.axons                       # Create a list of axons to query
            >>> responses = await dendrite(axons, synapse)    # Send the query to all axons and await the responses

        When querying an Axon that sends back data in chunks using the Dendrite, this function
        returns an AsyncGenerator that yields each chunk as it is received. The generator can be
        iterated over to process each chunk individually.

        For example::

            >>> ...
            >>> dendrte = bittensor.dendrite(wallet = wallet)
            >>> async for chunk in dendrite.forward(axons, synapse, timeout, deserialize, run_async, streaming):
            >>>     # Process each chunk here
            >>>     print(chunk)

        Args:
            axons (Union[List[Union['bittensor.AxonInfo', 'bittensor.axon']], Union['bittensor.AxonInfo', 'bittensor.axon']]):
                The target Axons to send requests to. Can be a single Axon or a list of Axons.
            synapse (bittensor.Synapse, optional): The Synapse object encapsulating the data. Defaults to a new :func:`bittensor.Synapse` instance.
            timeout (float, optional): Maximum duration to wait for a response from an Axon in seconds. Defaults to ``12.0``.
            deserialize (bool, optional): Determines if the received response should be deserialized. Defaults to ``True``.
            run_async (bool, optional): If ``True``, sends requests concurrently. Otherwise, sends requests sequentially. Defaults to ``True``.
            streaming (bool, optional): Indicates if the response is expected to be in streaming format. Defaults to ``False``.

        Returns:
            Union[AsyncGenerator, bittensor.Synapse, List[bittensor.Synapse]]: If a single Axon is targeted, returns its response.
            If multiple Axons are targeted, returns a list of their responses.
        """
        is_list = True
        # If a single axon is provided, wrap it in a list for uniform processing
        if not isinstance(axons, list):
            is_list = False
            axons = [axons]

        # Check if synapse is an instance of the StreamingSynapse class or if streaming flag is set.
        is_streaming_subclass = issubclass(
            synapse.__class__, bittensor.StreamingSynapse
        )
        if streaming != is_streaming_subclass:
            bittensor.logging.warning(
                f"Argument streaming is {streaming} while issubclass(synapse, StreamingSynapse) is {synapse.__class__.__name__}. This may cause unexpected behavior."
            )
        streaming = is_streaming_subclass or streaming

        async def query_all_axons(
            is_stream: bool,
        ) -> Union[
            AsyncGenerator[Any, Any], bittensor.Synapse, bittensor.StreamingSynapse
        ]:
            """
            Handles the processing of requests to all targeted axons, accommodating both streaming and non-streaming responses.

            This function manages the concurrent or sequential dispatch of requests to a list of axons.
            It utilizes the ``is_stream`` parameter to determine the mode of response handling (streaming
            or non-streaming). For each axon, it calls ``single_axon_response`` and aggregates the responses.

            Args:
                is_stream (bool): Flag indicating whether the axon responses are expected to be streamed.
                If ``True``, responses are handled in streaming mode.

            Returns:
                List[Union[AsyncGenerator, bittensor.Synapse, bittensor.StreamingSynapse]]: A list
                containing the responses from each axon. The type of each response depends on the
                streaming mode and the type of synapse used.
            """

            async def single_axon_response(
                target_axon,
            ) -> Union[
                AsyncGenerator[Any, Any], bittensor.Synapse, bittensor.StreamingSynapse
            ]:
                """
                Manages the request and response process for a single axon, supporting both streaming and non-streaming modes.

                This function is responsible for initiating a request to a single axon. Depending on the
                ``is_stream`` flag, it either uses ``call_stream`` for streaming responses or ``call`` for
                standard responses. The function handles the response processing, catering to the specifics
                of streaming or non-streaming data.

                Args:
                    target_axon: The target axon object to which the request is to be sent. This object contains the necessary information like IP address and port to formulate the request.

                Returns:
                    Union[AsyncGenerator, bittensor.Synapse, bittensor.StreamingSynapse]: The response
                    from the targeted axon. In streaming mode, an AsyncGenerator is returned, yielding
                    data chunks. In non-streaming mode, a Synapse or StreamingSynapse object is returned
                    containing the response.
                """
                if is_stream:
                    # If in streaming mode, return the async_generator
                    return self.call_stream(
                        target_axon=target_axon,
                        synapse=synapse.model_copy(),  # type: ignore
                        timeout=timeout,
                        deserialize=deserialize,
                    )
                else:
                    # If not in streaming mode, simply call the axon and get the response.
                    return await self.call(
                        target_axon=target_axon,
                        synapse=synapse.model_copy(),  # type: ignore
                        timeout=timeout,
                        deserialize=deserialize,
                    )

            # If run_async flag is False, get responses one by one.
            if not run_async:
                return [
                    await single_axon_response(target_axon) for target_axon in axons
                ]  # type: ignore
            # If run_async flag is True, get responses concurrently using asyncio.gather().
            return await asyncio.gather(
                *(single_axon_response(target_axon) for target_axon in axons)
            )  # type: ignore

        # Get responses for all axons.
        responses = await query_all_axons(streaming)
        # Return the single response if only one axon was targeted, else return all responses
        return responses[0] if len(responses) == 1 and not is_list else responses  # type: ignore

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
            synapse (bittensor.Synapse, optional): The Synapse object encapsulating the data. Defaults to a new :func:`bittensor.Synapse` instance.
            timeout (float, optional): Maximum duration to wait for a response from the Axon in seconds. Defaults to ``12.0``.
            deserialize (bool, optional): Determines if the received response should be deserialized. Defaults to ``True``.

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
        url = self._get_endpoint_url(target_axon, request_name=request_name)

        # Preprocess synapse for making a request
        synapse = self.preprocess_synapse_for_request(target_axon, synapse, timeout)

        try:
            # Log outgoing request
            self._log_outgoing_request(synapse)

            # Make the HTTP POST request
            async with (await self.session).post(
                url,
                headers=synapse.to_headers(),
                json=synapse.model_dump(),
                timeout=timeout,
            ) as response:
                # Extract the JSON response from the server
                json_response = await response.json()
                # Process the server response and fill synapse
                self.process_server_response(response, json_response, synapse)

            # Set process time and log the response
            synapse.dendrite.process_time = str(time.time() - start_time)  # type: ignore

        except Exception as e:
            self._handle_request_errors(synapse, request_name, e)

        finally:
            self._log_incoming_response(synapse)

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
        synapse: bittensor.StreamingSynapse = bittensor.Synapse(),  # type: ignore
        timeout: float = 12.0,
        deserialize: bool = True,
    ) -> AsyncGenerator[Any, Any]:
        """
        Sends a request to a specified Axon and yields streaming responses.

        Similar to ``call``, but designed for scenarios where the Axon sends back data in
        multiple chunks or streams. The function yields each chunk as it is received. This is
        useful for processing large responses piece by piece without waiting for the entire
        data to be transmitted.

        Args:
            target_axon (Union['bittensor.AxonInfo', 'bittensor.axon']): The target Axon to send the request to.
            synapse (bittensor.Synapse, optional): The Synapse object encapsulating the data. Defaults to a new :func:`bittensor.Synapse` instance.
            timeout (float, optional): Maximum duration to wait for a response (or a chunk of the response) from the Axon in seconds. Defaults to ``12.0``.
            deserialize (bool, optional): Determines if each received chunk should be deserialized. Defaults to ``True``.

        Yields:
            object: Each yielded object contains a chunk of the arbitrary response data from the Axon.
            bittensor.Synapse: After the AsyncGenerator has been exhausted, yields the final filled Synapse.
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
        synapse = self.preprocess_synapse_for_request(target_axon, synapse, timeout)  # type: ignore

        try:
            # Log outgoing request
            self._log_outgoing_request(synapse)

            # Make the HTTP POST request
            async with (await self.session).post(
                url,
                headers=synapse.to_headers(),
                json=synapse.model_dump(),
                timeout=timeout,
            ) as response:
                # Use synapse subclass' process_streaming_response method to yield the response chunks
                async for chunk in synapse.process_streaming_response(response):  # type: ignore
                    yield chunk  # Yield each chunk as it's processed
                json_response = synapse.extract_response_json(response)

                # Process the server response
                self.process_server_response(response, json_response, synapse)

            # Set process time and log the response
            synapse.dendrite.process_time = str(time.time() - start_time)  # type: ignore

        except Exception as e:
            self._handle_request_errors(synapse, request_name, e)

        finally:
            self._log_incoming_response(synapse)

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
                Defaults to ``12.0`` seconds.

        Returns:
            bittensor.Synapse: The preprocessed synapse.
        """
        # Set the timeout for the synapse
        synapse.timeout = timeout

        # Build the Dendrite headers using the local system's details
        synapse.dendrite = bittensor.TerminalInfo(
            ip=self.external_ip,
            version=bittensor.__version_as_int__,
            nonce=time.monotonic_ns(),
            uuid=self.uuid,
            hotkey=self.keypair.ss58_address,
        )

        # Build the Axon headers using the target axon's details
        synapse.axon = bittensor.TerminalInfo(
            ip=target_axon_info.ip,
            port=target_axon_info.port,
            hotkey=target_axon_info.hotkey,
        )

        # Sign the request using the dendrite, axon info, and the synapse body hash
        message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}.{synapse.axon.hotkey}.{synapse.dendrite.uuid}.{synapse.body_hash}"
        synapse.dendrite.signature = f"0x{self.keypair.sign(message).hex()}"

        return synapse

    def process_server_response(
        self,
        server_response: aiohttp.ClientResponse,
        json_response: dict,
        local_synapse: bittensor.Synapse,
    ):
        """
        Processes the server response, updates the local synapse state with the
        server's state and merges headers set by the server.

        Args:
            server_response (object): The `aiohttp <https://github.com/aio-libs/aiohttp>`_ response object from the server.
            json_response (dict): The parsed JSON response from the server.
            local_synapse (bittensor.Synapse): The local synapse object to be updated.

        Raises:
            None: But errors in attribute setting are silently ignored.
        """
        # Check if the server responded with a successful status code
        if server_response.status == 200:
            # If the response is successful, overwrite local synapse state with
            # server's state only if the protocol allows mutation. To prevent overwrites,
            # the protocol must set Frozen = True
            server_synapse = local_synapse.__class__(**json_response)
            for key in local_synapse.model_dump().keys():
                try:
                    # Set the attribute in the local synapse from the corresponding
                    # attribute in the server synapse
                    setattr(local_synapse, key, getattr(server_synapse, key))
                except:
                    # Ignore errors during attribute setting
                    pass

        # Extract server headers and overwrite None values in local synapse headers
        server_headers = bittensor.Synapse.from_headers(server_response.headers)  # type: ignore

        # Merge dendrite headers
        local_synapse.dendrite.__dict__.update(
            {
                **local_synapse.dendrite.model_dump(exclude_none=True),  # type: ignore
                **server_headers.dendrite.model_dump(exclude_none=True),  # type: ignore
            }
        )

        # Merge axon headers
        local_synapse.axon.__dict__.update(
            {
                **local_synapse.axon.model_dump(exclude_none=True),  # type: ignore
                **server_headers.axon.model_dump(exclude_none=True),  # type: ignore
            }
        )

        # Update the status code and status message of the dendrite to match the axon
        local_synapse.dendrite.status_code = local_synapse.axon.status_code  # type: ignore
        local_synapse.dendrite.status_message = local_synapse.axon.status_message  # type: ignore

    def __str__(self) -> str:
        """
        Returns a string representation of the Dendrite object.

        Returns:
            str: The string representation of the Dendrite object in the format :func:`dendrite(<user_wallet_address>)`.
        """
        return "dendrite({})".format(self.keypair.ss58_address)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Dendrite object, acting as a fallback for :func:`__str__()`.

        Returns:
            str: The string representation of the Dendrite object in the format :func:`dendrite(<user_wallet_address>)`.
        """
        return self.__str__()

    async def __aenter__(self):
        """
        Asynchronous context manager entry method.

        Enables the use of the ``async with`` statement with the Dendrite instance. When entering the context,
        the current instance of the class is returned, making it accessible within the asynchronous context.

        Returns:
            Dendrite: The current instance of the Dendrite class.

        Usage::

            async with Dendrite() as dendrite:
                await dendrite.some_async_method()
        """
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """
        Asynchronous context manager exit method.

        Ensures proper cleanup when exiting the ``async with`` context. This method will close the `aiohttp <https://github.com/aio-libs/aiohttp>`_ client session
        asynchronously, releasing any tied resources.

        Args:
            exc_type (Type[BaseException], optional): The type of exception that was raised.
            exc_value (BaseException, optional): The instance of exception that was raised.
            traceback (TracebackType, optional): A traceback object encapsulating the call stack at the point where the exception was raised.

        Usage::

            async with bt.dendrite( wallet ) as dendrite:
                await dendrite.some_async_method()

        Note:
            This automatically closes the session by calling :func:`__aexit__` after the context closes.
        """
        await self.aclose_session()

    def __del__(self):
        """
        Dendrite destructor.

        This method is invoked when the Dendrite instance is about to be destroyed. The destructor ensures that the
        aiohttp client session is closed before the instance is fully destroyed, releasing any remaining resources.

        Note:
            Relying on the destructor for cleanup can be unpredictable. It is recommended to explicitly close sessions using the provided methods or the ``async with`` context manager.

        Usage::

            dendrite = Dendrite()
            # ... some operations ...
            del dendrite  # This will implicitly invoke the __del__ method and close the session.
        """
        self.close_session()


# For back-compatibility with torch
BaseModel: Union["torch.nn.Module", object] = torch.nn.Module if use_torch() else object


class dendrite(DendriteMixin, BaseModel):  # type: ignore
    def __init__(
        self, wallet: Optional[Union[bittensor.wallet, bittensor.Keypair]] = None
    ):
        if use_torch():
            torch.nn.Module.__init__(self)
        DendriteMixin.__init__(self, wallet)


if not use_torch():

    async def call(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)

    dendrite.__call__ = call
