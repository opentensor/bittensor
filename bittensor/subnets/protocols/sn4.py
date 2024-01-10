import time
import torch
import aiohttp
import asyncio
import pydantic

import bittensor as bt

from typing import List
from starlette.responses import StreamingResponse

from typing import Dict, Optional, Tuple, Union, List, Callable, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from inspect import signature, Signature, Parameter




class TargonLinkPrediction( bt.Synapse ):
    url: str = pydantic.Field(
        ...,
        title="Query",
        description="The query to be asked. Immutable.",
        allow_mutation=False,
    )
    full_text: str = pydantic.Field(
        "",
        title="Full Text",
        description="The full text of the page",
    )
    query: str = pydantic.Field(
        "",
        title="Query",
        description="The predicted query of the page",
    )
    title: str = pydantic.Field(
        "",
        title="Title",
        description="The title of the page",
    )
    new_links: List[str] = pydantic.Field(
        [""],
        title="New Links",
        description="The New Links from the page",
    )
    stream: bool = False
    max_new_tokens: int = 256
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    top_k: int = 10
    top_p: float = 0.9
    required_hash_fields: List[str] = pydantic.Field(
        ["query"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )

class TargonSearchResult( bt.Synapse ):
    query: str = pydantic.Field(
        ...,
        title="Query",
        description="The query to be asked. Immutable.",
        allow_mutation=False,
    )
    sources: List[dict] = pydantic.Field(
        ...,
        title="Sources",
        description="The sources of the query. Mutable.",
    )
    context: List[str] = pydantic.Field(
        [],
        title="Context",
        description="The context of the query. Mutable.",
    )
    completion: str = pydantic.Field(
        "",
        title="Results",
        description="The results of the query. Mutable.",
    )
    stream: bool = False
    max_new_tokens: int = 256
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    top_k: int = 10
    top_p: float = 0.9
    required_hash_fields: List[str] = pydantic.Field(
        ["query"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )


class TargonSearchResultStream( bt.StreamingSynapse ):
    query: str = pydantic.Field(
        ...,
        title="Query",
        description="The query to be asked. Immutable.",
        allow_mutation=False,
    )
    sources: List[dict] = pydantic.Field(
        ...,
        title="Sources",
        description="The sources of the query. Mutable.",
    )
    context: List[str] = pydantic.Field(
        [],
        title="Context",
        description="The context of the query. Mutable.",
    )
    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="The results of the query. Mutable.",
    )
    stream: bool = False
    max_new_tokens: int = 12
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    top_k: int = 10
    top_p: float = 0.9
    required_hash_fields: List[str] = pydantic.Field(
        ["query"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )
    async def process_streaming_response(self, response: StreamingResponse):
        """
        `process_streaming_response` is an asynchronous method designed to process the incoming streaming response from the
        Bittensor network. It's the heart of the StreamPrompting class, ensuring that streaming tokens, which represent
        prompts or messages, are decoded and appropriately managed.

        As the streaming response is consumed, the tokens are decoded from their 'utf-8' encoded format, split based on
        newline characters, and concatenated into the `completion` attribute. This accumulation of decoded tokens in the
        `completion` attribute allows for a continuous and coherent accumulation of the streaming content.

        Args:
            response: The streaming response object containing the content chunks to be processed. Each chunk in this
                      response is expected to be a set of tokens that can be decoded and split into individual messages or prompts.

        Usage:
            Generally, this method is called when there's an incoming streaming response to be processed.

            ```python
            stream_prompter = StreamPrompting(roles=["role1", "role2"], messages=["message1", "message2"])
            await stream_prompter.process_streaming_response(response)
            ```

        Note:
            It's important to remember that this method is asynchronous. Ensure it's called within an appropriate
            asynchronous context.
        """
        if self.completion is None:
            self.completion = ""
        async for chunk in response.content.iter_any():
            tokens = chunk.decode("utf-8").split("\n")
            for token in tokens:
                if token:
                    self.completion += token
                    yield token  # yield token immediately



    def extract_response_json(self, response: StreamingResponse) -> dict:
        """
        `extract_response_json` is a method that performs the crucial task of extracting pertinent JSON data from the given
        response. The method is especially useful when you need a detailed insight into the streaming response's metadata
        or when debugging response-related issues.

        Beyond just extracting the JSON data, the method also processes and structures the data for easier consumption
        and understanding. For instance, it extracts specific headers related to dendrite and axon, offering insights
        about the Bittensor network's internal processes. The method ultimately returns a dictionary with a structured
        view of the extracted data.

        Args:
            response: The response object from which to extract the JSON data. This object typically includes headers and
                      content which can be used to glean insights about the response.

        Returns:
            dict: A structured dictionary containing:
                - Basic response metadata such as name, timeout, total_size, and header_size.
                - Dendrite and Axon related information extracted from headers.
                - Roles and Messages pertaining to the current StreamPrompting instance.
                - The accumulated completion.

        Usage:
            This method can be used after processing a response to gather detailed metadata:

            ```python
            stream_prompter = StreamPrompting(roles=["role1", "role2"], messages=["message1", "message2"])
            # After processing the response...
            json_info = stream_prompter.extract_response_json(response)
            ```

        Note:
            While the primary output is the structured dictionary, understanding this output can be instrumental in
            troubleshooting or in extracting specific insights about the interaction with the Bittensor network.
        """
        headers = {
            k.decode("utf-8"): v.decode("utf-8")
            for k, v in response.__dict__["_raw_headers"]
        }

        def extract_info(prefix):
            return {
                key.split("_")[-1]: value
                for key, value in headers.items()
                if key.startswith(prefix)
            }

        return {
            "name": headers.get("name", ""),
            "timeout": float(headers.get("timeout", 0)),
            "total_size": int(headers.get("total_size", 0)),
            "header_size": int(headers.get("header_size", 0)),
            "dendrite": extract_info("bt_header_dendrite"),
            "axon": extract_info("bt_header_axon"),
            "query": self.query,
            "sources": self.sources,
            # "images": self.images,
            "completion": self.completion,
        }


class TargonDendrite( bt.dendrite ):
    
    async def forward(
        self,
        axons: Union[
            List[Union[bt.AxonInfo, bt.axon]],
            Union[bt.AxonInfo, bt.axon],
        ],
        synapse: bt.Synapse = bt.Synapse(),
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
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
        is_list = True
        # If a single axon is provided, wrap it in a list for uniform processing
        if not isinstance(axons, list):
            is_list = False
            axons = [axons]

        if streaming:
            return self.call_stream(
                target_axon=axons[0],
                synapse=synapse.copy(),
                timeout=timeout,
                deserialize=deserialize,
            )

        # This asynchronous function is used to send queries to all axons.
        async def query_all_axons() -> List[bt.Synapse]:
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

    async def call_stream(
        self,
        target_axon: Union[bt.AxonInfo, bt.axon],
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
        target_axon = (
            target_axon.info()
            if isinstance(target_axon, bt.axon)
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
            bt.logging.debug(
                f"stream dendrite | --> | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | 0 | Success"
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
                    async for token in synapse.process_streaming_response(response):
                        yield token  # Yield each token as it's processed
                    json_response = synapse.extract_response_json(response)
                else:
                    bt.logging.info('stream dendrite | --> | ', response)

                    json_response = await response.json()

                # Process the server response
                self.process_server_response(response, json_response, synapse)

            # Set process time and log the response
            synapse.dendrite.process_time = str(time.time() - start_time)
            bt.logging.debug(
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
                f"Failed to parse response object with error: {e}"
            )

        finally:
            bt.logging.debug(
                f"stream dendrite | <-- | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | {synapse.dendrite.status_code} | {synapse.dendrite.status_message}"
            )

            # Log synapse event history
            self.synapse_history.append(
                bt.Synapse.from_headers(synapse.to_headers())
            )

            # Return the updated synapse object after deserializing if requested
            if deserialize:
                yield synapse.deserialize()
            else:
                yield synapse