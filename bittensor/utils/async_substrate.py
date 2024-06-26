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

"""An Asyncio-compatible implementation of the Py-Substrate API, as used within bittensor."""

import asyncio
import functools
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Any, Union, Callable, Awaitable

import websockets
from scalecodec import GenericExtrinsic
from scalecodec.base import ScaleBytes, ScaleType, RuntimeConfigurationObject
from scalecodec.type_registry import load_type_registry_preset
from scalecodec.types import GenericCall
from substrateinterface import Keypair, ExtrinsicReceipt
from substrateinterface.base import SubstrateInterface, QueryMapResult
from substrateinterface.exceptions import SubstrateRequestException, BlockNotFound
from substrateinterface.storage import StorageKey

import bittensor

TEST_CHAIN_ENDPOINT = "wss://test.finney.opentensor.ai:443"

ResultHandler = Callable[[dict, Any], Awaitable[tuple[dict, bool]]]


def ensure_initialized(func):
    """Wrapper for initialization of SubstrateInterface before connection."""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not self.initialized and not self.substrate:
            await self.initialize()
        return await func(self, *args, **kwargs)

    return wrapper


def no_wrap(func):
    """
    Marks a function as "do not run" for the class wrapper
    """
    func._no_wrap = True
    return func


def all_coroutine_methods(wrapper):
    """Applies a given wrapper to all coroutines in a class."""

    def class_decorator(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if asyncio.iscoroutinefunction(attr_value) and not getattr(
                attr_value, "_no_wrap", False
            ):
                setattr(cls, attr_name, wrapper(attr_value))
        return cls

    return class_decorator


@dataclass
class Preprocessed:
    queryable: str
    method: str
    params: list
    value_scale_type: str
    storage_item: ScaleType


class Runtime:
    block_hash: str
    block_id: int
    runtime_version = None
    transaction_version = None
    cache_region = None
    metadata = None
    type_registry_preset = None

    def __init__(self, chain, runtime_config, metadata):
        self.runtime_config = RuntimeConfigurationObject()
        self.config = {}
        self.chain = chain
        self.type_registry = bittensor.__type_registry__
        self.runtime_config = runtime_config
        self.metadata = metadata

    @property
    def implements_scaleinfo(self) -> bool:
        """
        Returns True if current runtime implementation a `PortableRegistry` (`MetadataV14` and higher)
        """
        if self.metadata:
            return self.metadata.portable_registry is not None
        else:
            return False

    def reload_type_registry(
        self, use_remote_preset: bool = True, auto_discover: bool = True
    ):
        """
        Reload type registry and preset used to instantiate the SubstrateInterface object. Useful to periodically apply
        changes in type definitions when a runtime upgrade occurred

        Parameters
        ----------
        use_remote_preset: When True preset is downloaded from Github master, otherwise use files from local installed
                           scalecodec package
        auto_discover

        Returns
        -------

        """
        self.runtime_config.clear_type_registry()

        self.runtime_config.implements_scale_info = self.implements_scaleinfo

        # Load metadata types in runtime configuration
        self.runtime_config.update_type_registry(load_type_registry_preset(name="core"))
        self.apply_type_registry_presets(
            use_remote_preset=use_remote_preset, auto_discover=auto_discover
        )

    def apply_type_registry_presets(
        self,
        use_remote_preset: bool = True,
        auto_discover: bool = True,
    ):
        """
        Applies type registry presets to the runtime
        :param use_remote_preset: bool, whether to use presets from remote
        :param auto_discover: bool, whether to use presets from local installed scalecodec package
        """
        if self.type_registry_preset is not None:
            # Load type registry according to preset
            type_registry_preset_dict = load_type_registry_preset(
                name=self.type_registry_preset, use_remote_preset=use_remote_preset
            )

            if not type_registry_preset_dict:
                raise ValueError(
                    f"Type registry preset '{self.type_registry_preset}' not found"
                )

        elif auto_discover:
            # Try to auto discover type registry preset by chain name
            type_registry_name = self.chain.lower().replace(" ", "-")
            try:
                type_registry_preset_dict = load_type_registry_preset(
                    type_registry_name
                )
                self.type_registry_preset = type_registry_name
            except ValueError:
                type_registry_preset_dict = None

        else:
            type_registry_preset_dict = None

        if type_registry_preset_dict:
            # Load type registries in runtime configuration
            if self.implements_scaleinfo is False:
                # Only runtime with no embedded types in metadata need the default set of explicit defined types
                self.runtime_config.update_type_registry(
                    load_type_registry_preset(
                        "legacy", use_remote_preset=use_remote_preset
                    )
                )

            if self.type_registry_preset != "legacy":
                self.runtime_config.update_type_registry(type_registry_preset_dict)

        if self.type_registry:
            # Load type registries in runtime configuration
            self.runtime_config.update_type_registry(self.type_registry)


class RequestManager:
    RequestResults = dict[Union[str, int], list[Union[ScaleType, dict]]]

    def __init__(self, payloads):
        self.response_map = {}
        self.responses = defaultdict(lambda: {"complete": False, "results": []})
        self.payloads_count = len(payloads)

    def add_request(self, item_id: int, request_id: Any):
        """
        Adds an outgoing request to the responses map for later retrieval
        """
        self.response_map[item_id] = request_id

    def overwrite_request(self, item_id: int, request_id: Any):
        """
        Overwrites an existing request in the responses map with a new request_id. This is used
        for multipart responses that generate a subscription id we need to watch, rather than the initial
        request_id.
        """
        self.response_map[request_id] = self.response_map.pop(item_id)
        return request_id

    def add_response(self, item_id: int, response: dict, complete: bool):
        """
        Maps a response to the request for later retrieval
        """
        request_id = self.response_map[item_id]
        self.responses[request_id]["results"].append(response)
        self.responses[request_id]["complete"] = complete

    @property
    def is_complete(self):
        """
        Returns whether all requests in the manager have completed
        """
        return (
            all(info["complete"] for info in self.responses.values())
            and len(self.responses) == self.payloads_count
        )

    def get_results(self) -> RequestResults:
        """
        Generates a dictionary mapping the requests initiated to the responses received.
        """
        return {
            request_id: info["results"] for request_id, info in self.responses.items()
        }


class Websocket:
    def __init__(
        self,
        ws_url: str,
        max_subscriptions=1024,
        max_connections=100,
        shutdown_timer=5,
        options: dict = None,
    ):
        """
        Websocket manager object. Allows for the use of a single websocket connection by multiple
        calls.

        :param ws_url: Websocket URL to connect to
        :param max_subscriptions: Maximum number of subscriptions per websocket connection
        :param max_connections: Maximum number of connections total
        :param shutdown_timer: Number of seconds to shut down websocket connection after last use
        """
        # TODO allow setting max concurrent connections and rpc subscriptions per connection
        # TODO reconnection logic
        self.ws_url = ws_url
        self.ws = None
        self.id = 0
        self.max_subscriptions = max_subscriptions
        self.max_connections = max_connections
        self.shutdown_timer = shutdown_timer
        self._received = {}
        self._in_use = 0
        self._receiving_task = None
        self._attempts = 0
        self._initialized = False
        self._lock = asyncio.Lock()
        self._exit_task = None
        self._open_subscriptions = 0
        self._options = options if options else {}

    async def __aenter__(self):
        async with self._lock:
            self._in_use += 1
            if self._exit_task:
                self._exit_task.cancel()
            if not self._initialized:
                self._initialized = True
                await self._connect()
                self._receiving_task = asyncio.create_task(self._start_receiving())
        return self

    async def _connect(self):
        self.ws = await asyncio.wait_for(
            websockets.connect(self.ws_url, **self._options), timeout=None
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self._lock:
            self._in_use -= 1
            if self._exit_task is not None:
                self._exit_task.cancel()
                try:
                    await self._exit_task
                except asyncio.CancelledError:
                    pass
            if self._in_use == 0 and self.ws is not None:
                self.id = 0
                self._open_subscriptions = 0
                self._exit_task = asyncio.create_task(self._exit_with_timer())

    async def _exit_with_timer(self, timer=None):
        """
        Allows for graceful shutdown of websocket connection after specified number of seconds, allowing
        for reuse of the websocket connection.
        """
        shutdown_timer = timer if timer is not None else self.shutdown_timer
        try:
            await asyncio.sleep(shutdown_timer)
            async with self._lock:
                if self._receiving_task:
                    self._receiving_task.cancel()
                    try:
                        await self._receiving_task
                    except asyncio.CancelledError:
                        pass
                if self.ws:
                    await self.ws.close()
                    self.ws = None
                self._initialized = False
                self._receiving_task = None
                self.id = 0
        except asyncio.CancelledError:
            pass

    async def close(self):
        await self._exit_with_timer(0)

    async def _recv(self) -> None:
        try:
            response = json.loads(await self.ws.recv())
            async with self._lock:
                self._open_subscriptions -= 1
                if "id" in response:
                    self._received[response["id"]] = response
                elif "params" in response:
                    self._received[response["params"]["subscription"]] = response
                else:
                    raise KeyError(response)
        except websockets.ConnectionClosed:
            raise
        except KeyError as e:
            print(f"Unhandled websocket response: {e}")
            raise e

    async def _start_receiving(self):
        try:
            while True:
                await self._recv()
        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed:
            # TODO try reconnect, but only if it's needed
            raise

    async def send(self, payload: dict) -> int:
        """
        Sends a payload to the websocket connection.

        :param payload: payload, generate a payload with the AsyncSubstrateInterface.make_payload method
        """
        async with self._lock:
            original_id = self.id
            try:
                await self.ws.send(json.dumps({**payload, **{"id": original_id}}))
                self.id += 1
                self._open_subscriptions += 1
                return original_id
            except websockets.ConnectionClosed:
                raise

    async def retrieve(self, item_id: int) -> Optional[dict]:
        """
        Retrieves a single item from received responses dict queue

        :param item_id: id of the item to retrieve

        :return: retrieved item
        """
        while True:
            async with self._lock:
                if item_id in self._received:
                    return self._received.pop(item_id)
            await asyncio.sleep(0.1)


class RuntimeCache:
    def __init__(self):
        self.cache = {}
        self.by_hash_cache = {}
        self.metadata_cache = {}

    def by_block_id(self, block_id: int) -> Optional[Runtime]:
        """
        Retrieve a runtime by the specified block id
        """
        if block_id in self.cache:
            return self.cache[block_id]

    def by_hash(self, block_hash: str) -> Optional[Runtime]:
        """
        Retrieve a runtime by its block hash
        """
        if block_hash in self.by_hash_cache:
            return self.cache[self.by_hash_cache[block_hash]]

    def add(self, block_id: int, block_hash: str, runtime: Runtime) -> None:
        """
        Adds a runtime to the cache, mapped by its block hash and id
        """
        self.cache[block_id] = runtime
        self.by_hash_cache[block_hash] = block_id


@all_coroutine_methods(ensure_initialized)
class AsyncSubstrateInterface:
    runtime = None
    substrate = None

    def __init__(
        self,
        url: str,
        use_remote_preset=False,
        auto_discover=True,
        auto_reconnect=True,
    ):
        """
        The asyncio-compatible version of the subtensor interface commands we use in bittensor
        """
        self.chain_endpoint = url
        self.__chain = None
        self.ws = Websocket(
            url,
            options={
                "max_size": 2**32,
                "read_limit": 2**32,
                "write_limit": 2**32,
            },
        )
        self._lock = asyncio.Lock()
        self.last_block_hash = None
        self.runtime_cache = RuntimeCache()
        self.config = {
            "use_remote_preset": use_remote_preset,
            "auto_discover": auto_discover,
            "auto_reconnect": auto_reconnect,
            "rpc_methods": None,
            "strict_scale_decode": True,
        }
        self.initialized = False
        self._forgettable_task = None

    async def __aenter__(self):
        await self.initialize()

    @no_wrap
    async def initialize(self):
        """
        Initialize the attached substrate object
        """
        async with self._lock:
            if not self.substrate:
                self.substrate = SubstrateInterface(
                    ss58_format=bittensor.__ss58_format__,
                    use_remote_preset=True,
                    url=self.chain_endpoint,
                    type_registry=bittensor.__type_registry__,
                )
                self.__chain = (await self.rpc_request("system_chain", [])).get(
                    "result"
                )
            self.initialized = True

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.ws.close()

    @property
    def chain(self):
        """
        Returns the substrate chain currently associated with object
        """
        return self.__chain

    async def get_storage_item(self, module: str, storage_function: str):
        if not self.substrate.metadata:
            self.substrate.init_runtime()
        metadata_pallet = self.substrate.metadata.get_metadata_pallet(module)
        storage_item = metadata_pallet.get_storage_function(storage_function)
        return storage_item

    def _get_current_block_hash(self, block_hash: Optional[str], reuse: bool):
        return block_hash if block_hash else (self.last_block_hash if reuse else None)

    async def _init_runtime(
        self, block_hash: Optional[str] = None, block_id: Optional[int] = None
    ) -> Runtime:
        """
        Not currently working or used. Will be added in the future.
        """

        def get_runtime() -> Optional[Runtime]:
            if block_id:
                return self.runtime_cache.by_block_id(block_id)

            if block_hash:
                return self.runtime_cache.by_hash(block_hash)

        if block_id and block_hash:
            raise ValueError("Cannot provide block_hash and block_id at the same time")

        if runtime := get_runtime():
            return runtime

        runtime = Runtime(
            self.chain, self.substrate.runtime_config, self.substrate.metadata
        )

        if block_id is not None:
            block_hash = await self.get_block_hash(block_id)

        if not block_hash:
            block_hash = await self.get_chain_head()

        runtime.block_hash = block_hash
        runtime.block_id = block_id

        # In fact calls and storage functions are decoded against runtime of previous block, therefor retrieve
        # metadata and apply type registry of runtime of parent block
        block_header = await self.rpc_request("chain_getHeader", [runtime.block_hash])

        if block_header["result"] is None:
            raise BlockNotFound(f'Block not found for "{runtime.block_hash}"')

        parent_block_hash = block_header["result"]["parentHash"]

        if (
            parent_block_hash
            == "0x0000000000000000000000000000000000000000000000000000000000000000"
        ):
            runtime_block_hash = runtime.block_hash
        else:
            runtime_block_hash = parent_block_hash

        runtime_info = await self.get_block_runtime_version(
            block_hash=runtime_block_hash
        )

        if runtime_info is None:
            raise SubstrateRequestException(
                f"No runtime information for block '{block_hash}'"
            )

        runtime.runtime_version = runtime_info.get("specVersion")
        runtime.transaction_version = runtime_info.get("transactionVersion")

        if runtime.runtime_version in self.runtime_cache.metadata_cache:
            # Get metadata from cache
            # self.debug_message('Retrieved metadata for {} from memory'.format(self.runtime_version))
            runtime.metadata = self.runtime_cache.metadata_cache[
                runtime.runtime_version
            ]
        else:
            runtime.metadata = (
                await self.get_block_metadata(
                    block_hash=runtime_block_hash, decode=True
                )
            )["result"]

            # Update metadata cache
            self.runtime_cache.metadata_cache[runtime.runtime_version] = (
                runtime.metadata
            )

        # Update type registry
        runtime.reload_type_registry(
            use_remote_preset=self.config.get("use_remote_preset"),
            auto_discover=self.config.get("auto_discover"),
        )

        # Check if PortableRegistry is present in metadata (V14+), otherwise fall back on legacy type registry (<V14)
        if runtime.implements_scaleinfo:
            runtime.runtime_config.add_portable_registry(runtime.metadata)

        # Set active runtime version
        runtime.runtime_config.set_active_spec_version_id(runtime.runtime_version)

        # Check and apply runtime constants
        ss58_prefix_constant = await self.get_constant(
            "System", "SS58Prefix", block_hash=block_hash
        )

        if ss58_prefix_constant:
            runtime.ss58_format = ss58_prefix_constant.value

        # Set runtime compatibility flags
        try:
            _ = runtime.runtime_config.create_scale_object(
                "sp_weights::weight_v2::Weight"
            )
            runtime.config["is_weight_v2"] = True
            runtime.runtime_config.update_type_registry_types(
                {"Weight": "sp_weights::weight_v2::Weight"}
            )
        except NotImplementedError:
            runtime.config["is_weight_v2"] = False
            runtime.runtime_config.update_type_registry_types({"Weight": "WeightV1"})

        return runtime

    async def init_runtime(
        self, block_hash: Optional[str] = None, block_id: Optional[int] = None
    ) -> Runtime:
        """
        This method is used by all other methods that deals with metadata and types defined in the type registry.
        It optionally retrieves the block_hash when block_id is given and sets the applicable metadata for that
        block_hash. Also, it applies all the versioned types at the time of the block_hash.

        Because parsing of metadata and type registry is quite heavy, the result will be cached per runtime id.
        In the future there could be support for caching backends like Redis to make this cache more persistent.

        :param block_hash: optional block hash, should not be specified if block_id is
        :param block_id: optional block id, should not be specified if block_hash is

        :returns: Runtime object
        """
        async with self._lock:
            await asyncio.get_event_loop().run_in_executor(
                None, self.substrate.init_runtime, block_hash, block_id
            )
            return Runtime(
                self.chain, self.substrate.runtime_config, self.substrate.metadata
            )

    async def get_block_runtime_version(self, block_hash: str) -> dict:
        """
        Retrieve the runtime version id of given block_hash
        """
        response = await self.rpc_request("state_getRuntimeVersion", [block_hash])
        return response.get("result")

    async def get_block_metadata(
        self, block_hash=None, decode=True
    ) -> Union[dict, ScaleType]:
        """
        A pass-though to existing JSONRPC method `state_getMetadata`.

        Parameters
        ----------
        block_hash
        decode: True for decoded version

        Returns
        -------

        """
        params = None
        if decode and not self.substrate.runtime_config:
            raise ValueError(
                "Cannot decode runtime configuration without a supplied runtime_config"
            )

        if block_hash:
            params = [block_hash]
        response = await self.rpc_request("state_getMetadata", params)

        if "error" in response:
            raise SubstrateRequestException(response["error"]["message"])

        if response.get("result") and decode:
            metadata_decoder = self.substrate.runtime_config.create_scale_object(
                "MetadataVersioned", data=ScaleBytes(response.get("result"))
            )
            metadata_decoder.decode()

            return metadata_decoder

        return response

    async def _preprocess(
        self,
        query_for: Optional[list],
        block_hash: str,
        storage_function: str,
        module: str,
    ) -> Preprocessed:
        """
        Creates a Preprocessed data object for passing to ``_make_rpc_request``
        """
        params = query_for if query_for else []
        # Search storage call in metadata
        metadata_pallet = self.substrate.metadata.get_metadata_pallet(module)

        if not metadata_pallet:
            raise Exception(f'Pallet "{module}" not found')

        storage_item = metadata_pallet.get_storage_function(storage_function)

        if not metadata_pallet or not storage_item:
            raise Exception(f'Storage function "{module}.{storage_function}" not found')

        # SCALE type string of value
        param_types = storage_item.get_params_type_string()
        value_scale_type = storage_item.get_value_type_string()

        if len(params) != len(param_types):
            raise ValueError(
                f"Storage function requires {len(param_types)} parameters, {len(params)} given"
            )

        storage_key = StorageKey.create_from_storage_function(
            module,
            storage_item.value["name"],
            params,
            runtime_config=self.substrate.runtime_config,
            metadata=self.substrate.metadata,
        )
        method = (
            "state_getStorageAt"
            if self.substrate.supports_rpc_method("state_getStorageAt")
            else "state_getStorage"
        )
        return Preprocessed(
            str(query_for),
            method,
            [storage_key.to_hex(), block_hash],
            value_scale_type,
            storage_item,
        )

    async def _process_response(
        self,
        response: dict,
        subscription_id: Union[int, str],
        value_scale_type: str,
        storage_item: Optional[ScaleType] = None,
        runtime: Optional[Runtime] = None,
        result_handler: Optional[ResultHandler] = None,
    ) -> tuple[Union[ScaleType, dict], bool]:
        obj = response

        if value_scale_type:
            if not runtime:
                async with self._lock:
                    runtime = Runtime(
                        self.chain,
                        self.substrate.runtime_config,
                        self.substrate.metadata,
                    )
            if response.get("result") is not None:
                query_value = response.get("result")
            elif storage_item.value["modifier"] == "Default":
                # Fallback to default value of storage function if no result
                query_value = storage_item.value_object["default"].value_object
            else:
                # No result is interpreted as an Option<...> result
                value_scale_type = f"Option<{value_scale_type}>"
                query_value = storage_item.value_object["default"].value_object

            obj = runtime.runtime_config.create_scale_object(
                type_string=value_scale_type,
                data=ScaleBytes(query_value),
                metadata=runtime.metadata,
            )
            obj.decode(check_remaining=True)
            obj.meta_info = {"result_found": response.get("result") is not None}
        if asyncio.iscoroutinefunction(result_handler):
            # For multipart responses as a result of subscriptions.
            message, bool_result = await result_handler(obj, subscription_id)
            return message, bool_result
        return obj, True

    async def _make_rpc_request(
        self,
        payloads: list[dict],
        value_scale_type: Optional[str] = None,
        storage_item: Optional[ScaleType] = None,
        runtime: Optional[Runtime] = None,
        result_handler: Optional[ResultHandler] = None,
    ) -> RequestManager.RequestResults:
        request_manager = RequestManager(payloads)

        subscription_added = False

        async with self.ws as ws:
            for item in payloads:
                item_id = await ws.send(item["payload"])
                request_manager.add_request(item_id, item["id"])

            while True:
                for item_id in request_manager.response_map.keys():
                    if (
                        item_id not in request_manager.responses
                        or asyncio.iscoroutinefunction(result_handler)
                    ):
                        if response := await ws.retrieve(item_id):
                            if (
                                asyncio.iscoroutinefunction(result_handler)
                                and not subscription_added
                            ):
                                # handles subscriptions, overwrites the previous mapping of {item_id : payload_id}
                                # with {subscription_id : payload_id}
                                item_id = request_manager.overwrite_request(
                                    item_id, response["result"]
                                )
                            decoded_response, complete = await self._process_response(
                                response,
                                item_id,
                                value_scale_type,
                                storage_item,
                                runtime,
                                result_handler,
                            )
                            request_manager.add_response(
                                item_id, decoded_response, complete
                            )
                    if (
                        asyncio.iscoroutinefunction(result_handler)
                        and not subscription_added
                    ):
                        subscription_added = True
                        break

                if request_manager.is_complete:
                    break

        return request_manager.get_results()

    @staticmethod
    def make_payload(id_: str, method: str, params: list) -> dict:
        """
        Creates a payload for making an rpc_request with _make_rpc_request

        :param id_: a unique name you would like to give to this request
        :param method: the method in the RPC request
        :param params: the params in the RPC request

        :return: the payload dict
        """
        return {
            "id": id_,
            "payload": {"jsonrpc": "2.0", "method": method, "params": params},
        }

    async def rpc_request(
        self,
        method: str,
        params: list,
        block_hash: Optional[str] = None,
        reuse_block_hash: bool = False,
    ) -> Any:
        """
        Makes an RPC request to the subtensor. Use this only if ``self.query`` and ``self.query_multiple`` and
        ``self.query_map`` do not meet your needs.

        :param method: str the method in the RPC request
        :param params: list of the params in the RPC request
        :param block_hash: optional str, the hash of the block — only supply this if not supplying the block
                           hash in the params, and not reusing the block hash
        :param reuse_block_hash: optional bool, whether to reuse the block hash in the params — only mark as True
                                 if not supplying the block hash in the params, or via the `block_hash` parameter

        :return: the response from the RPC request
        """
        block_hash = self._get_current_block_hash(block_hash, reuse_block_hash)
        payloads = [
            self.make_payload(
                "rpc_request",
                method,
                params + [block_hash] if block_hash else params,
            )
        ]
        runtime = Runtime(
            self.chain, self.substrate.runtime_config, self.substrate.metadata
        )
        result = await self._make_rpc_request(payloads, runtime=runtime)
        if "error" in result["rpc_request"][0]:
            raise SubstrateRequestException(
                result["rpc_request"][0]["error"]["message"]
            )
        if "result" in result["rpc_request"][0]:
            return result["rpc_request"][0]
        else:
            raise SubstrateRequestException(result["rpc_request"][0])

    async def get_block_hash(self, block_id: int) -> str:
        return (await self.rpc_request("chain_getBlockHash", [block_id]))["result"]

    async def get_chain_head(self) -> str:
        return (await self.rpc_request("chain_getHead", []))["result"]

    async def compose_call(
        self,
        call_module: str,
        call_function: str,
        call_params: dict = None,
        block_hash: str = None,
    ) -> GenericCall:
        """
        Composes a call payload which can be used in an extrinsic.

        :param call_module: Name of the runtime module e.g. Balances
        :param call_function: Name of the call function e.g. transfer
        :param call_params: This is a dict containing the params of the call. e.g.
                            `{'dest': 'EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk', 'value': 1000000000000}`
        :param block_hash: Use metadata at given block_hash to compose call

        :return: A composed call
        """
        runtime = await self.init_runtime(block_hash=block_hash)
        call = runtime.runtime_config.create_scale_object(
            type_string="Call", metadata=runtime.metadata
        )
        call.encode(
            {
                "call_module": call_module,
                "call_function": call_function,
                "call_args": call_params or {},
            }
        )

        return call

    async def query_multiple(
        self,
        params: list,
        storage_function: str,
        module: str,
        block_hash: Optional[str] = None,
        reuse_block_hash: bool = False,
    ) -> RequestManager.RequestResults:
        """
        Queries the subtensor. Only use this when making multiple queries, else use ``self.query``
        """
        # By allowing for specifying the block hash, users, if they have multiple query types they want
        # to do, can simply query the block hash first, and then pass multiple query_subtensor calls
        # into an asyncio.gather, with the specified block hash
        block_hash = (
            block_hash
            if block_hash
            else (
                self.last_block_hash
                if reuse_block_hash
                else await self.get_chain_head()
            )
        )
        self.last_block_hash = block_hash
        runtime = await self.init_runtime(block_hash=block_hash)
        preprocessed: tuple[Preprocessed] = await asyncio.gather(
            *[
                self._preprocess([x], block_hash, storage_function, module)
                for x in params
            ]
        )
        all_info = [
            self.make_payload(item.queryable, item.method, item.params)
            for item in preprocessed
        ]
        # These will always be the same throughout the preprocessed list, so we just grab the first one
        value_scale_type = preprocessed[0].value_scale_type
        storage_item = preprocessed[0].storage_item

        responses = await self._make_rpc_request(
            all_info, value_scale_type, storage_item, runtime
        )
        return responses

    async def create_scale_object(
        self,
        type_string: str,
        data: ScaleBytes = None,
        block_hash: str = None,
        **kwargs,
    ) -> "ScaleType":
        """
        Convenience method to create a SCALE object of type `type_string`, this will initialize the runtime
        automatically at moment of `block_hash`, or chain tip if omitted.

        :param type_string: str Name of SCALE type to create
        :param data: ScaleBytes Optional ScaleBytes to decode
        :param block_hash: Optional block hash for moment of decoding, when omitted the chain tip will be used
        :param kwargs: keyword args for the Scale Type constructor

        :return: The created Scale Type object
        """
        runtime = await self.init_runtime(block_hash=block_hash)
        if "metadata" not in kwargs:
            kwargs["metadata"] = runtime.metadata

        return runtime.runtime_config.create_scale_object(
            type_string, data=data, **kwargs
        )

    async def create_signed_extrinsic(
        self,
        call: GenericCall,
        keypair: Keypair,
        era: dict = None,
        nonce: int = None,
        tip: int = 0,
        tip_asset_id: int = None,
        signature: Union[bytes, str] = None,
    ) -> "GenericExtrinsic":
        """
        Creates an extrinsic signed by given account details

        :param call: GenericCall to create extrinsic for
        :param keypair: Keypair used to sign the extrinsic
        :param era: Specify mortality in blocks in follow format:
                    {'period': [amount_blocks]} If omitted the extrinsic is immortal
        :param nonce: nonce to include in extrinsics, if omitted the current nonce is retrieved on-chain
        :param tip: The tip for the block author to gain priority during network congestion
        :param tip_asset_id: Optional asset ID with which to pay the tip
        :param signature: Optionally provide signature if externally signed

        :return: The signed Extrinsic
        """
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.substrate.create_signed_extrinsic(
                call=call,
                keypair=keypair,
                era=era,
                nonce=nonce,
                tip=tip,
                tip_asset_id=tip_asset_id,
                signature=signature,
            ),
        )

    async def runtime_call(
        self,
        api: str,
        method: str,
        params: Union[list, dict] = None,
        block_hash: str = None,
    ) -> ScaleType:
        """
        Calls a runtime API method

        :param api: Name of the runtime API e.g. 'TransactionPaymentApi'
        :param method: Name of the method e.g. 'query_fee_details'
        :param params: List of parameters needed to call the runtime API
        :param block_hash: Hash of the block at which to make the runtime API call

        :return: ScaleType from the runtime call
        """
        await self.init_runtime()

        if params is None:
            params = {}

        async with self._lock:
            try:
                runtime_call_def = self.substrate.runtime_config.type_registry[
                    "runtime_api"
                ][api]["methods"][method]
                runtime_api_types = self.substrate.runtime_config.type_registry[
                    "runtime_api"
                ][api].get("types", {})
            except KeyError:
                raise ValueError(
                    f"Runtime API Call '{api}.{method}' not found in registry"
                )

            if isinstance(params, list) and len(params) != len(
                runtime_call_def["params"]
            ):
                raise ValueError(
                    f"Number of parameter provided ({len(params)}) does not "
                    f"match definition {len(runtime_call_def['params'])}"
                )

            # Add runtime API types to registry
            self.substrate.runtime_config.update_type_registry_types(runtime_api_types)
            runtime = Runtime(
                self.chain, self.substrate.runtime_config, self.substrate.metadata
            )

        # Encode params
        param_data = ScaleBytes(bytes())
        for idx, param in enumerate(runtime_call_def["params"]):
            scale_obj = runtime.runtime_config.create_scale_object(param["type"])
            if isinstance(params, list):
                param_data += scale_obj.encode(params[idx])
            else:
                if param["name"] not in params:
                    raise ValueError(f"Runtime Call param '{param['name']}' is missing")

                param_data += scale_obj.encode(params[param["name"]])

        # RPC request
        result_data = await self.rpc_request(
            "state_call", [f"{api}_{method}", str(param_data), block_hash]
        )

        # Decode result
        result_obj = runtime.runtime_config.create_scale_object(
            runtime_call_def["type"]
        )
        result_obj.decode(
            ScaleBytes(result_data["result"]),
            check_remaining=self.config.get("strict_scale_decode"),
        )

        return result_obj

    async def get_account_nonce(self, account_address: str) -> int:
        """
        Returns current nonce for given account address

        :param account_address: SS58 formatted address

        :return: Nonce for given account address
        """
        nonce_obj = await self.runtime_call(
            "AccountNonceApi", "account_nonce", [account_address]
        )
        return nonce_obj.value

    async def get_constant(
        self, module_name: str, constant_name: str, block_hash: Optional[str] = None
    ) -> Optional["ScaleType"]:
        """
        Returns the decoded `ScaleType` object of the constant for given module name, call function name and block_hash
        (or chaintip if block_hash is omitted)

        Parameters
        ----------
        :param module_name: Name of the module to query
        :param constant_name: Name of the constant to query
        :param block_hash: Hash of the block at which to make the runtime API call

        :return: ScaleType from the runtime call
        """
        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.substrate.get_constant(
                    module_name, constant_name, block_hash
                ),
            )

    async def get_payment_info(
        self, call: GenericCall, keypair: Keypair
    ) -> dict[str, Any]:
        """
        Retrieves fee estimation via RPC for given extrinsic

        Parameters
        ----------
        call: Call object to estimate fees for
        keypair: Keypair of the sender, does not have to include private key because no valid signature is required

        Returns
        -------
        Dict with payment info

        E.g. `{'class': 'normal', 'partialFee': 151000000, 'weight': {'ref_time': 143322000}}`

        """

        # Check requirements
        if not isinstance(call, GenericCall):
            raise TypeError("'call' must be of type Call")

        if not isinstance(keypair, Keypair):
            raise TypeError("'keypair' must be of type Keypair")

        # No valid signature is required for fee estimation
        signature = "0x" + "00" * 64

        # Create extrinsic
        extrinsic = await self.create_signed_extrinsic(
            call=call, keypair=keypair, signature=signature
        )
        async with self._lock:
            extrinsic_len = self.substrate.runtime_config.create_scale_object("u32")
        extrinsic_len.encode(len(extrinsic.data))

        result = await self.runtime_call(
            "TransactionPaymentApi", "query_info", [extrinsic, extrinsic_len]
        )

        return result.value

    async def query(
        self,
        module: str,
        storage_function: str,
        params: Optional[list] = None,
        block_hash: Optional[str] = None,
        raw_storage_key: bytes = None,
        subscription_handler=None,
        reuse_block_hash: bool = False,
    ) -> "ScaleType":
        """
        Queries subtensor. This should only be used when making a single request. For multiple requests,
        you should use ``self.query_multiple``
        """
        block_hash = (
            block_hash
            if block_hash
            else (
                self.last_block_hash
                if reuse_block_hash
                else await self.get_chain_head()
            )
        )
        self.last_block_hash = block_hash
        runtime = await self.init_runtime(block_hash=block_hash)
        preprocessed: Preprocessed = await self._preprocess(
            params, block_hash, storage_function, module
        )
        payload = [
            self.make_payload(
                preprocessed.queryable, preprocessed.method, preprocessed.params
            )
        ]
        value_scale_type = preprocessed.value_scale_type
        storage_item = preprocessed.storage_item

        responses = await self._make_rpc_request(
            payload,
            value_scale_type,
            storage_item,
            runtime,
            result_handler=subscription_handler,
        )
        return responses[preprocessed.queryable][0]

    async def query_map(
        self,
        module: str,
        storage_function: str,
        params: Optional[list] = None,
        block_hash: Optional[str] = None,
        max_results: int = None,
        start_key: str = None,
        page_size: int = 100,
        ignore_decoding_errors: bool = True,
        reuse_block_hash: bool = False,
    ) -> "QueryMapResult":
        """
        Iterates over all key-pairs located at the given module and storage_function. The storage
        item must be a map.

        Example:

        ```
        result = substrate.query_map('System', 'Account', max_results=100)

        for account, account_info in result:
            print(f"Free balance of account '{account.value}': {account_info.value['data']['free']}")
        ```

        :param module: The module name in the metadata, e.g. System or Balances.
        :param storage_function: The storage function name, e.g. Account or Locks.
        :param params: The input parameters in case of for example a `DoubleMap` storage function
        :param block_hash: Optional block hash for result at given block, when left to None the chain tip will be used.
        :param max_results: the maximum of results required, if set the query will stop fetching results when number is
                            reached
        :param start_key: The storage key used as offset for the results, for pagination purposes
        :param page_size: The results are fetched from the node RPC in chunks of this size
        :param ignore_decoding_errors: When set this will catch all decoding errors, set the item to None and continue
                                       decoding
        :param reuse_block_hash: use True if you wish to make the query using the last-used block hash. Do not mark True
                                 if supplying a block_hash

        :return: QueryMapResult object
        """
        params = params or []
        block_hash = (
            block_hash
            if block_hash
            else (
                self.last_block_hash
                if reuse_block_hash
                else await self.get_chain_head()
            )
        )
        self.last_block_hash = block_hash
        runtime = await self.init_runtime(block_hash=block_hash)

        metadata_pallet = runtime.metadata.get_metadata_pallet(module)
        if not metadata_pallet:
            raise ValueError(f'Pallet "{module}" not found')

        storage_item = metadata_pallet.get_storage_function(storage_function)

        if not metadata_pallet or not storage_item:
            raise ValueError(
                f'Storage function "{module}.{storage_function}" not found'
            )

        value_type = storage_item.get_value_type_string()
        param_types = storage_item.get_params_type_string()
        key_hashers = storage_item.get_param_hashers()

        # Check MapType conditions
        if len(param_types) == 0:
            raise ValueError("Given storage function is not a map")
        if len(params) > len(param_types) - 1:
            raise ValueError(
                f"Storage function map can accept max {len(param_types) - 1} parameters, {len(params)} given"
            )

        # Generate storage key prefix
        storage_key = StorageKey.create_from_storage_function(
            module,
            storage_item.value["name"],
            params,
            runtime_config=runtime.runtime_config,
            metadata=runtime.metadata,
        )
        prefix = storage_key.to_hex()

        if not start_key:
            start_key = prefix

        # Make sure if the max result is smaller than the page size, adjust the page size
        if max_results is not None and max_results < page_size:
            page_size = max_results

        # Retrieve storage keys
        response = await self.rpc_request(
            method="state_getKeysPaged",
            params=[prefix, page_size, start_key, block_hash],
        )

        if "error" in response:
            raise SubstrateRequestException(response["error"]["message"])

        result_keys = response.get("result")

        result = []
        last_key = None

        def concat_hash_len(key_hasher: str) -> int:
            """
            Helper function to avoid if statements
            """
            if key_hasher == "Blake2_128Concat":
                return 16
            elif key_hasher == "Twox64Concat":
                return 8
            elif key_hasher == "Identity":
                return 0
            else:
                raise ValueError("Unsupported hash type")

        if len(result_keys) > 0:
            last_key = result_keys[-1]

            # Retrieve corresponding value
            response = await self.rpc_request(
                method="state_queryStorageAt", params=[result_keys, block_hash]
            )

            if "error" in response:
                raise SubstrateRequestException(response["error"]["message"])

            for result_group in response["result"]:
                for item in result_group["changes"]:
                    try:
                        # Determine type string
                        key_type_string = []
                        for n in range(len(params), len(param_types)):
                            key_type_string.append(
                                f"[u8; {concat_hash_len(key_hashers[n])}]"
                            )
                            key_type_string.append(param_types[n])

                        item_key_obj = self.substrate.decode_scale(
                            type_string=f"({', '.join(key_type_string)})",
                            scale_bytes="0x" + item[0][len(prefix) :],
                            return_scale_obj=True,
                            block_hash=block_hash,
                        )

                        # strip key_hashers to use as item key
                        if len(param_types) - len(params) == 1:
                            item_key = item_key_obj.value_object[1]
                        else:
                            item_key = tuple(
                                item_key_obj.value_object[key + 1]
                                for key in range(len(params), len(param_types) + 1, 2)
                            )

                    except Exception as _:
                        if not ignore_decoding_errors:
                            raise
                        item_key = None

                    try:
                        item_value = self.substrate.decode_scale(
                            type_string=value_type,
                            scale_bytes=item[1],
                            return_scale_obj=True,
                            block_hash=block_hash,
                        )
                    except Exception as _:
                        if not ignore_decoding_errors:
                            raise
                        item_value = None

                    result.append([item_key, item_value])

        return QueryMapResult(
            records=result,
            page_size=page_size,
            module=module,
            storage_function=storage_function,
            params=params,
            block_hash=block_hash,
            substrate=self.substrate,
            last_key=last_key,
            max_results=max_results,
            ignore_decoding_errors=ignore_decoding_errors,
        )

    async def submit_extrinsic(
        self,
        extrinsic: GenericExtrinsic,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
    ) -> "ExtrinsicReceipt":
        """
        Submit an extrinsic to the connected node, with the possibility to wait until the extrinsic is included
         in a block and/or the block is finalized. The receipt returned provided information about the block and
         triggered events

        Parameters
        ----------
        extrinsic: Extrinsic The extrinsic to be sent to the network
        wait_for_inclusion: wait until extrinsic is included in a block (only works for websocket connections)
        wait_for_finalization: wait until extrinsic is finalized (only works for websocket connections)

        Returns
        -------
        ExtrinsicReceipt

        """

        # Check requirements
        if not isinstance(extrinsic, GenericExtrinsic):
            raise TypeError("'extrinsic' must be of type Extrinsics")

        async def result_handler(message: dict, subscription_id) -> tuple[dict, bool]:
            """
            Result handler function passed as an arg to _make_rpc_request as the result_handler
            to handle the results of the extrinsic rpc call, which are multipart, and require
            subscribing to the message

            :param message: message received from the rpc call
            :param subscription_id: subscription id received from the initial rpc call for the subscription

            :returns: tuple containing the dict of the block info for the subscription, and bool for whether
                      the subscription is completed.
            """
            # Check if extrinsic is included and finalized
            if "params" in message and isinstance(message["params"]["result"], dict):
                # Convert result enum to lower for backwards compatibility
                message_result = {
                    k.lower(): v for k, v in message["params"]["result"].items()
                }

                if "finalized" in message_result and wait_for_finalization:
                    # Created as a task because we don't actually care about the result
                    self._forgettable_task = asyncio.create_task(
                        self.rpc_request("author_unwatchExtrinsic", [subscription_id])
                    )
                    return {
                        "block_hash": message_result["finalized"],
                        "extrinsic_hash": "0x{}".format(extrinsic.extrinsic_hash.hex()),
                        "finalized": True,
                    }, True
                elif (
                    "inblock" in message_result
                    and wait_for_inclusion
                    and not wait_for_finalization
                ):
                    # Created as a task because we don't actually care about the result
                    self._forgettable_task = asyncio.create_task(
                        self.rpc_request(
                            "author_unwatchExtrinsic", [subscription_id]
                        )
                    )
                    return {
                        "block_hash": message_result["inblock"],
                        "extrinsic_hash": "0x{}".format(extrinsic.extrinsic_hash.hex()),
                        "finalized": False,
                    }, True
            return message, False

        if wait_for_inclusion or wait_for_finalization:
            responses = (
                await self._make_rpc_request(
                    [
                        self.make_payload(
                            "rpc_request",
                            "author_submitAndWatchExtrinsic",
                            [str(extrinsic.data)],
                        )
                    ],
                    result_handler=result_handler,
                )
            )["rpc_request"]
            response = next(
                (r for r in responses if "block_hash" in r and "extrinsic_hash" in r),
                None,
            )

            if not response:
                raise SubstrateRequestException(responses)

            # Also, this will be a multipart response, so maybe should change to everything after the first response?
            # The following code implies this will be a single response after the initial subscription id.
            result = ExtrinsicReceipt(
                substrate=self.substrate,
                extrinsic_hash=response["extrinsic_hash"],
                block_hash=response["block_hash"],
                finalized=response["finalized"],
            )

        else:
            response = await self.rpc_request(
                "author_submitExtrinsic", [str(extrinsic.data)]
            )

            if "result" not in response:
                raise SubstrateRequestException(response.get("error"))

            result = ExtrinsicReceipt(
                substrate=self.substrate, extrinsic_hash=response["result"]
            )

        return result

    async def get_metadata_call_function(
        self, module_name: str, call_function_name: str, block_hash: str = None
    ) -> list:
        """
        Retrieves a list of all call functions in metadata active for given block_hash (or chaintip if block_hash
        is omitted)

        :param module_name: name of the module
        :param call_function_name: name of the call function
        :param block_hash: optional block hash

        :return: list of call functions
        """
        runtime = await self.init_runtime(block_hash=block_hash)

        for pallet in runtime.metadata.pallets:
            if pallet.name == module_name and pallet.calls:
                for call in pallet.calls:
                    if call.name == call_function_name:
                        return call

    async def get_block_number(self, block_hash: str) -> int:
        """Async version of `substrateinterface.base.get_block_number` method."""
        response = await self.rpc_request("chain_getHeader", [block_hash])

        if "error" in response:
            raise SubstrateRequestException(response["error"]["message"])

        elif "result" in response:
            if response["result"]:
                return int(response["result"]["number"], 16)

    async def close(self):
        """
        Closes the substrate connection, and the websocket connection.
        """
        self.substrate.close()
        try:
            await self.ws.ws.close()
        except AttributeError:
            pass


if __name__ == "__main__":
    pass
