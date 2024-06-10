import asyncio
import functools
from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Optional, Any, Union

from substrateinterface.base import (
    SubstrateInterface,
    Keypair,
    QueryMapResult,
    ExtrinsicReceipt,
)
from substrateinterface.storage import StorageKey
from substrateinterface.exceptions import SubstrateRequestException
from scalecodec.base import ScaleBytes, ScaleType
from scalecodec.types import GenericMetadataVersioned, GenericCall, GenericExtrinsic
import websockets

import bittensor

TEST_CHAIN_ENDPOINT = "wss://test.finney.opentensor.ai:443"


@dataclass
class Preprocessed:
    queryable: str
    method: str
    params: list
    value_scale_type: str
    storage_item: ScaleType


def ensure_initialized(func):
    """Wrapper for initialization of SubstrateInterface before connection."""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        async with self._lock:
            if not self.substrate:
                await self.initialize()
        return await func(self, *args, **kwargs)

    return wrapper


class AsyncSubstrateInterface:
    runtime = None
    substrate = None

    def __init__(self, chain_endpoint: str, ws_options=None):
        self.init_runtime = None
        self.chain_endpoint = chain_endpoint
        self.ws = Websocket(chain_endpoint, options=ws_options, shutdown_timer=None)
        self._lock = asyncio.Lock()
        self.substrate: Optional[SubstrateInterface] = None

    async def initialize(self):
        if not self.substrate:
            self.substrate = SubstrateInterface(
                ss58_format=bittensor.__ss58_format__,
                use_remote_preset=True,
                url=self.chain_endpoint,
                type_registry=bittensor.__type_registry__,
            )
            # self.init_runtime = self.substrate.init_runtime

    async def init_runtime(self): ...

    async def get_storage_item(self, module: str, storage_function: str):
        if not self.substrate.metadata:
            self.substrate.init_runtime()
        metadata_pallet = self.substrate.metadata.get_metadata_pallet(module)
        storage_item = metadata_pallet.get_storage_function(storage_function)
        return storage_item

    async def _preprocess(
        self,
        query_for: str,
        block_hash: str,
        storage_function: str,
        module: str,
    ) -> Preprocessed:
        """
        Creates a Preprocessed data object for passing to ``make_call``
        """
        params = [query_for]
        # Search storage call in metadata
        metadata_pallet = self.substrate.metadata.get_metadata_pallet(module)

        if not metadata_pallet:
            raise Exception(f'Pallet "{module}" not found')

        storage_item = metadata_pallet.get_storage_function(storage_function)

        if not metadata_pallet or not storage_item:
            raise Exception(f'Storage function "{module}.{storage_function}" not found')

        # SCALE type string of value
        value_scale_type = storage_item.get_value_type_string()

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
            query_for,
            method,
            [storage_key.to_hex(), block_hash],
            value_scale_type,
            storage_item,
        )

    async def _process_response(
        self,
        response: dict,
        value_scale_type: str,
        storage_item: Optional[ScaleType] = None,
        metadata: Optional[GenericMetadataVersioned] = None,
    ) -> tuple[dict, bool]:
        # TODO add logic for handling multipart responses
        if value_scale_type:
            if response.get("result") is not None:
                query_value = response.get("result")
            elif storage_item.value["modifier"] == "Default":
                # Fallback to default value of storage function if no result
                query_value = storage_item.value_object["default"].value_object
            else:
                # No result is interpreted as an Option<...> result
                value_scale_type = f"Option<{value_scale_type}>"
                query_value = storage_item.value_object["default"].value_object

            obj = self.substrate.runtime_config.create_scale_object(
                type_string=value_scale_type,
                data=ScaleBytes(query_value),
                metadata=metadata,
            )
            obj.decode(check_remaining=True)
            obj.meta_info = {"result_found": response.get("result") is not None}
            return obj, True
        return response, True

    async def _make_rpc_request(
        self,
        payloads: list[dict],
        value_scale_type: Optional[str] = None,
        storage_item: Optional[ScaleType] = None,
        metadata: Optional[GenericMetadataVersioned] = None,
    ) -> dict:
        request_manager = RequestManager(payloads)

        async with self.ws as ws:
            for item in payloads:
                item_id = await ws.send(item["payload"])
                await request_manager.add_request(item_id, item["id"])

            while True:
                for item_id in request_manager.response_map.keys():
                    if item_id not in request_manager.responses:
                        if response := await ws.retrieve(item_id):
                            decoded_response, complete = await self._process_response(
                                response, value_scale_type, storage_item, metadata
                            )
                            await request_manager.add_response(
                                item_id, decoded_response, complete
                            )

                if await request_manager.is_complete():
                    break

        return await request_manager.get_results()

    @ensure_initialized
    async def rpc_request(
        self, method: str, params: list, block_hash: Optional[str] = None
    ) -> Any:
        payloads = [
            {
                "id": "rpc_request",
                "payload": {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params + [block_hash] if block_hash else params,
                },
            }
        ]
        result = await self._make_rpc_request(payloads)
        if "error" in result["rpc_request"][0]:
            raise SubstrateRequestException(
                result["rpc_request"][0]["error"]["message"]
            )
        return result["rpc_request"][0]

    async def get_block_hash(self, block_id: int) -> str:
        response = await self.rpc_request("chain_getBlockHash", [block_id])

        if "error" in response:
            raise SubstrateRequestException(response["error"]["message"])
        else:
            return response.get("result")

    async def get_chain_head(self) -> str:
        result = await self.rpc_request("chain_getHead", [])
        return result

    async def query_subtensor(
        self,
        query_for: list,
        storage_function: str,
        module: str,
        block_hash: Optional[str] = None,
    ) -> dict:
        # By allowing for specifying the block hash, users, if they have multiple query types they want
        # to do, can simply query the block hash first, and then pass multiple query_subtensor calls
        # into an asyncio.gather, with the specified block hash
        block_hash = block_hash or await self.get_chain_head()
        self.substrate.init_runtime(block_hash=block_hash)  # TODO
        preprocessed: tuple[Preprocessed] = await asyncio.gather(
            *[
                self._preprocess(x, block_hash, storage_function, module)
                for x in query_for
            ]
        )
        all_info = [
            {
                "id": item.queryable,
                "payload": {
                    "jsonrpc": "2.0",
                    "method": item.method,
                    "params": item.params,
                },
            }
            for item in preprocessed
        ]
        # These will always be the same throughout the preprocessed list, so we just grab the first one
        value_scale_type = preprocessed[0].value_scale_type
        storage_item = preprocessed[0].storage_item

        responses = await self._make_rpc_request(
            all_info,
            value_scale_type,
            storage_item,
            self.substrate.metadata,  # individual because I would like to break this out from SSI
        )
        return responses

    async def query_map_subtensor(
        self, module: str, storage_function: str, block_hash: Optional[str] = None
    ) -> dict:
        raise NotImplementedError()

    async def compose_call(
        self,
        call_module: str,
        call_function: str,
        call_params: dict[str, Any] = None,
        block_hash: str = None,
    ) -> "GenericCall":
        raise NotImplementedError()

    async def create_scale_object(
        self,
        type_string: str,
        data: ScaleBytes = None,
        block_hash: str = None,
        **kwargs,
    ) -> "ScaleType":
        raise NotImplementedError()

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
        raise NotImplementedError()

    async def get_account_nonce(self, account_address: str) -> int:
        raise NotImplementedError()

    async def get_constant(
        self, module_name: str, constant_name: str, block_hash: Optional[str] = None
    ) -> Optional["ScaleType"]:
        raise NotImplementedError()

    async def get_payment_info(
        self, call: GenericCall, keypair: Keypair
    ) -> dict[str, Any]:
        raise NotImplementedError()

    async def query(
        self,
        module: str,
        storage_function: str,
        params: Optional[list] = None,
        block_hash: Optional[str] = None,
        subscription_handler: Optional[callable] = None,
        raw_storage_key: Optional[bytes] = None,
    ) -> "ScaleType":
        raise NotImplementedError()

    async def query_map(
        self,
        module: str,
        storage_function: str,
        params: Optional[list] = None,
        block_hash: Optional[str] = None,
        max_results: Optional[int] = None,
        start_key: Optional[str] = None,
        page_size: int = 100,
        ignore_decoding_errors: bool = True,
    ) -> "QueryMapResult":
        raise NotImplementedError()

    async def submit_extrinsic(
        self,
        extrinsic: GenericExtrinsic,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
    ) -> "ExtrinsicReceipt":
        raise NotImplementedError()

    async def get_metadata_call_function(
        self, module_name: str, call_function_name: str, block_hash: str = None
    ):
        raise NotImplementedError()

    async def get_block_number(self, block_hash: str) -> int:
        """Async version of `substrateinterface.base.get_block_number` method."""
        response = await self.rpc_request("chain_getHeader", [block_hash])

        if "error" in response:
            raise SubstrateRequestException(response["error"]["message"])

        elif "result" in response:
            if response["result"]:
                return int(response["result"]["number"], 16)

    def close(self):
        self.substrate.close()
        # TODO: Websocket close need to be implemented and added here.


class RequestManager:
    def __init__(self, payloads):
        self.response_map = {}
        self.responses = defaultdict(lambda: {"complete": False, "results": []})
        self.payloads_count = len(payloads)

    async def add_request(self, item_id: int, request_id: int):
        self.response_map[item_id] = request_id

    async def add_response(self, item_id: int, response: dict, complete: bool):
        request_id = self.response_map[item_id]
        self.responses[request_id]["results"].append(response)
        self.responses[request_id]["complete"] = complete

    async def is_complete(self):
        return (
            all(info["complete"] for info in self.responses.values())
            and len(self.responses) == self.payloads_count
        )

    async def get_results(self):
        return {
            request_id: info["results"] for request_id, info in self.responses.items()
        }


class Websocket:
    def __init__(
        self,
        ws_url: str,
        max_subscriptions=1024,
        max_connections=100,
        shutdown_timer: Optional[int] = 15,
        options: dict = None,
    ):
        """
        Websocket manager object. Allows for the use of a single websocket connection by multiple
        calls.

        :param ws_url: Websocket URL to connect to
        :param max_subscriptions: Maximum number of subscriptions per websocket connection
        :param max_connections: Maximum number of connections total
        :param shutdown_timer: Number of seconds to shut down websocket connection after last use
        :options: WS connection options
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
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5  # seconds

    async def __aenter__(self):
        async with self._lock:
            self._in_use += 1
            if self._exit_task:
                self._exit_task.cancel()
            if not self._initialized:
                await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self._lock:
            self._in_use -= 1
            if self._exit_task is not None:
                self._exit_task.cancel()
                await self._exit_task
            if self._in_use == 0 and self.ws is not None and self.shutdown_timer:
                self.id = 0
                self._open_subscriptions = 0
                self._exit_task = asyncio.create_task(self._exit_with_timer())

    async def _initialize(self):
        self._initialized = True
        try:
            self.ws = await asyncio.wait_for(
                websockets.connect(self.ws_url, **self._options), timeout=None
            )
            self._receiving_task = asyncio.create_task(self._start_receiving())
        except Exception as e:
            await self._reconnect()

    async def _reconnect(self):
        self._attempts = 0
        while self._attempts < self._max_reconnect_attempts:
            try:
                await asyncio.sleep(self._reconnect_delay)
                self.ws = await asyncio.wait_for(
                    websockets.connect(self.ws_url, **self._options), timeout=None
                )
                self._receiving_task = asyncio.create_task(self._start_receiving())
                self._initialized = True
                return
            except Exception as e:
                self._attempts += 1
        raise ConnectionError(
            f"Failed to reconnect after {self._max_reconnect_attempts} attempts. {e}"
        )

    async def _exit_with_timer(self):
        try:
            await asyncio.sleep(self.shutdown_timer)
            async with self._lock:
                if self.ws:
                    self._receiving_task.cancel()
                    await self._receiving_task
                    await self.ws.close()
                    self.ws = None
                    self._initialized = False
                    self._receiving_task = None
                    self.id = 0
        except asyncio.CancelledError:
            pass

    async def _recv(self) -> None:
        try:
            response = json.loads(await self.ws.recv())
            async with self._lock:
                self._open_subscriptions -= 1
                self._received[response["id"]] = response
        except websockets.exceptions.ConnectionClosed:
            await self._reconnect()

    async def _start_receiving(self):
        try:
            while True:
                await self._recv()
        except asyncio.CancelledError:
            pass

    async def send(self, payload: dict) -> int:
        async with self._lock:
            original_id = self.id
            await self.ws.send(json.dumps({**payload, **{"id": original_id}}))
            self.id += 1
            self._open_subscriptions += 1
            return original_id

    async def retrieve(self, item_id: int) -> Optional[dict]:
        while True:
            async with self._lock:
                if item_id in self._received:
                    return self._received.pop(item_id)
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    pass
