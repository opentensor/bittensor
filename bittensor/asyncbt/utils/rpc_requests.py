import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Optional

from substrateinterface.base import SubstrateInterface
from substrateinterface.storage import StorageKey
from scalecodec.base import ScaleBytes, ScaleType
from scalecodec.types import GenericMetadataVersioned
import websockets

import bittensor

CHAIN_ENDPOINT = "wss://test.finney.opentensor.ai:443"


@dataclass
class Preprocessed:
    queryable: str
    method: str
    params: list
    value_scale_type: str
    storage_item: ScaleType


class Websocket:
    def __init__(self, ws_url: str):
        # TODO allow setting max concurrent connections and rpc subscriptions per connection, default 100/1024
        self.ws_url = ws_url
        self.ws = None
        self.id = 0
        self._received = {}
        self._in_use = 0
        self._receiving_task = None
        self._attempts = 0
        self._initialized = False
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        async with self._lock:
            self._in_use += 1
            if not self._initialized:
                self._initialized = True
                self.ws = await asyncio.wait_for(
                    websockets.connect(self.ws_url), timeout=None
                )
                self._receiving_task = asyncio.create_task(self.start_receiving())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self._lock:
            self._in_use -= 1
            if self._in_use == 0 and self.ws is not None:
                self._receiving_task.cancel()
                await self._receiving_task
                await self.ws.close()
                self.ws = None
                self._initialized = False
                self._receiving_task = None
                self.id = 0

    async def send(self, payload: dict) -> int:
        async with self._lock:
            original_id = self.id
            await self.ws.send(json.dumps({**payload, **{"id": original_id}}))
            self.id += 1
            return original_id

    async def _recv(self) -> None:
        response = json.loads(await self.ws.recv())
        async with self._lock:
            self._received[response["id"]] = response

    async def start_receiving(self):
        try:
            while True:
                await self._recv()
        except asyncio.CancelledError:
            pass

    async def retrieve(self, item_id: int) -> Optional[dict]:
        while True:
            async with self._lock:
                if item_id in self._received:
                    return self._received.pop(item_id)
            await asyncio.sleep(0.1)


class RPCRequest:
    runtime = None
    substrate = None

    def __init__(self, chain_endpoint: str):
        self.chain_endpoint = chain_endpoint
        self.ws = Websocket(chain_endpoint)

    async def __aenter__(self):
        if not self.substrate:
            self.substrate = SubstrateInterface(
                ss58_format=bittensor.__ss58_format__,
                use_remote_preset=True,
                url=self.chain_endpoint,
                type_registry=bittensor.__type_registry__,
            )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def get_storage_item(self, module: str, storage_function: str):
        if not self.substrate.metadata:
            self.substrate.init_runtime()
        metadata_pallet = self.substrate.metadata.get_metadata_pallet(module)
        storage_item = metadata_pallet.get_storage_function(storage_function)
        return storage_item

    async def _preprocess(
        self,
        query_for: str,
        block_hash: int,
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
        payloads: dict[int, dict],
        value_scale_type: Optional[str] = None,
        storage_item: Optional[ScaleType] = None,
        metadata: Optional[GenericMetadataVersioned] = None,
    ):
        response_map = {}
        responses = defaultdict(lambda: {"complete": False, "results": []})
        async with self.ws as ws:
            for item in payloads.values():
                item_id = await ws.send(item["payload"])
                response_map[item_id] = item["id"]
                # {0: rpc_request}

            while True:
                for item_id in response_map.keys():
                    if response := await ws.retrieve(item_id):
                        decoded_response, complete = await self._process_response(
                            response, value_scale_type, storage_item, metadata
                        )
                        responses[response_map[item_id]]["results"].append(
                            decoded_response
                        )
                        responses[response_map[item_id]]["complete"] = complete
                if all(key["complete"] for key in responses.values()) and len(
                    responses
                ) == len(payloads.values()):
                    break
        return {k: v["results"] for k, v in responses.items()}

    async def rpc_request(
        self, method: str, params: list, block_hash: Optional[str] = None
    ) -> dict:
        payloads = {
            1: {
                "id": "rpc_request",
                "payload": {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params + [block_hash] if block_hash else params,
                },
            }
        }
        result = await self._make_rpc_request(payloads)
        return result["rpc_request"][0]["result"]

    async def get_block_hash(self, block: int):
        pass

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
        block_hash = block_hash or self.substrate.get_chain_head()
        self.substrate.init_runtime(block_hash=block_hash)  # TODO
        preprocessed: tuple[Preprocessed] = await asyncio.gather(
            *[
                self._preprocess(x, block_hash, storage_function, module)
                for x in query_for
            ]
        )
        all_info = {
            i: {
                "id": item.queryable,
                "payload": {
                    "jsonrpc": "2.0",
                    "method": item.method,
                    "params": item.params,
                },
            }
            for (i, item) in enumerate(preprocessed)
        }
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
        pass


if __name__ == "__main__":
    pass
