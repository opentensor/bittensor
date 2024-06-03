import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import time
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


class RPCRequest:
    runtime = None
    substrate = None

    def __init__(self, chain_endpoint: str):
        self.chain_endpoint = chain_endpoint

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
        now = time.time()
        self.substrate.init_runtime(block_hash=block_hash)  # TODO
        print("Init Runtime", time.time() - now)
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
        value_scale_type: str,
        storage_item: ScaleType,
        metadata: GenericMetadataVersioned,
    ):
        async with websockets.connect(self.chain_endpoint) as websocket:
            for payload in (x["payload"] for x in payloads.values()):
                await websocket.send(json.dumps(payload))

            responses = defaultdict(lambda: {"complete": False, "results": []})

            while True:
                response = json.loads(await websocket.recv())
                decoded_response, complete = await self._process_response(
                    response, value_scale_type, storage_item, metadata
                )

                response_id: int = response.get("id")
                if response_id in payloads:
                    responses[payloads[response_id]["id"]]["results"].append(decoded_response)
                    responses[payloads[response_id]["id"]]["complete"] = complete
                if all(key["complete"] for key in responses.values()):
                    break
            responses = {k: v["results"][0] for k, v in responses.items()}
            return responses

    async def rpc_request(
        self, method, params, block_hash: Optional[str] = None
    ) -> dict:
        pass

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
        preprocessed: tuple[Preprocessed] = await asyncio.gather(
            *[
                self._preprocess(
                    x, block_hash, storage_function, module
                )
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
                    "id": i,
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
