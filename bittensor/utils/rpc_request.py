import asyncio
from dataclasses import dataclass
import json

from substrateinterface.base import SubstrateInterface, RuntimeConfigurationObject
from substrateinterface.storage import StorageKey
from scalecodec.base import ScaleBytes, ScaleType
from scalecodec.types import GenericMetadataVersioned
import websockets

CHAIN_ENDPOINT = "wss://test.finney.opentensor.ai:443"


@dataclass
class Preprocessed:
    queryable: str
    method: str
    params: list
    value_scale_type: str
    storage_item: ScaleType


async def preprocess(
    query_for: str,
    substrate_interface: SubstrateInterface,
    block_hash: int,
    storage_function: str,
    module: str,
) -> Preprocessed:
    """
    Creates a Preprocessed data object for passing to ``make_call``
    """
    params = [query_for]

    substrate_interface.init_runtime(block_hash=block_hash)  # TODO

    # Search storage call in metadata
    metadata_pallet = substrate_interface.metadata.get_metadata_pallet(module)

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
        runtime_config=substrate_interface.runtime_config,
        metadata=substrate_interface.metadata,
    )
    method = (
        "state_getStorageAt"
        if substrate_interface.supports_rpc_method("state_getStorageAt")
        else "state_getStorage"
    )
    return Preprocessed(
        query_for,
        method,
        [storage_key.to_hex(), block_hash],
        value_scale_type,
        storage_item,
    )


async def process_response(
    response: dict,
    value_scale_type: str,
    storage_item: ScaleType,
    runtime_config: RuntimeConfigurationObject,
    metadata: GenericMetadataVersioned,
):
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

        obj = runtime_config.create_scale_object(
            type_string=value_scale_type,
            data=ScaleBytes(query_value),
            metadata=metadata,
        )
        obj.decode(check_remaining=True)
        obj.meta_info = {"result_found": response.get("result") is not None}
        return obj


async def make_call(
    payloads: dict[int, dict],
    value_scale_type: str,
    storage_item: ScaleType,
    runtime_config: RuntimeConfigurationObject,
    metadata: GenericMetadataVersioned,
):
    async with websockets.connect(CHAIN_ENDPOINT) as websocket:
        for payload in (x["payload"] for x in payloads.values()):
            await websocket.send(json.dumps(payload))

        responses = {}

        for _ in payloads:
            response = json.loads(await websocket.recv())
            decoded_response = await process_response(
                response, value_scale_type, storage_item, runtime_config, metadata
            )

            request_id = response.get("id")
            responses[payloads[request_id]["id"]] = decoded_response

        return responses


async def query_subtensor(
    subtensor, query_for: list, storage_function, module, block_hash: str = None
) -> dict:
    # By allowing for specifying the block hash, users, if they have multiple query types they want
    # to do, can simply query the block hash first, and then pass multiple query_subtensor calls
    # into an asyncio.gather, with the specified block hash
    block_hash = block_hash or subtensor.substrate.get_chain_head()
    preprocessed: tuple[Preprocessed] = await asyncio.gather(
        *[
            preprocess(x, subtensor.substrate, block_hash, storage_function, module)
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

    responses = await make_call(
        all_info,
        value_scale_type,
        storage_item,
        subtensor.substrate.runtime_config,  # individual because I would like to break this out from SSI
        subtensor.substrate.metadata,  # individual because I would like to break this out from SSI
    )
    return responses


if __name__ == "__main__":
    pass
