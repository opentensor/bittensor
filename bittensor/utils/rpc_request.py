import asyncio
import json

from substrateinterface.storage import StorageKey
from scalecodec.base import ScaleBytes
import websockets

import bittensor

CHAIN_ENDPOINT = "wss://test.finney.opentensor.ai:443"


async def preprocess(
    ss58_address: str, substrate_interface, block_hash, storage_function
):
    module = "SubtensorModule"
    params = [ss58_address]

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
    return (
        ss58_address,
        method,
        [storage_key.to_hex(), block_hash],
        value_scale_type,
        storage_item,
    )


async def process_response(
    response: dict, value_scale_type, storage_item, runtime_config, metadata
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
    payloads: dict[int, dict], value_scale_type, storage_item, runtime_config, metadata
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
            responses[payloads[request_id]["ss58"]] = decoded_response

        return responses


async def query_subtensor(subtensor, hotkeys: list[str], storage_function) -> dict:
    # TODO make this more general
    block_hash = subtensor.substrate.get_chain_head()
    stuff = await asyncio.gather(
        *[
            preprocess(x, subtensor.substrate, block_hash, storage_function)
            for x in hotkeys
        ]
    )
    all_info = {
        i: {
            "ss58": ss58,
            "payload": {"jsonrpc": "2.0", "method": method, "params": params, "id": i},
        }
        for (i, (ss58, method, params, *_)) in enumerate(stuff)
    }
    value_scale_type = stuff[0][3]
    storage_item = stuff[0][4]
    responses = await make_call(
        all_info,
        value_scale_type,
        storage_item,
        subtensor.substrate.runtime_config,
        subtensor.substrate.metadata,
    )
    return responses
