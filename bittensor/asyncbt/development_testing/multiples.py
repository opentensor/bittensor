import asyncio
from pprint import PrettyPrinter
import time

import scalecodec
from scalecodec.base import RuntimeConfiguration
from scalecodec.type_registry import load_type_registry_preset

import bittensor
from bittensor import subtensor_module
from bittensor.asyncbt.utils import rpc_requests
from bittensor.chain_data import custom_rpc_type_registry


pp = PrettyPrinter(indent=4).pprint


async def get_delegated(ss58: str, rpc: rpc_requests.RPCRequest, block_hash=None):
    encoded = subtensor_module.ss58_to_vec_u8(ss58)
    result = await rpc.rpc_request("delegateInfo_getDelegated", [encoded], block_hash)
    return result


async def root_list(block_hash, rpc: rpc_requests.RPCRequest):
    async def process_neuron_result(json_):
        call_definition = bittensor.__type_registry__["runtime_api"][
            "NeuronInfoRuntimeApi"
        ]["methods"]["get_neurons_lite"]
        return_type = call_definition["type"]
        as_scale_bytes = scalecodec.ScaleBytes(json_)  # type: ignore
        rpc_runtime_config = RuntimeConfiguration()
        rpc_runtime_config.update_type_registry(load_type_registry_preset("legacy"))
        rpc_runtime_config.update_type_registry(custom_rpc_type_registry)
        obj = rpc_runtime_config.create_scale_object(return_type, as_scale_bytes)
        hex_bytes_result = obj.decode()
        bytes_result = bytes.fromhex(hex_bytes_result[2:])
        return bittensor.NeuronInfoLite.list_from_vec_u8(bytes_result)  # type: ignore

    json_result = await rpc.rpc_request(
        "state_call", ["NeuronInfoRuntimeApi_get_neurons_lite", "0x0000"], block_hash
    )
    hotkeys = [n.hotkey for n in await process_neuron_result(json_result)]
    result = await rpc.query_subtensor(
        hotkeys, "TotalHotkeyStake", "SubtensorModule", block_hash
    )
    return {x: bittensor.Balance.from_rao(y[0].value) for x, y in result.items()}


async def get_balance(block_hash, address: str, rpc: rpc_requests.RPCRequest):
    result = await rpc.query_subtensor([address], "Account", "System", block_hash)
    return result[address][0].value["data"]["free"]


async def main():
    start = time.time()
    rpc = rpc_requests.RPCRequest(rpc_requests.CHAIN_ENDPOINT)
    async with rpc:
        block_hash = rpc.substrate.get_chain_head()
        results = await asyncio.gather(
            get_delegated(
                "5H11iQ22o3cLLNzE1uwEjHdQgRXpueSPyCBFHAX3VKQiz3v3", rpc, block_hash
            ),
            root_list(block_hash, rpc),
            get_balance(
                block_hash, "5H11iQ22o3cLLNzE1uwEjHdQgRXpueSPyCBFHAX3VKQiz3v3", rpc
            ),
        )
    pp(results)
    end = time.time()
    print("Total time:", end - start)
    new_start = time.time()
    async with rpc:
        block_hash = rpc.substrate.get_chain_head()
        await asyncio.gather(
            get_delegated(
                "5H11iQ22o3cLLNzE1uwEjHdQgRXpueSPyCBFHAX3VKQiz3v3", rpc, block_hash
            ),
            root_list(block_hash, rpc),
            get_balance(
                block_hash, "5H11iQ22o3cLLNzE1uwEjHdQgRXpueSPyCBFHAX3VKQiz3v3", rpc
            ),
        )
    print("Total time:", time.time() - new_start)


if __name__ == "__main__":
    asyncio.run(main())
