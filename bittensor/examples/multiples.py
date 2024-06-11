import asyncio
from pprint import PrettyPrinter
import time

import scalecodec
from scalecodec.base import RuntimeConfiguration
from scalecodec.type_registry import load_type_registry_preset

import bittensor
from bittensor import subtensor_module
from bittensor.utils import async_substrate
from bittensor.chain_data import custom_rpc_type_registry

pp = PrettyPrinter(indent=4).pprint


async def get_delegated(
    ss58: str, rpc: async_substrate.AsyncSubstrateInterface, block_hash=None
):
    encoded = subtensor_module.ss58_to_vec_u8(ss58)
    result = await rpc.rpc_request(
        "delegateInfo_getDelegated", [encoded], reuse_block_hash=True
    )
    return result


async def root_list(block_hash, rpc: async_substrate.AsyncSubstrateInterface):
    async def process_neuron_result(json_):
        call_definition = bittensor.__type_registry__["runtime_api"][
            "NeuronInfoRuntimeApi"
        ]["methods"]["get_neurons_lite"]
        return_type = call_definition["type"]
        as_scale_bytes = scalecodec.ScaleBytes(json_["result"])  # type: ignore
        rpc_runtime_config = RuntimeConfiguration()
        rpc_runtime_config.update_type_registry(load_type_registry_preset("legacy"))
        rpc_runtime_config.update_type_registry(custom_rpc_type_registry)
        obj = rpc_runtime_config.create_scale_object(return_type, as_scale_bytes)
        hex_bytes_result = obj.decode()
        bytes_result = bytes.fromhex(hex_bytes_result[2:])
        return bittensor.NeuronInfoLite.list_from_vec_u8(bytes_result)  # type: ignore

    json_result = await rpc.rpc_request(
        "state_call",
        ["NeuronInfoRuntimeApi_get_neurons_lite", "0x0000"],
        reuse_block_hash=True,
    )
    hotkeys = [n.hotkey for n in await process_neuron_result(json_result)]
    result = await rpc.query_multiple(
        hotkeys, "TotalHotkeyStake", "SubtensorModule", reuse_block_hash=True
    )
    return {x: bittensor.Balance.from_rao(y[0].value) for x, y in result.items()}


async def get_balance(
    block_hash, address: str, rpc: async_substrate.AsyncSubstrateInterface
):
    result = await rpc.query_multiple(
        [address], "Account", "System", reuse_block_hash=True
    )
    return result[address][0].value["data"]["free"]


async def main():
    start = time.time()
    rpc = async_substrate.AsyncSubstrateInterface(async_substrate.CHAIN_ENDPOINT)
    # rpc = async_substrate.AsyncSubstrateInterface("ws://127.0.0.1:9946")
    # rpc = rpc_requests.RPCRequest("wss://entrypoint-finney.opentensor.ai:443")
    async with rpc:
        # block_hash = await rpc.get_chain_head()
        block_hash = None
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
    print("First run time:", end - start)
    new_start = time.time()
    async with rpc:
        # block_hash = await rpc.get_chain_head()
        await asyncio.gather(
            get_delegated(
                "5H11iQ22o3cLLNzE1uwEjHdQgRXpueSPyCBFHAX3VKQiz3v3", rpc, block_hash
            ),
            root_list(block_hash, rpc),
            get_balance(
                block_hash, "5H11iQ22o3cLLNzE1uwEjHdQgRXpueSPyCBFHAX3VKQiz3v3", rpc
            ),
        )
    print("Second run time:", time.time() - new_start)


if __name__ == "__main__":
    asyncio.run(main())
