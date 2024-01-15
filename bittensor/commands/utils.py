# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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

import sys
import os
import torch
import requests
import bittensor

from retry import retry
from typing import List, Optional, Dict, Tuple, Union, Any
from dataclasses import dataclass
from rich.prompt import Confirm, Prompt, PromptBase
from substrateinterface import SubstrateInterface
from scalecodec import ScaleBytes
from scalecodec.base import RuntimeConfiguration
from substrateinterface import SubstrateInterface
from scalecodec.type_registry import load_type_registry_preset

from . import defaults

console = bittensor.__console__


class IntListPrompt(PromptBase):
    """Prompt for a list of integers."""

    def check_choice(self, value: str) -> bool:
        assert self.choices is not None
        # check if value is a valid choice or all the values in a list of ints are valid choices
        return (
            value == "All"
            or value in self.choices
            or all(
                val.strip() in self.choices for val in value.replace(",", " ").split()
            )
        )


def check_netuid_set(
    config: "bittensor.config",
    subtensor: "bittensor.subtensor",
    allow_none: bool = False,
):
    if subtensor.network != "nakamoto":
        all_netuids = [str(netuid) for netuid in subtensor.get_subnets()]
        if len(all_netuids) == 0:
            console.print(":cross_mark:[red]There are no open networks.[/red]")
            sys.exit()

        # Make sure netuid is set.
        if not config.is_set("netuid"):
            if not config.no_prompt:
                netuid = IntListPrompt.ask(
                    "Enter netuid", choices=all_netuids, default=str(all_netuids[0])
                )
            else:
                netuid = str(defaults.netuid) if not allow_none else "None"
        else:
            netuid = config.netuid

        if isinstance(netuid, str) and netuid.lower() in ["none"] and allow_none:
            config.netuid = None
        else:
            if isinstance(netuid, list):
                netuid = netuid[0]
            try:
                config.netuid = int(netuid)
            except:
                raise ValueError('netuid must be an integer or "None" (if applicable)')


def check_for_cuda_reg_config(config: "bittensor.config") -> None:
    """Checks, when CUDA is available, if the user would like to register with their CUDA device."""
    if torch.cuda.is_available():
        if not config.no_prompt:
            if config.pow_register.cuda.get("use_cuda") == None:  # flag not set
                # Ask about cuda registration only if a CUDA device is available.
                cuda = Confirm.ask("Detected CUDA device, use CUDA for registration?\n")
                config.pow_register.cuda.use_cuda = cuda

            # Only ask about which CUDA device if the user has more than one CUDA device.
            if (
                config.pow_register.cuda.use_cuda
                and config.pow_register.cuda.get("dev_id") is None
            ):
                devices: List[str] = [str(x) for x in range(torch.cuda.device_count())]
                device_names: List[str] = [
                    torch.cuda.get_device_name(x)
                    for x in range(torch.cuda.device_count())
                ]
                console.print("Available CUDA devices:")
                choices_str: str = ""
                for i, device in enumerate(devices):
                    choices_str += "  {}: {}\n".format(device, device_names[i])
                console.print(choices_str)
                dev_id = IntListPrompt.ask(
                    "Which GPU(s) would you like to use? Please list one, or comma-separated",
                    choices=devices,
                    default="All",
                )
                if dev_id.lower() == "all":
                    dev_id = list(range(torch.cuda.device_count()))
                else:
                    try:
                        # replace the commas with spaces then split over whitespace.,
                        # then strip the whitespace and convert to ints.
                        dev_id = [
                            int(dev_id.strip())
                            for dev_id in dev_id.replace(",", " ").split()
                        ]
                    except ValueError:
                        console.log(
                            ":cross_mark:[red]Invalid GPU device[/red] [bold white]{}[/bold white]\nAvailable CUDA devices:{}".format(
                                dev_id, choices_str
                            )
                        )
                        sys.exit(1)
                config.pow_register.cuda.dev_id = dev_id
        else:
            # flag was not set, use default value.
            if config.pow_register.cuda.get("use_cuda") is None:
                config.pow_register.cuda.use_cuda = defaults.pow_register.cuda.use_cuda


def get_hotkey_wallets_for_wallet(wallet) -> List["bittensor.wallet"]:
    hotkey_wallets = []
    hotkeys_path = wallet.path + "/" + wallet.name + "/hotkeys"
    try:
        hotkey_files = next(os.walk(os.path.expanduser(hotkeys_path)))[2]
    except StopIteration:
        hotkey_files = []
    for hotkey_file_name in hotkey_files:
        try:
            hotkey_for_name = bittensor.wallet(
                path=wallet.path, name=wallet.name, hotkey=hotkey_file_name
            )
            if (
                hotkey_for_name.hotkey_file.exists_on_device()
                and not hotkey_for_name.hotkey_file.is_encrypted()
            ):
                hotkey_wallets.append(hotkey_for_name)
        except Exception:
            pass
    return hotkey_wallets


def get_coldkey_wallets_for_path(path: str) -> List["bittensor.wallet"]:
    try:
        wallet_names = next(os.walk(os.path.expanduser(path)))[1]
        return [bittensor.wallet(path=path, name=name) for name in wallet_names]
    except StopIteration:
        # No wallet files found.
        wallets = []
    return wallets


def get_all_wallets_for_path(path: str) -> List["bittensor.wallet"]:
    all_wallets = []
    cold_wallets = get_coldkey_wallets_for_path(path)
    for cold_wallet in cold_wallets:
        if (
            cold_wallet.coldkeypub_file.exists_on_device()
            and not cold_wallet.coldkeypub_file.is_encrypted()
        ):
            all_wallets.extend(get_hotkey_wallets_for_wallet(cold_wallet))
    return all_wallets


def filter_netuids_by_registered_hotkeys(
    cli, subtensor, netuids, all_hotkeys
) -> List[int]:
    netuids_with_registered_hotkeys = []
    for wallet in all_hotkeys:
        netuids_list = subtensor.get_netuids_for_hotkey(wallet.hotkey.ss58_address)
        bittensor.logging.debug(
            f"Hotkey {wallet.hotkey.ss58_address} registered in netuids: {netuids_list}"
        )
        netuids_with_registered_hotkeys.extend(netuids_list)

    if cli.config.netuids == None or cli.config.netuids == []:
        netuids = netuids_with_registered_hotkeys

    elif cli.config.netuids != []:
        netuids = [netuid for netuid in netuids if netuid in cli.config.netuids]
        netuids.extend(netuids_with_registered_hotkeys)

    return list(set(netuids))


@dataclass
class DelegatesDetails:
    name: str
    url: str
    description: str
    signature: str

    @classmethod
    def from_json(cls, json: Dict[str, any]) -> "DelegatesDetails":
        return cls(
            name=json["name"],
            url=json["url"],
            description=json["description"],
            signature=json["signature"],
        )


def _get_delegates_details_from_github(
    requests_get, url: str
) -> Dict[str, DelegatesDetails]:
    response = requests_get(url)

    if response.status_code == 200:
        all_delegates: Dict[str, Any] = response.json()
        all_delegates_details = {}
        for delegate_hotkey, delegates_details in all_delegates.items():
            all_delegates_details[delegate_hotkey] = DelegatesDetails.from_json(
                delegates_details
            )
        return all_delegates_details
    else:
        return {}


def get_delegates_details(url: str) -> Optional[Dict[str, DelegatesDetails]]:
    try:
        return _get_delegates_details_from_github(requests.get, url)
    except Exception:
        return None  # Fail silently


def query_runtime_api(
    chain_endpoint: str,
    runtime_api: str,
    method: str,
    params: Optional[List[dict]],
    block: Optional[int] = None,
) -> Optional[bytes]:
    def state_call(
        substrate: SubstrateInterface,
        method: str,
        data: str,
        block: Optional[int] = None,
    ) -> Optional[object]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with substrate as sub:
                block_hash = None if block == None else sub.get_block_hash(block)
                params = [method, data]
                if block_hash:
                    params = params + [block_hash]
                return sub.rpc_request(method="state_call", params=params)

        return make_substrate_call_with_retry()

    def _encode_params(
        substrate: SubstrateInterface,
        call_definition: List[dict],
        params: Union[List[Any], Dict[str, str]],
    ) -> str:
        with substrate as subs:
            param_data = ScaleBytes(b"")

            for i, param in enumerate(call_definition["params"]):
                scale_obj = subs.create_scale_object(param["type"])
                if type(params) is list:
                    param_data += scale_obj.encode(params[i])
                else:
                    if param["name"] not in params:
                        raise ValueError(
                            f"Missing param {param['name']} in params dict."
                        )

                    param_data += scale_obj.encode(params[param["name"]])

        return param_data.to_hex()

    substrate = SubstrateInterface(
        ss58_format=bittensor.__ss58_format__,
        use_remote_preset=True,
        url=chain_endpoint,
        type_registry=bittensor.__type_registry__,
    )

    call_definition = bittensor.__type_registry__["runtime_api"][runtime_api][
        "methods"
    ][method]

    json_result = state_call(
        substrate=substrate,
        method=f"{runtime_api}_{method}",
        data="0x"
        if params is None
        else _encode_params(
            substrate=substrate, call_definition=call_definition, params=params
        ),
        block=block,
    )

    if json_result is None:
        return None

    return_type = call_definition["type"]

    as_scale_bytes = ScaleBytes(json_result["result"])

    rpc_runtime_config = RuntimeConfiguration()
    rpc_runtime_config.update_type_registry(load_type_registry_preset("legacy"))
    rpc_runtime_config.update_type_registry(bittensor.custom_rpc_type_registry)

    obj = rpc_runtime_config.create_scale_object(return_type, as_scale_bytes)
    if obj.data.to_hex() == "0x0400":  # RPC returned None result
        return None

    return obj.decode()


def neurons_lite_rpc(
    chain_endpoint: str, netuid: int, block: Optional[int] = None
) -> List[bittensor.NeuronInfoLite]:
    hex_bytes_result = query_runtime_api(
        chain_endpoint=chain_endpoint,
        runtime_api="NeuronInfoRuntimeApi",
        method="get_neurons_lite",
        params=[netuid],
        block=block,
    )

    if hex_bytes_result == None:
        return []

    if hex_bytes_result.startswith("0x"):
        bytes_result = bytes.fromhex(hex_bytes_result[2:])
    else:
        bytes_result = bytes.fromhex(hex_bytes_result)

    return bittensor.NeuronInfoLite.list_from_vec_u8(bytes_result)
