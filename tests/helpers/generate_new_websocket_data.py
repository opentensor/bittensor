"""
This is a script whose purpose is to generate new stub data for given commands for use in the integration tests.

The integration tests rely on actual websocket sends/responses, which have been more-or-less manually entered.

The Async Substrate Interface package includes a raw websocket logger [logging.getLogger("raw_websocket")] which will
be used to gather this data. It is imperative that this script only uses the SubstrateInterface class, as sorting the
sent/received data from an asynchronous manner will be significantly more difficult to parse (though it's doable by
checking IDs if we ever would absolutely need to.

Notes:
 - received websocket responses begin with `WEBSOCKET_RECEIVE> `
 - sent websocket begin with `WEBSOCKET_SEND> `
 - both are stringified JSON
 - logging level is DEBUG
 - metadata and metadataV15 (metadata at version) must be discarded, or rather just dumped to their respective txt files
 - metadata/metadataV15 txt files are just the "result" portion of the response:
    e.g. `{"jsonrpc": "2.0", "id": _id, "result": METADATA}`
 - This script should NOT overwrite "retry_archive", as that uses a specific cycling mechanism to cycle between possible
    runtime responses
 - the data is structured as follows:
    seed_name: {
      rpc_method: {
        params: {
          response
        }
      }
    }

 - seed_name is basically just the name of the test that is being run. Specifying the seed name tells the FakeWebsocket
    which set of data to use
 - some workflows may specify the same parameters for a given RPC call, and expect different results. This is alleviated
    typically by specifying block hashes/numbers, though that does not always fit well into the tests.
 - the tests should be short and concise, testing just a single "thing" because of the possibility of conflicting rpc
    requests with the same params
 - all requests include {"json_rpc": "2.0"} in them — this can be removed for the sake of saving space, but is not
    imperative to do so
 - the websocket logger will include the id of the request: these should be stripped away, as they are dynamically
    created in SubstrateInterface, and then attached to the next response by FakeWebsocket
"""

import os

# Not really necessary, but doesn't hurt to have
os.environ["SUBSTRATE_CACHE_METHOD_SIZE"] = "0"
os.environ["SUBSTRATE_RUNTIME_CACHE_SIZE"] = "0"

import logging
import json
import subprocess
import importlib.util
import pathlib
import re

from async_substrate_interface.sync_substrate import (
    raw_websocket_logger,
    SubstrateInterface,
)
from bittensor.core.subtensor import Subtensor

from bittensor.core.settings import LATENT_LITE_ENTRYPOINT

RAW_WS_LOG = "/tmp/bittensor-raw-ws.log"
OUTPUT_DIR = "/tmp/bittensor-ws-output.txt"
OUTPUT_METADATA = "/tmp/integration_websocket_metadata.txt"
OUTPUT_METADATA_V15 = "/tmp/integration_websocket_at_version.txt"


def main(seed: str, method: str, *args, **kwargs):
    """
    Runs the given method on Subtensor, processes the websocket data that occurred during that method's execution,
    attaches it with the "seed" arg as a key to a new tmp file ("/tmp/bittensor-ws-output.txt")

    This data should then be manually added to the `bittensor/tests/helpers/integration_websocket_data.py` file

    The metadata and metadataV15 are dumped to the same `/tmp` dir, but with their respective txt names, as exist in
    `bittensor/tests/helpers/`: `integration_websocket_metadata.txt` and `integration_websocket_at_version.txt`

    While we could automate this, I think it could potentially cause users to accidentally run and then submit this as
    part of a PR.

    """
    if os.path.isfile(RAW_WS_LOG):
        os.remove(RAW_WS_LOG)
    if os.path.isfile(OUTPUT_DIR):
        os.remove(OUTPUT_DIR)

    path = pathlib.Path(__file__).parent / "integration_websocket_data.py"
    spec = importlib.util.spec_from_file_location("bittensor", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    raw_websocket_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(RAW_WS_LOG)
    handler.setLevel(logging.DEBUG)
    raw_websocket_logger.addHandler(handler)

    # we're essentially throwing away Subtensor's underlying Substrate connection, because there's no way to
    # initialize it with `_log_raw_websockets=True` from Subtensor (nor should there be)
    subtensor = Subtensor(LATENT_LITE_ENTRYPOINT)
    subtensor.substrate.close()
    subtensor.substrate = SubstrateInterface(
        LATENT_LITE_ENTRYPOINT,
        chain_name="Bittensor",
        ss58_format=42,
        _log_raw_websockets=True,
    )

    executor = getattr(subtensor, method)
    result = executor(*args, **kwargs)
    print(result)

    subtensor.close()
    raw_websocket_logger.removeHandler(handler)

    with open(RAW_WS_LOG, "r") as f:
        all_ws_data = f.readlines()

    metadata = None
    metadataV15 = None

    output_dict = {seed: {}}
    output_dict_at_seed = output_dict[seed]
    upcoming_metadata = False
    upcoming_metadataV15 = False

    for l in all_ws_data:
        if l.startswith("WEBSOCKET_SEND> "):
            data = json.loads(l[len("WEBSOCKET_SEND> ") :])
            del data["jsonrpc"]
            del data["id"]
            send_method = data["method"]
            if send_method == "state_getMetadata":
                upcoming_metadata = True
                continue
            send_params = json.dumps(data["params"])
            if (
                send_method == "state_call"
                and "Metadata_metadata_at_version" in send_params
            ):
                upcoming_metadataV15 = True
                continue
            if send_method in output_dict_at_seed.keys():
                output_dict_at_seed[send_method][send_params] = {}
            else:
                output_dict_at_seed[send_method] = {send_params: {}}
        elif l.startswith("WEBSOCKET_RECEIVE> "):
            data = json.loads(l[len("WEBSOCKET_RECEIVE> ") :])
            if upcoming_metadata:
                upcoming_metadata = False
                metadata = data["result"]
                continue
            elif upcoming_metadataV15:
                upcoming_metadataV15 = False
                metadataV15 = data["result"]
                continue
            del data["id"]
            del data["jsonrpc"]
            try:
                output_dict_at_seed[send_method][send_params] = data
            except (NameError, KeyError):
                raise KeyError(
                    f"Attempting to add a received value before its keys have been added: {l}"
                )

    with open(OUTPUT_DIR, "w+") as f:
        f.write(str(output_dict))


    mod.WEBSOCKET_RESPONSES[seed] = output_dict_at_seed
    source = path.read_text(encoding="utf-8")
    pattern = re.compile(r"(?ms)^WEBSOCKET_RESPONSES\s*=\s*{.*?}")
    replacement = f"WEBSOCKET_RESPONSES = {repr(mod.WEBSOCKET_RESPONSES)}"
    new_source = pattern.sub(replacement, source)
    path.write_text(new_source, encoding="utf-8")

    if metadata is not None:
        with open(OUTPUT_METADATA, "w+") as f:
            f.write(metadata)
    if metadataV15 is not None:
        with open(OUTPUT_METADATA_V15, "w+") as f:
            f.write(metadataV15)
    # ruff format the output info to make it easier to copy/paste into the file
    subprocess.run(["ruff", "format", path.as_posix()])


if __name__ == "__main__":
    # Example usage
    # main("subnetwork_n", "subnetwork_n", 1)
    # main("get_all_subnets_info", "get_all_subnets_info")
    # main("metagraph", "metagraph", 1)
    # main(
    #     "get_netuids_for_hotkey",
    #     "get_netuids_for_hotkey",
    #     "5Cf4LPRv6tiyuFsfLRQaFYEEn3zJRGi4bAE9DwbbKmbCSHpV",
    # )
    # main("get_current_block", "get_current_block")
    # main(
    #     "is_hotkey_registered_on_subnet",
    #     "is_hotkey_registered_on_subnet",
    #     "5Cf4LPRv6tiyuFsfLRQaFYEEn3zJRGi4bAE9DwbbKmbCSHpV",
    #     14,
    # )
    # main(
    #     "is_hotkey_registered",
    #     "is_hotkey_registered",
    #     "5Cf4LPRv6tiyuFsfLRQaFYEEn3zJRGi4bAE9DwbbKmbCSHpV",
    # )
    # main("blocks_since_last_update", "blocks_since_last_update", 1, 0)
    # main("get_block_hash", "get_block_hash", 6522038)
    # main(
    #     "get_neuron_for_pubkey_and_subnet",
    #     "get_neuron_for_pubkey_and_subnet",
    #     "5Cf4LPRv6tiyuFsfLRQaFYEEn3zJRGi4bAE9DwbbKmbCSHpV",
    #     14,
    # )
    main("metagraph", "metagraph", 1)
