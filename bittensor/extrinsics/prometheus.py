# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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
import bittensor

import json
import bittensor.utils.networking as net

console = bittensor.__console__


def prometheus_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    port: int,
    netuid: int,
    ip: int = None,
    wait_for_inclusion: bool = False,
    wait_for_finalization=True,
) -> bool:
    r"""Subscribes an Bittensor endpoint to the substensor chain.

    Args:
        subtensor (bittensor.subtensor):
            Bittensor subtensor object.
        wallet (bittensor.wallet):
            Bittensor wallet object.
        ip (str):
            Endpoint host port i.e., ``192.122.31.4``.
        port (int):
            Endpoint port number i.e., `9221`.
        netuid (int):
            Network `uid` to serve on.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or uncluded in the block.
            If we did not wait for finalization / inclusion, the response is ``true``.
    """

    # ---- Get external ip ----
    if ip == None:
        try:
            external_ip = net.get_external_ip()
            console.success(f"Found external ip: {external_ip}")
            bittensor.logging.success(
                prefix="External IP", sufix="<blue>{}</blue>".format(external_ip)
            )
        except Exception as E:
            raise RuntimeError(
                "Unable to attain your external ip. Check your internet connection. error: {}".format(
                    E
                )
            ) from E
    else:
        external_ip = ip

    call_params: "bittensor.PrometheusServeCallParams" = {
        "version": bittensor.__version_as_int__,
        "ip": net.ip_to_int(external_ip),
        "port": port,
        "ip_type": net.ip_version(external_ip),
    }

    with console.status("Checking Prometheus..."):
        neuron = subtensor.get_neuron_for_pubkey_and_subnet(
            wallet.hotkey.ss58_address, netuid=netuid
        )
        neuron_up_to_date = not neuron.is_null and call_params == {
            "version": neuron.prometheus_info.version,
            "ip": net.ip_to_int(neuron.prometheus_info.ip),
            "port": neuron.prometheus_info.port,
            "ip_type": neuron.prometheus_info.ip_type,
        }

    if neuron_up_to_date:
        console.print(
            f"\u2714 <g>Prometheus already Served</g>\n"
            f"<g>- Status: </g> | "
            f"<g>ip: </g><w>{net.int_to_ip(neuron.prometheus_info.ip)}</w> | "
            f"<g>ip_type: </g><w>{neuron.prometheus_info.ip_type}</w> | "
            f"<g>port: </g><w>{neuron.prometheus_info.port}</w> | "
            f"<g>version: </g><w>{neuron.prometheus_info.version}</w>\n"
        )

        console.info(
            "\u2714 <w>Prometheus already served.</w>"
        )
        return True

    # Add netuid, not in prometheus_info
    call_params["netuid"] = netuid

    with console.status(
        "Serving prometheus on: <w>{}:{}</w> ...".format(
            subtensor.network, netuid
        )
    ):
        success, err = subtensor._do_serve_prometheus(
            wallet=wallet,
            call_params=call_params,
            wait_for_finalization=wait_for_finalization,
            wait_for_inclusion=wait_for_inclusion,
        )

        if wait_for_inclusion or wait_for_finalization:
            if success == True:
                console.success(
                    "Served prometheus",
                    "\n  <w><b>{}</b></w>".format(
                        json.dumps(call_params, indent=4, sort_keys=True)
                    )
                )
                return True
            else:
                console.error("Failed to serve prometheus", err)
                return False
        else:
            return True
