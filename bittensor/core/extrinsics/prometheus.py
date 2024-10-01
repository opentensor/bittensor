# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import json
from typing import Optional, TYPE_CHECKING

from retry import retry

from bittensor.core.extrinsics.utils import submit_extrinsic
from bittensor.core.settings import version_as_int, bt_console
from bittensor.utils import networking as net, format_error_message
from bittensor.utils.btlogging import logging
from bittensor.utils.networking import ensure_connected

# For annotation purposes
if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor
    from bittensor.core.types import PrometheusServeCallParams


# Chain call for `prometheus_extrinsic`
@ensure_connected
def do_serve_prometheus(
    self: "Subtensor",
    wallet: "Wallet",
    call_params: "PrometheusServeCallParams",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> tuple[bool, Optional[dict]]:
    """
    Sends a serve prometheus extrinsic to the chain.

    Args:
        self (bittensor.core.subtensor.Subtensor): Bittensor subtensor object
        wallet (bittensor_wallet.Wallet): Wallet object.
        call_params (bittensor.core.types.PrometheusServeCallParams): Prometheus serve call parameters.
        wait_for_inclusion (bool): If ``true``, waits for inclusion.
        wait_for_finalization (bool): If ``true``, waits for finalization.

    Returns:
        success (bool): ``True`` if serve prometheus was successful.
        error (Optional[str]): Error message if serve prometheus failed, ``None`` otherwise.
    """

    @retry(delay=1, tries=3, backoff=2, max_delay=4)
    def make_substrate_call_with_retry():
        call = self.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="serve_prometheus",
            call_params=call_params,
        )
        extrinsic = self.substrate.create_signed_extrinsic(
            call=call, keypair=wallet.hotkey
        )
        response = submit_extrinsic(
            substrate=self.substrate,
            extrinsic=extrinsic,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
        if wait_for_inclusion or wait_for_finalization:
            response.process_events()
            if response.is_success:
                return True, None
            else:
                return False, response.error_message
        else:
            return True, None

    return make_substrate_call_with_retry()


def prometheus_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    port: int,
    netuid: int,
    ip: int = None,
    wait_for_inclusion: bool = False,
    wait_for_finalization=True,
) -> bool:
    """Subscribes a Bittensor endpoint to the Subtensor chain.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): Bittensor subtensor object.
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        ip (str): Endpoint host port i.e., ``192.122.31.4``.
        port (int): Endpoint port number i.e., `9221`.
        netuid (int): Network `uid` to serve on.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """

    # Get external ip
    if ip is None:
        try:
            external_ip = net.get_external_ip()
            bt_console.print(
                f":white_heavy_check_mark: [green]Found external ip: {external_ip}[/green]"
            )
            logging.success(prefix="External IP", suffix="<blue>{external_ip}</blue>")
        except Exception as e:
            raise RuntimeError(
                f"Unable to attain your external ip. Check your internet connection. error: {e}"
            ) from e
    else:
        external_ip = ip

    call_params: "PrometheusServeCallParams" = {
        "version": version_as_int,
        "ip": net.ip_to_int(external_ip),
        "port": port,
        "ip_type": net.ip_version(external_ip),
    }

    with bt_console.status(":satellite: Checking Prometheus..."):
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
        bt_console.print(
            f":white_heavy_check_mark: [green]Prometheus already Served[/green]\n"
            f"[green not bold]- Status: [/green not bold] |"
            f"[green not bold] ip: [/green not bold][white not bold]{neuron.prometheus_info.ip}[/white not bold] |"
            f"[green not bold] ip_type: [/green not bold][white not bold]{neuron.prometheus_info.ip_type}[/white not bold] |"
            f"[green not bold] port: [/green not bold][white not bold]{neuron.prometheus_info.port}[/white not bold] | "
            f"[green not bold] version: [/green not bold][white not bold]{neuron.prometheus_info.version}[/white not bold] |"
        )

        bt_console.print(
            f":white_heavy_check_mark: [white]Prometheus already served.[/white]"
        )
        return True

    # Add netuid, not in prometheus_info
    call_params["netuid"] = netuid

    with bt_console.status(
        f":satellite: Serving prometheus on: [white]{subtensor.network}:{netuid}[/white] ..."
    ):
        success, error_message = do_serve_prometheus(
            self=subtensor,
            wallet=wallet,
            call_params=call_params,
            wait_for_finalization=wait_for_finalization,
            wait_for_inclusion=wait_for_inclusion,
        )

        if wait_for_inclusion or wait_for_finalization:
            if success is True:
                json_ = json.dumps(call_params, indent=4, sort_keys=True)
                bt_console.print(
                    f":white_heavy_check_mark: [green]Served prometheus[/green]\n  [bold white]{json_}[/bold white]"
                )
                return True
            else:
                bt_console.print(
                    f":cross_mark: [red]Failed[/red]: {format_error_message(error_message)}"
                )
                return False
        else:
            return True
