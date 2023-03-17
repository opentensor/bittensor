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
from rich.prompt import Confirm
import bittensor.utils.networking as net
from ..errors import *

def prometheus_extrinsic(
    subtensor: 'bittensor.Subtensor',
    wallet: 'bittensor.wallet',
    port: int, 
    netuid: int,
    ip: int = None,
    wait_for_inclusion: bool = False,
    wait_for_finalization = True,
) -> bool:
    r""" Subscribes an bittensor endpoint to the substensor chain.
    Args:
        subtensor (bittensor.subtensor):
            bittensor subtensor object.
        wallet (bittensor.wallet):
            bittensor wallet object.
        ip (str):
            endpoint host port i.e. 192.122.31.4
        port (int):
            endpoint port number i.e. 9221
        netuid (int):
            network uid to serve on.
        wait_for_inclusion (bool):
            if set, waits for the extrinsic to enter a block before returning true, 
            or returns false if the extrinsic fails to enter the block within the timeout.   
        wait_for_finalization (bool):
            if set, waits for the extrinsic to be finalized on the chain before returning true,
            or returns false if the extrinsic fails to be finalized within the timeout.
    Returns:
        success (bool):
            flag is true if extrinsic was finalized or uncluded in the block. 
            If we did not wait for finalization / inclusion, the response is true.
    """
        
    # ---- Get external ip ----
    if ip == None:
        try:
            external_ip = net.get_external_ip()
            bittensor.__console__.print(":white_heavy_check_mark: [green]Found external ip: {}[/green]".format( external_ip ))
            bittensor.logging.success(prefix = 'External IP', sufix = '<blue>{}</blue>'.format( external_ip ))
        except Exception as E:
            raise RuntimeError('Unable to attain your external ip. Check your internet connection. error: {}'.format(E)) from E
    else:
        external_ip = ip

    call_params={
        'version': bittensor.__version_as_int__, 
        'ip': net.ip_to_int(external_ip), 
        'port': port, 
        'ip_type': net.ip_version(external_ip),
    }

    with bittensor.__console__.status(":satellite: Checking Prometheus..."):
        neuron = subtensor.get_neuron_for_pubkey_and_subnet( wallet.hotkey.ss58_address, netuid = netuid )
        neuron_up_to_date = not neuron.is_null and call_params == {
            'version': neuron.prometheus_info.version,
            'ip': net.ip_to_int(neuron.prometheus_info.ip),
            'port': neuron.prometheus_info.port,
            'ip_type': neuron.prometheus_info.ip_type,
        }

    if neuron_up_to_date:
        bittensor.__console__.print(
            f":white_heavy_check_mark: [green]Prometheus already Served[/green]\n"
            f"[green not bold]- Status: [/green not bold] |"
            f"[green not bold] ip: [/green not bold][white not bold]{net.int_to_ip(neuron.prometheus_info.ip)}[/white not bold] |"
            f"[green not bold] ip_type: [/green not bold][white not bold]{neuron.prometheus_info.ip_type}[/white not bold] |"
            f"[green not bold] port: [/green not bold][white not bold]{neuron.prometheus_info.port}[/white not bold] | "
            f"[green not bold] version: [/green not bold][white not bold]{neuron.prometheus_info.version}[/white not bold] |"
        )


        bittensor.__console__.print(":white_heavy_check_mark: [white]Prometheus already served.[/white]".format( external_ip ))
        return True
    
    # Add netuid, not in prometheus_info
    call_params['netuid'] = netuid

    with bittensor.__console__.status(":satellite: Serving prometheus on: [white]{}:{}[/white] ...".format(subtensor.network, netuid)):
        with subtensor.substrate as substrate:
            call = substrate.compose_call(
                call_module='SubtensorModule',
                call_function='serve_prometheus',
                call_params = call_params
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.hotkey)
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
            if wait_for_inclusion or wait_for_finalization:
                response.process_events()
                if response.is_success:
                    bittensor.__console__.print(':white_heavy_check_mark: [green]Served prometheus[/green]\n  [bold white]{}[/bold white]'.format(
                        json.dumps(call_params, indent=4, sort_keys=True)
                    ))
                    return True
                else:
                    bittensor.__console__.print(':cross_mark: [green]Failed to serve prometheus[/green] error: {}'.format(response.error_message))
                    return False
            else:
                return True
        