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
        