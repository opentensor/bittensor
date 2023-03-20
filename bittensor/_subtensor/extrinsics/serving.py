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

def serve_extrinsic (
    subtensor: 'bittensor.Subtensor',
    wallet: 'bittensor.wallet',
    ip: str, 
    port: int, 
    protocol: int, 
    netuid: int,
    placeholder1: int = 0,
    placeholder2: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization = True,
    prompt: bool = False,
) -> bool:
    r""" Subscribes an bittensor endpoint to the substensor chain.
    Args:
        wallet (bittensor.wallet):
            bittensor wallet object.
        ip (str):
            endpoint host port i.e. 192.122.31.4
        port (int):
            endpoint port number i.e. 9221
        protocol (int):
            int representation of the protocol 
        netuid (int):
            network uid to serve on.
        placeholder1 (int):
            placeholder for future use.
        placeholder2 (int):
            placeholder for future use.
        wait_for_inclusion (bool):
            if set, waits for the extrinsic to enter a block before returning true, 
            or returns false if the extrinsic fails to enter the block within the timeout.   
        wait_for_finalization (bool):
            if set, waits for the extrinsic to be finalized on the chain before returning true,
            or returns false if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If true, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            flag is true if extrinsic was finalized or uncluded in the block. 
            If we did not wait for finalization / inclusion, the response is true.
    """
    # Decrypt hotkey
    wallet.hotkey

    params = {
        'version': bittensor.__version_as_int__,
        'ip': net.ip_to_int(ip),
        'port': port,
        'ip_type': net.ip_version(ip),
        'netuid': netuid,
        'coldkey': wallet.coldkeypub.ss58_address,
        'protocol': protocol,
        'placeholder1': placeholder1,
        'placeholder2': placeholder2,
    }

    with bittensor.__console__.status(":satellite: Checking Axon..."):
        neuron = subtensor.get_neuron_for_pubkey_and_subnet( wallet.hotkey.ss58_address, netuid = netuid )
        neuron_up_to_date = not neuron.is_null and params == {
            'version': neuron.axon_info.version,
            'ip': net.ip_to_int(neuron.axon_info.ip),
            'port': neuron.axon_info.port,
            'ip_type': neuron.axon_info.ip_type,
            'netuid': neuron.netuid,
            'coldkey': neuron.coldkey,
            'protocol': neuron.axon_info.protocol,
            'placeholder1': neuron.axon_info.placeholder1,
            'placeholder2': neuron.axon_info.placeholder2,
        }

    output = params.copy()
    output['coldkey'] = wallet.coldkeypub.ss58_address
    output['hotkey'] = wallet.hotkey.ss58_address

    if neuron_up_to_date:
        bittensor.__console__.print(f":white_heavy_check_mark: [green]Axon already Served[/green]\n"
                                    f"[green not bold]- coldkey: [/green not bold][white not bold]{output['coldkey']}[/white not bold] \n"
                                    f"[green not bold]- hotkey: [/green not bold][white not bold]{output['hotkey']}[/white not bold] \n"
                                    f"[green not bold]- Status: [/green not bold] |"
                                    f"[green not bold] ip: [/green not bold][white not bold]{net.int_to_ip(output['ip'])}[/white not bold] |"
                                    f"[green not bold] ip_type: [/green not bold][white not bold]{output['ip_type']}[/white not bold] |"
                                    f"[green not bold] port: [/green not bold][white not bold]{output['port']}[/white not bold] | "
                                    f"[green not bold] netuid: [/green not bold][white not bold]{output['netuid']}[/white not bold] |"
                                    f"[green not bold] protocol: [/green not bold][white not bold]{output['protocol']}[/white not bold] |"
                                    f"[green not bold] version: [/green not bold][white not bold]{output['version']}[/white not bold] |"
        )


        return True

    if prompt:
        output = params.copy()
        output['coldkey'] = wallet.coldkeypub.ss58_address
        output['hotkey'] = wallet.hotkey.ss58_address
        if not Confirm.ask("Do you want to serve axon:\n  [bold white]{}[/bold white]".format(
            json.dumps(output, indent=4, sort_keys=True)
        )):
            return False

    with bittensor.__console__.status(":satellite: Serving axon on: [white]{}:{}[/white] ...".format(subtensor.network, netuid)):
        with subtensor.substrate as substrate:
            call = substrate.compose_call(
                call_module='SubtensorModule',
                call_function='serve_axon',
                call_params=params
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.hotkey)
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
            if wait_for_inclusion or wait_for_finalization:
                response.process_events()
                if response.is_success:
                    bittensor.__console__.print(':white_heavy_check_mark: [green]Served[/green]\n  [bold white]{}[/bold white]'.format(
                        json.dumps(params, indent=4, sort_keys=True)
                    ))
                    return True
                else:
                    bittensor.__console__.print(':cross_mark: [green]Failed to Serve axon[/green] error: {}'.format(response.error_message))
                    return False
            else:
                return True

def serve_axon_extrinsic (
    subtensor: 'bittensor.Subtensor',
    axon: 'bittensor.Axon',
    use_upnpc: bool = False,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
) -> bool:
    r""" Serves the axon to the network.
    Args:
        axon (bittensor.Axon):
            Axon to serve.
        use_upnpc (:type:bool, `optional`): 
            If true, the axon attempts port forward through your router before 
            subscribing.                
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning true, 
            or returns false if the extrinsic fails to enter the block within the timeout.   
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning true,
            or returns false if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If true, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            flag is true if extrinsic was finalized or uncluded in the block. 
            If we did not wait for finalization / inclusion, the response is true.
    """
    axon.wallet.hotkey
    axon.wallet.coldkeypub

    # ---- Setup UPNPC ----
    if use_upnpc:
        if prompt:
            if not Confirm.ask("Attempt port forwarding with upnpc?"):
                return False
        try:
            external_port = net.upnpc_create_port_map( port = axon.port )
            bittensor.__console__.print(":white_heavy_check_mark: [green]Forwarded port: {}[/green]".format( axon.port ))
            bittensor.logging.success(prefix = 'Forwarded port', sufix = '<blue>{}</blue>'.format( axon.port ))
        except net.UPNPCException as upnpc_exception:
            raise RuntimeError('Failed to hole-punch with upnpc with exception {}'.format( upnpc_exception )) from upnpc_exception
    else:
        external_port = axon.external_port

    # ---- Get external ip ----
    if axon.external_ip == None:
        try:
            external_ip = net.get_external_ip()
            bittensor.__console__.print(":white_heavy_check_mark: [green]Found external ip: {}[/green]".format( external_ip ))
            bittensor.logging.success(prefix = 'External IP', sufix = '<blue>{}</blue>'.format( external_ip ))
        except Exception as E:
            raise RuntimeError('Unable to attain your external ip. Check your internet connection. error: {}'.format(E)) from E
    else:
        external_ip = axon.external_ip
    
    # ---- Subscribe to chain ----
    serve_success = subtensor.serve(
            wallet = axon.wallet,
            ip = external_ip,
            port = external_port,
            netuid = axon.netuid,
            protocol = axon.protocol,
            wait_for_inclusion = wait_for_inclusion,
            wait_for_finalization = wait_for_finalization,
            prompt = prompt
    )
    return serve_success
