
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

import torch
import time
from rich.prompt import Confirm
from typing import List, Dict, Union, Optional
import bittensor.utils.networking as net
from bittensor.utils.registration import POWSolution, create_pow
from ..errors import *

def register_extrinsic (
    subtensor: 'bittensor.Subtensor',
    wallet: 'bittensor.Wallet',
    netuid: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
    max_allowed_attempts: int = 3,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[List[int], int] = 0,
    TPB: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
) -> bool:
    r""" Registers the wallet to chain.
    Args:
        wallet (bittensor.wallet):
            bittensor wallet object.
        netuid (int):
            The netuid of the subnet to register on.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning true, 
            or returns false if the extrinsic fails to enter the block within the timeout.   
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning true,
            or returns false if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If true, the call waits for confirmation from the user before proceeding.
        max_allowed_attempts (int):
            Maximum number of attempts to register the wallet.
        cuda (bool):
            If true, the wallet should be registered using CUDA device(s).
        dev_id (Union[List[int], int]):
            The CUDA device id to use, or a list of device ids.
        TPB (int):
            The number of threads per block (CUDA).
        num_processes (int):
            The number of processes to use to register.
        update_interval (int):
            The number of nonces to solve between updates.
        log_verbose (bool):
            If true, the registration process will log more information.
    Returns:
        success (bool):
            flag is true if extrinsic was finalized or uncluded in the block. 
            If we did not wait for finalization / inclusion, the response is true.
    """
    if not subtensor.subnet_exists( netuid ):
        bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error: [bold white]subnet:{}[/bold white] does not exist.".format(netuid))
        return False

    with bittensor.__console__.status(f":satellite: Checking Account on [bold]subnet:{netuid}[/bold]..."):
        neuron = subtensor.get_neuron_for_pubkey_and_subnet( wallet.hotkey.ss58_address, netuid = netuid )
        if not neuron.is_null:
            bittensor.__console__.print(
            ':white_heavy_check_mark: [green]Already Registered[/green]:\n'\
            'uid: [bold white]{}[/bold white]\n' \
            'netuid: [bold white]{}[/bold white]\n' \
            'hotkey: [bold white]{}[/bold white]\n' \
            'coldkey: [bold white]{}[/bold white]' 
            .format(neuron.uid, neuron.netuid, neuron.hotkey, neuron.coldkey))
            return True

    if prompt:
        if not Confirm.ask("Continue Registration?\n  hotkey:     [bold white]{}[/bold white]\n  coldkey:    [bold white]{}[/bold white]\n  network:    [bold white]{}[/bold white]".format( wallet.hotkey.ss58_address, wallet.coldkeypub.ss58_address, subtensor.network ) ):
            return False

    # Attempt rolling registration.
    attempts = 1
    while True:
        bittensor.__console__.print(":satellite: Registering...({}/{})".format(attempts, max_allowed_attempts))
        # Solve latest POW.
        if cuda:
            if not torch.cuda.is_available():
                if prompt:
                    bittensor.__console__.error('CUDA is not available.')
                return False
            pow_result: Optional[POWSolution] = create_pow( subtensor, wallet, netuid, output_in_place, cuda, dev_id, TPB, num_processes=num_processes, update_interval=update_interval, log_verbose=log_verbose )
        else:
            pow_result: Optional[POWSolution] = create_pow( subtensor, wallet, netuid, output_in_place, num_processes=num_processes, update_interval=update_interval, log_verbose=log_verbose )

        # pow failed
        if not pow_result:
            # might be registered already on this subnet
            if (wallet.is_registered( subtensor = subtensor, netuid = netuid )):
                bittensor.__console__.print(f":white_heavy_check_mark: [green]Already registered on netuid:{netuid}[/green]")
                return True
            
        # pow successful, proceed to submit pow to chain for registration
        else:
            with bittensor.__console__.status(":satellite: Submitting POW..."):
                # check if pow result is still valid
                while not pow_result.is_stale(subtensor=subtensor):
                    with subtensor.substrate as substrate:
                        # create extrinsic call
                        call = substrate.compose_call( 
                            call_module='SubtensorModule',  
                            call_function='register', 
                            call_params={ 
                                'netuid': netuid,
                                'block_number': pow_result.block_number, 
                                'nonce': pow_result.nonce, 
                                'work': [int(byte_) for byte_ in pow_result.seal],
                                'hotkey': wallet.hotkey.ss58_address, 
                                'coldkey': wallet.coldkeypub.ss58_address,
                            } 
                        )
                        extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.hotkey )
                        response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization )
                        
                        # We only wait here if we expect finalization.
                        if not wait_for_finalization and not wait_for_inclusion:
                            bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                            return True
                        
                        # process if registration successful, try again if pow is still valid
                        response.process_events()
                        if not response.is_success:
                            if 'key is already registered' in response.error_message:
                                # Error meant that the key is already registered.
                                bittensor.__console__.print(f":white_heavy_check_mark: [green]Already Registered on [bold]subnet:{netuid}[/bold][/green]")
                                return True

                            bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))
                            time.sleep(0.5)
                        
                        # Successful registration, final check for neuron and pubkey
                        else:
                            bittensor.__console__.print(":satellite: Checking Balance...")
                            is_registered = wallet.is_registered( subtensor = subtensor, netuid = netuid )
                            if is_registered:
                                bittensor.__console__.print(":white_heavy_check_mark: [green]Registered[/green]")
                                return True
                            else:
                                # neuron not found, try again
                                bittensor.__console__.print(":cross_mark: [red]Unknown error. Neuron not found.[/red]")
                                continue
                else:
                    # Exited loop because pow is no longer valid.
                    bittensor.__console__.print( "[red]POW is stale.[/red]" )
                    # Try again.
                    continue
                    
        if attempts < max_allowed_attempts:
            #Failed registration, retry pow
            attempts += 1
            bittensor.__console__.print( ":satellite: Failed registration, retrying pow ...({}/{})".format(attempts, max_allowed_attempts))
        else:
            # Failed to register after max attempts.
            bittensor.__console__.print( "[red]No more attempts.[/red]" )
            return False 
        

def burned_register_extrinsic (
    subtensor: 'bittensor.Subtensor',
    wallet: 'bittensor.Wallet',
    netuid: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False
) -> bool:
    r""" Registers the wallet to chain by recycling TAO.
    Args:
        wallet (bittensor.wallet):
            bittensor wallet object.
        netuid (int):
            The netuid of the subnet to register on.
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
    if not subtensor.subnet_exists( netuid ):
        bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error: [bold white]subnet:{}[/bold white] does not exist.".format(netuid))
        return False

    wallet.coldkey # unlock coldkey
    with bittensor.__console__.status(f":satellite: Checking Account on [bold]subnet:{netuid}[/bold]..."):
        neuron = subtensor.get_neuron_for_pubkey_and_subnet( wallet.hotkey.ss58_address, netuid = netuid )

        old_balance = subtensor.get_balance( wallet.coldkeypub.ss58_address )

        burn_amount = subtensor.burn( netuid = netuid )
        if not neuron.is_null:
            bittensor.__console__.print(
            ':white_heavy_check_mark: [green]Already Registered[/green]:\n'\
            'uid: [bold white]{}[/bold white]\n' \
            'netuid: [bold white]{}[/bold white]\n' \
            'hotkey: [bold white]{}[/bold white]\n' \
            'coldkey: [bold white]{}[/bold white]' 
            .format(neuron.uid, neuron.netuid, neuron.hotkey, neuron.coldkey))
            return True
        
    if prompt:
        # Prompt user for confirmation.
        if not Confirm.ask( f"Recycle {burn_amount} to register on subnet:{netuid}?" ):
            return False

    with bittensor.__console__.status(":satellite: Recycling TAO for Registration..."):
       with subtensor.substrate as substrate:
            # create extrinsic call
            call = substrate.compose_call( 
                call_module='SubtensorModule',  
                call_function='burned_register', 
                call_params={ 
                    'netuid': netuid,
                    'hotkey': wallet.hotkey.ss58_address
                } 
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization )
            
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                return True
            
            # process if registration successful, try again if pow is still valid
            response.process_events()
            if not response.is_success:
                bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))
                time.sleep(0.5)
            
            # Successful registration, final check for neuron and pubkey
            else:
                bittensor.__console__.print(":satellite: Checking Balance...")
                block = subtensor.get_current_block()
                new_balance = subtensor.get_balance( wallet.coldkeypub.ss58_address, block = block )

                bittensor.__console__.print("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_balance, new_balance ))
                is_registered = wallet.is_registered( subtensor = subtensor, netuid = netuid )
                if is_registered:
                    bittensor.__console__.print(":white_heavy_check_mark: [green]Registered[/green]")
                    return True
                else:
                    # neuron not found, try again
                    bittensor.__console__.print(":cross_mark: [red]Unknown error. Neuron not found.[/red]")