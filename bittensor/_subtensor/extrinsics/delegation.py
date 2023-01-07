
            
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
import bittensor
from ..errors import *

from loguru import logger
logger = logger.opt(colors=True)

def become_delegate( 
    subtensor: 'bittensor.Subtensor',
    wallet: 'bittensor.Wallet', 
    wait_for_finalization: bool = False, 
    wait_for_inclusion: bool = True 
) -> bool:
    r""" Becomes a delegate for the hotkey.
    Args:
        wallet ( bittensor.Wallet ):
            The wallet to become a delegate for.
    Returns:
        success (bool):
            True if the transaction was successful.
    """
    # Unlock the coldkey.
    wallet.coldkey
    wallet.hotkey

    # Check if the hotkey is already a delegate.
    if subtensor.is_hotkey_delegate( wallet.hotkey.ss58_address ):
        logger.error('Hotkey {} is already a delegate.'.format(wallet.hotkey.ss58_address))
        return False

    with bittensor.__console__.status(":satellite: Sending become delegate call on [white]{}[/white] ...".format(subtensor.network)):
        try:
            with subtensor.substrate as substrate:
                call = substrate.compose_call(
                    call_module='Paratensor',
                    call_function='become_delegate',
                    call_params = {
                        'hotkey': wallet.hotkey.ss58_address
                    }
                )
                extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey ) # sign with coldkey
                response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                    return True

                response.process_events()
                if response.is_success:
                    bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")
                    bittensor.logging.success(  prefix = 'Become Delegate', sufix = '<green>Finalized: </green>' + str(response.is_success) )
                else:
                    bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))
                    bittensor.logging.warning(  prefix = 'Set weights', sufix = '<red>Failed: </red>' + str(response.error_message) )

        except Exception as e:
            bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(e))
            bittensor.logging.warning(  prefix = 'Set weights', sufix = '<red>Failed: </red>' + str(e) )
            return False

    if response.is_success:
        return True
    
    return False