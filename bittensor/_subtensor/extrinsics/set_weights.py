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
from rich.prompt import Confirm
from typing import Union
import bittensor.utils.weight_utils as weight_utils
from ..errors import *

from loguru import logger
logger = logger.opt(colors=True)

def set_weights_extrinsic(
        subtensor: 'bittensor.Subtensor',
        wallet: 'bittensor.wallet',
        netuid: int,
        uids: Union[torch.LongTensor, list],
        weights: Union[torch.FloatTensor, list],
        version_key: int = 0,
        wait_for_inclusion:bool = False,
        wait_for_finalization:bool = False,
        prompt:bool = False
    ) -> bool:
    r""" Sets the given weights and values on chain for wallet hotkey account.
    Args:
        wallet (bittensor.wallet):
            bittensor wallet object.
        netuid (int):
            netuid of the subent to set weights for.
        uids (Union[torch.LongTensor, list]):
            uint64 uids of destination neurons.
        weights ( Union[torch.FloatTensor, list]):
            weights to set which must floats and correspond to the passed uids.
        version_key (int):
            version key of the validator.
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
    # First convert types.
    if isinstance( uids, list ):
        uids = torch.tensor( uids, dtype = torch.int64 )
    if isinstance( weights, list ):
        weights = torch.tensor( weights, dtype = torch.float32 )

    # Reformat and normalize.
    weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit( uids, weights )

    # Ask before moving on.
    if prompt:
        if not Confirm.ask("Do you want to set weights:\n[bold white]  weights: {}\n  uids: {}[/bold white ]?".format( [float(v/65535) for v in weight_vals], weight_uids) ):
            return False

    with bittensor.__console__.status(":satellite: Setting weights on [white]{}[/white] ...".format(subtensor.network)):
        try:
            with subtensor.substrate as substrate:
                call = substrate.compose_call(
                    call_module='SubtensorModule',
                    call_function='set_weights',
                    call_params = {
                        'dests': weight_uids,
                        'weights': weight_vals,
                        'netuid': netuid,
                        'version_key': version_key,
                    }
                )
                # Period dictates how long the extrinsic will stay as part of waiting pool
                extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.hotkey, era={'period':100})
                response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                    return True

                response.process_events()
                if response.is_success:
                    bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")
                    bittensor.logging.success(  prefix = 'Set weights', sufix = '<green>Finalized: </green>' + str(response.is_success) )
                    return True
                else:
                    bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))
                    bittensor.logging.warning(  prefix = 'Set weights', sufix = '<red>Failed: </red>' + str(response.error_message) )
                    return False

        except Exception as e:
            bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(e))
            bittensor.logging.warning(  prefix = 'Set weights', sufix = '<red>Failed: </red>' + str(e) )
            return False

    if response.is_success:
        bittensor.__console__.print("Set weights:\n[bold white]  weights: {}\n  uids: {}[/bold white ]".format( [float(v/4294967295) for v in weight_vals], weight_uids ))
        message = '<green>Success: </green>' + f'Set {len(uids)} weights, top 5 weights' + str(list(zip(uids.tolist()[:5], [round (w,4) for w in weights.tolist()[:5]] )))
        logger.debug('Set weights:'.ljust(20) +  message)
        return True
    
    return False