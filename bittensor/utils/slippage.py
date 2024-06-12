# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

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

"""Module providing helper functions for displaying slippage warnings."""

import bittensor
from bittensor.utils.user_io import (
    user_input_confirmation,
    print_summary_header,
    print_summary_footer,
    print_summary_message,
)
from enum import Enum

# Maximum slippage percentage
# MAX_SLIPPAGE_PCT = 5.0
MAX_SLIPPAGE_PCT = 0.01

class Operation(Enum):
    UNSTAKE = 1
    STAKE = 2

def show_slippage_warning_if_needed(
    subtensor: "bittensor.subtensor",
    netuid: int,
    op: Operation,
    amount: bittensor.Balance,
    prompt: bool = False,
):
    r"""
    Query subtensor for dynamic pool info and display slippage warning if percentage of 
    slippage is above threshold defined by MAX_SLIPPAGE_PCT

    Args:
        subtensor (bittensor.subtensor):
            Subtensor interface
        netuid (int):
            The subnetwork uid of to stake with.
        op (Operation):
            The staking or unstaking operation
        amount (Balance):
            Amount to stake as Bittensor balance
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding

    Returns:
        success (bool):
            Flag is ``true`` if no slippage warning is needed or the user chose to proceed with high slippage.
    """
    # Get dynamic pool info for slippage calculation
    dynamic_info = subtensor.get_dynamic_info_for_netuid(netuid)

    # Calculate slippage
    subnet_stake_amount = bittensor.Balance.from_tao(amount.tao)
    if op == Operation.STAKE:
        received_amount, slippage = dynamic_info.tao_to_alpha_with_slippage(
            subnet_stake_amount
        )
    else:
        received_amount, slippage = dynamic_info.alpha_to_tao_with_slippage(
            subnet_stake_amount
        )
    slippage_pct = 0
    if slippage + received_amount != 0:
        slippage_pct = 100 * float(slippage) / float(slippage + received_amount)

    # Check if slippage exceeds the maximum threshold
    if slippage_pct > MAX_SLIPPAGE_PCT:
        print_summary_header(f":warning: [yellow]Slippage Warning:[/yellow]")
        print_summary_message(
            f"Slippage exceeds [green][bold]{MAX_SLIPPAGE_PCT}%[/bold][/green] for subnet {str(netuid)}:"
            f" [green][bold]{bittensor.Balance.from_tao(slippage.tao).set_unit(netuid)} ({slippage_pct:.2f}%)[/bold][/green]"
        )
        if op == Operation.STAKE:
            estimated = (
                bittensor.Balance.from_tao(received_amount.tao).set_unit(netuid).__str__()
            )
            expected = (
                bittensor.Balance.from_tao(slippage.tao + received_amount.tao)
                .set_unit(netuid)
                .__str__()
            )
        else:
            estimated = bittensor.Balance.from_tao(received_amount.tao).__str__()
            expected = bittensor.Balance.from_tao(slippage.tao + received_amount.tao).__str__()
        print_summary_message(
            f"You will only receive [green][bold]{estimated}[/bold][/green] vs. expected [green][bold]{expected}[/bold][/green]"
        )
        print_summary_message(
            f"[bold]Note:[/bold] This is only an approximation, the actual result may be slighly different"
        )
        print_summary_footer()
        if prompt:
            if not user_input_confirmation("proceed despite the high slippage"):
                return False
    return True
