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

import logging
import bittensor
from ..errors import (
    NominationError,
    NotDelegateError,
    NotRegisteredError,
    StakeError,
    TakeError,
)
from rich.prompt import Confirm
from typing import Union, Optional, Tuple, List
from bittensor.utils.balance import Balance
from bittensor.utils.slippage import (
    Operation, show_slippage_warning_if_needed
)
from bittensor.utils.user_io import (
    user_input_confirmation,
    print_summary_header,
    print_summary_footer,
    print_summary_item,
)
from bittensor.btlogging.defines import BITTENSOR_LOGGER_NAME

logger = logging.getLogger(BITTENSOR_LOGGER_NAME)

def decrease_take_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    hotkey_ss58: Optional[str] = None,
    netuid: int = 0,
    take: float = 0.0,
    wait_for_finalization: bool = False,
    wait_for_inclusion: bool = True,
) -> bool:
    r"""Decrease delegate take for the hotkey and subnet.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        hotkey_ss58 (Optional[str]):
            The ``ss58`` address of the hotkey account to stake to defaults to the wallet's hotkey.
        netuid (int):
            The ``netuid`` of the subnet to set take for.
        take (float):
            The ``take`` of the hotkey for the given subnet.
    Returns:
        success (bool): ``True`` if the transaction was successful.
    """
    # Unlock the coldkey.
    wallet.coldkey
    wallet.hotkey

    with bittensor.__console__.status(
        ":satellite: Sending decrease_take_extrinsic call on [white]{}[/white] ...".format(
            subtensor.network
        )
    ):
        try:
            success = subtensor._do_decrease_take(
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
                netuid=netuid,
                take=take,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if success == True:
                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )
                bittensor.logging.success(
                    prefix="Decrease Delegate Take",
                    suffix="<green>Finalized: </green>" + str(success),
                )

            return success

        except (TakeError, Exception) as e:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            bittensor.logging.warning(
                prefix="Set weights", suffix="<red>Failed: </red>" + str(e)
            )

    return False


def increase_take_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    hotkey_ss58: Optional[str] = None,
    netuid: int = 0,
    take: float = 0.0,
    wait_for_finalization: bool = False,
    wait_for_inclusion: bool = True,
) -> bool:
    r"""Increase delegate take for the hotkey and subnet.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        hotkey_ss58 (Optional[str]):
            The ``ss58`` address of the hotkey account to stake to defaults to the wallet's hotkey.
        netuid (int):
            The ``netuid`` of the subnet to set take for.
        take (float):
            The ``take`` of the hotkey for the given subnet.
    Returns:
        success (bool): ``True`` if the transaction was successful.
    """
    # Unlock the coldkey.
    wallet.coldkey
    wallet.hotkey

    with bittensor.__console__.status(
        ":satellite: Sending increase_take_extrinsic call on [white]{}[/white] ...".format(
            subtensor.network
        )
    ):
        try:
            success = subtensor._do_increase_take(
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
                netuid=netuid,
                take=take,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if success == True:
                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )
                bittensor.logging.success(
                    prefix="Increase Delegate Take",
                    suffix="<green>Finalized: </green>" + str(success),
                )

            return success

        except Exception as e:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            bittensor.logging.warning(
                prefix="Set weights", suffix="<red>Failed: </red>" + str(e)
            )
        except TakeError as e:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            bittensor.logging.warning(
                prefix="Set weights", suffix="<red>Failed: </red>" + str(e)
            )

    return False


def set_delegates_takes_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    takes: List[Tuple[int, float]],
    hotkey_ss58: Optional[str] = None,
    wait_for_finalization: bool = False,
    wait_for_inclusion: bool = True,
) -> bool:
    r"""Set multiple delegate takes for the hotkey across different subnets.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        hotkey_ss58 (Optional[str]):
            The ``ss58`` address of the hotkey account to stake to defaults to the wallet's hotkey.
        takes (List[Tuple[int, float]]):
            A list of tuples where each tuple contains a subnet ID (`netuid`) and the new take (`take`) for that subnet.
    Returns:
        success (bool): ``True`` if the transaction was successful.
    """
    # Unlock the coldkey.
    wallet.coldkey
    wallet.hotkey

    with bittensor.__console__.status(
        ":satellite: Sending set_delegates_takes_extrinsic call on [white]{}[/white] ...".format(
            subtensor.network
        )
    ):
        try:
            success = subtensor._set_delegate_takes(
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
                takes=takes,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if success:
                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )
                bittensor.logging.success(
                    prefix="Set Delegate Takes",
                    suffix="<green>Finalized: </green>" + str(success),
                )

                return True

        except Exception as e:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            bittensor.logging.warning(
                prefix="Set Delegate Takes", suffix="<red>Failed: </red>" + str(e)
            )
        except TakeError as e:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            bittensor.logging.warning(
                prefix="Set Delegate Takes", suffix="<red>Failed: </red>" + str(e)
            )

            return False
