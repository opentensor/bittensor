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

import argparse
import re
from typing import Union
from rich.prompt import Confirm
from numpy.typing import NDArray
import numpy as np
from rich.prompt import Prompt
from typing import Tuple
import bittensor
from .. import defaults, GetChildrenCommand  # type: ignore
from ...utils.formatting import float_to_u16

console = bittensor.__console__


class SetChildrenCommand:
    """
    Executes the ``set_children`` command to add children hotkeys on a specified subnet on the Bittensor network.

    This command is used to delegate authority to different hotkeys, securing their position and influence on the subnet.

    Usage:
        Users can specify the amount or 'proportion' to delegate to a child hotkey (either by name or ``SS58`` address),
        the user needs to have sufficient authority to make this call, and the sum of proportions cannot be greater than 1.

    The command prompts for confirmation before executing the set_children operation.

    Example usage::

        btcli stake set_children --children <child_hotkey>,<child_hotkey> --hotkey <parent_hotkey> --netuid 1 --proportion 0.3,0.3

    Note:
        This command is critical for users who wish to delegate children hotkeys among different neurons (hotkeys) on the network.
        It allows for a strategic allocation of authority to enhance network participation and influence.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Set children hotkeys."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            SetChildrenCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        wallet = bittensor.wallet(config=cli.config)

        children = GetChildrenCommand.run(cli)

        # Calculate the sum of all 'proportion' values
        current_proportions = sum(child["proportion"] for child in children)

        # Get values if not set.
        if not cli.config.is_set("netuid"):
            cli.config.netuid = int(Prompt.ask("Enter netuid"))

        if not cli.config.is_set("children"):
            cli.config.children = Prompt.ask(
                "Enter children hotkey (ss58) as comma-separated values"
            )

        if not cli.config.is_set("hotkey"):
            cli.config.hotkey = Prompt.ask("Enter parent hotkey (ss58)")

        if not cli.config.is_set("proportions"):
            cli.config.proportions = Prompt.ask(
                "Enter proportions for children as comma-separated values"
            )

        # Parse from strings
        netuid = cli.config.netuid

        proportions = np.array(
            [float(x) for x in re.split(r"[ ,]+", cli.config.proportions)],
            dtype=np.float32,
        )
        children = np.array(
            [str(x) for x in re.split(r"[ ,]+", cli.config.children)], dtype=str
        )

        total_proposed = np.sum(proportions) + current_proportions
        if total_proposed > 1:
            raise ValueError(
                f":cross_mark:[red] The sum of all proportions cannot be greater than 1. Proposed sum of proportions is {total_proposed}[/red]"
            )

        success, message = SetChildrenCommand.do_set_children_multiple(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
            children=children,
            hotkey=cli.config.hotkey,
            proportions=proportions,
            wait_for_inclusion=cli.config.wait_for_inclusion,
            wait_for_finalization=cli.config.wait_for_finalization,
            prompt=cli.config.prompt,
        )

        # Result
        if success:
            console.print(
                ":white_heavy_check_mark: [green]Set children hotkeys.[/green]"
            )
        else:
            console.print(
                f":cross_mark:[red] Unable to set children hotkeys.[/red] {message}"
            )

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        set_children_parser = parser.add_parser(
            "set_children", help="""Set multiple children hotkeys."""
        )
        set_children_parser.add_argument(
            "--netuid", dest="netuid", type=int, required=False
        )
        set_children_parser.add_argument(
            "--children", dest="children", type=str, required=False
        )
        set_children_parser.add_argument(
            "--hotkey", dest="hotkey", type=str, required=False
        )
        set_children_parser.add_argument(
            "--proportions", dest="proportions", type=str, required=False
        )
        set_children_parser.add_argument(
            "--wait-for-inclusion",
            dest="wait_for_inclusion",
            action="store_true",
            default=False,
        )
        set_children_parser.add_argument(
            "--wait-for-finalization",
            dest="wait_for_finalization",
            action="store_true",
            default=True,
        )
        set_children_parser.add_argument(
            "--prompt",
            dest="prompt",
            action="store_true",
            default=False,
        )
        bittensor.wallet.add_args(set_children_parser)
        bittensor.subtensor.add_args(set_children_parser)

    @staticmethod
    def do_set_children_multiple(
        subtensor: "bittensor.subtensor",
        wallet: "bittensor.wallet",
        hotkey: str,
        children: Union[NDArray[str], list],
        netuid: int,
        proportions: Union[NDArray[np.float32], list],
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> Tuple[bool, str]:
        r"""
        Sets children hotkeys with a proportion assigned from the parent.

        Args:
            subtensor (bittensor.subtensor):
                Subtensor endpoint to use.
            wallet (bittensor.wallet):
                Bittensor wallet object.
            hotkey (str):
                Parent hotkey.
            children (np.ndarray):
                Children hotkeys.
            netuid (int):
                Unique identifier of for the subnet.
            proportions (np.ndarray):
                Proportions assigned to children hotkeys.
            wait_for_inclusion (bool):
                If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
            wait_for_finalization (bool):
                If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If ``true``, the call waits for confirmation from the user before proceeding.
        Returns:
            Tuple[bool, Optional[str]]: A tuple containing a success flag and an optional error message.
        Raises:
            bittensor.errors.ChildHotkeyError:
                If the extrinsic fails to be finalized or included in the block.
            bittensor.errors.NotRegisteredError:
                If the hotkey is not registered in any subnets.

        """
        # Ask before moving on.
        if prompt:
            if not Confirm.ask(
                "Do you want to add children hotkeys:\n[bold white]  children: {}\n  proportions: {}[/bold white ]?".format(
                    children, proportions
                )
            ):
                return False, "Operation Cancelled"

        with bittensor.__console__.status(
            ":satellite: Setting children hotkeys on [white]{}[/white] ...".format(
                subtensor.network
            )
        ):
            try:
                # Convert to list if ndarray
                proportions_val = (
                    proportions.tolist()
                    if isinstance(proportions, np.ndarray)
                    else proportions
                )

                # Convert each proportion value to u16
                proportions_val = [
                    float_to_u16(proportion) for proportion in proportions_val
                ]

                children_with_proportions = list(zip(children, proportions_val))

                call_module = "SubtensorModule"
                call_function = "set_children_multiple"
                call_params = {
                    "hotkey": hotkey,
                    "children_with_proportions": children_with_proportions,
                    "netuid": netuid,
                }

                success, error_message = subtensor.call(
                    call_module=call_module,
                    call_function=call_function,
                    call_params=call_params,
                    wallet=wallet,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )

                bittensor.__console__.print(success, error_message)

                if not wait_for_finalization and not wait_for_inclusion:
                    return True, "Not waiting for finalization or inclusion."

                if success is True:
                    bittensor.__console__.print(
                        ":white_heavy_check_mark: [green]Finalized[/green]"
                    )
                    bittensor.logging.success(
                        prefix="Set children hotkeys",
                        suffix="<green>Finalized: </green>" + str(success),
                    )
                    return True, "Successfully set children hotkeys and Finalized."
                else:
                    bittensor.__console__.print(
                        f":cross_mark: [red]Failed[/red]: {error_message}"
                    )
                    bittensor.logging.warning(
                        prefix="Set children hotkeys",
                        suffix="<red>Failed: </red>" + str(error_message),
                    )
                    return False, error_message

            except Exception as e:
                bittensor.__console__.print(
                    ":cross_mark: [red]Failed[/red]: error:{}".format(e)
                )
                bittensor.logging.warning(
                    prefix="Set children hotkeys", suffix="<red>Failed: </red>" + str(e)
                )
                return False, "Exception Occurred while setting children hotkeys."
