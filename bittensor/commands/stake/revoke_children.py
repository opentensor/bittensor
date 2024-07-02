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

import argparse
import re
from typing import Tuple, List
from rich.prompt import Confirm, Prompt

import bittensor
from .. import defaults, GetChildrenCommand
from ...utils import is_valid_ss58_address

console = bittensor.__console__


class RevokeChildrenCommand:
    """
    Executes the ``revoke_children`` command to remove children hotkeys on a specified subnet on the Bittensor network.

    This command is used to remove delegated authority to child hotkeys, removing their position and influence on the subnet.

    Usage:
        Users can specify the child hotkeys (either by name or ``SS58`` address),
        the user needs to have sufficient authority to make this call.

    The command prompts for confirmation before executing the revoke_children operation.

    Example usage::

        btcli stake revoke_children --children <child_hotkey>,<child_hotkey> --hotkey <parent_hotkey> --netuid 1

    Note:
        This command is critical for users who wish to remove children hotkeys among different neurons (hotkeys) on the network.
        It allows for a strategic removal of authority to enhance network participation and influence.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Revokes children hotkeys."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            RevokeChildrenCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        wallet = bittensor.wallet(config=cli.config)

        GetChildrenCommand.run(cli)

        # Get values if not set.
        if not cli.config.is_set("netuid"):
            cli.config.netuid = int(Prompt.ask("Enter netuid"))

        if not cli.config.is_set("children"):
            cli.config.children = Prompt.ask(
                "Enter children hotkey (ss58) as comma-separated values"
            )

        if not cli.config.is_set("hotkey"):
            cli.config.hotkey = Prompt.ask("Enter parent hotkey (ss58)")

        # Parse from strings
        netuid = cli.config.netuid

        children = re.split(r"[ ,]+", cli.config.children.strip())

        # Validate children SS58 addresses
        for child in children:
            if not is_valid_ss58_address(child):
                console.print(f":cross_mark:[red] Invalid SS58 address: {child}[/red]")
                return

        success, message = RevokeChildrenCommand.do_revoke_children_multiple(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
            children=children,
            hotkey=cli.config.hotkey,
            wait_for_inclusion=cli.config.wait_for_inclusion,
            wait_for_finalization=cli.config.wait_for_finalization,
            prompt=cli.config.prompt,
        )

        # Result
        if success:
            console.print(
                ":white_heavy_check_mark: [green]Revoked children hotkeys.[/green]"
            )
        else:
            console.print(
                f":cross_mark:[red] Unable to revoke children hotkeys.[/red] {message}"
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
        parser = parser.add_parser(
            "revoke_children", help="""Revoke multiple children hotkeys."""
        )
        parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        parser.add_argument("--children", dest="children", type=str, required=False)
        parser.add_argument("--hotkey", dest="hotkey", type=str, required=False)
        parser.add_argument(
            "--wait-for-inclusion",
            dest="wait_for_inclusion",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--wait-for-finalization",
            dest="wait_for_finalization",
            action="store_true",
            default=True,
        )
        parser.add_argument(
            "--prompt",
            dest="prompt",
            action="store_true",
            default=False,
        )
        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def do_revoke_children_multiple(
        subtensor: "bittensor.subtensor",
        wallet: "bittensor.wallet",
        hotkey: str,
        children: List[str],
        netuid: int,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> Tuple[bool, str]:
        r"""
        Revokes children hotkeys from subnet.

        Args:
            subtensor (bittensor.subtensor):
                Subtensor endpoint to use.
            wallet (bittensor.wallet):
                Bittensor wallet object.
            hotkey (str):
                Parent hotkey.
            children (List[str]):
                Children hotkeys.
            netuid (int):
                Unique identifier of for the subnet.
            wait_for_inclusion (bool):
                If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
            wait_for_finalization (bool):
                If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If ``true``, the call waits for confirmation from the user before proceeding.
        Returns:
            success (bool):
                Flag is ``true`` if extrinsic was finalized or included in the block. If we did not wait for finalization / inclusion, the response is ``true``.
        Raises:
            bittensor.errors.ChildHotkeyError:
                If the extrinsic fails to be finalized or included in the block.
            bittensor.errors.NotRegisteredError:
                If the hotkey is not registered in any subnets.

        """
        # Ask before moving on.
        if prompt:
            if not Confirm.ask(
                "Do you want to revoke the children hotkeys:\n[bold white]  children: {}[/bold white ]?".format(
                    children
                )
            ):
                return False, "Operation Cancelled"

        with bittensor.__console__.status(
            ":satellite: Revoking children hotkeys on [white]{}[/white] ...".format(
                subtensor.network
            )
        ):
            try:
                call_module = "SubtensorModule"
                call_function = "revoke_children_multiple"
                call_params = {
                    "hotkey": hotkey,
                    "children": children,
                    "netuid": netuid,
                }

                success, error_message = subtensor.call(
                    call_module=call_module,
                    call_params=call_params,
                    call_function=call_function,
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
                        prefix="Revoked children hotkeys",
                        suffix="<green>Finalized: </green>" + str(success),
                    )
                    return True, "Successfully revoked children hotkeys and Finalized."
                else:
                    bittensor.__console__.print(
                        f":cross_mark: [red]Failed[/red]: {error_message}"
                    )
                    bittensor.logging.warning(
                        prefix="Revoked children hotkeys",
                        suffix="<red>Failed: </red>" + str(error_message),
                    )
                    return False, error_message

            except Exception as e:
                bittensor.__console__.print(
                    ":cross_mark: [red]Failed[/red]: error:{}".format(e)
                )
                bittensor.logging.warning(
                    prefix="Revoked children hotkeys",
                    suffix="<red>Failed: </red>" + str(e),
                )
                return False, "Exception Occurred while revoking children hotkeys."
