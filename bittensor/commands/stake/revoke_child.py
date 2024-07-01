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

from rich.prompt import Prompt, Confirm
from typing import Tuple

import bittensor
from .. import defaults, GetChildrenCommand

console = bittensor.__console__


class RevokeChildCommand:
    """
    Executes the ``revoke_child`` command to remove a child hotkey on a specified subnet on the Bittensor network.

    This command is used to remove delegated authority to children hotkeys, removing their role as a child hotkey owner on the subnet.

    Usage:
        Users can specify the child (``SS58`` address),
        the user needs to have sufficient authority to make this call.

    The command prompts for confirmation before executing the revoke_child operation.

    Example usage::

        btcli stake revoke_child --child <child_hotkey> --hotkey <parent_hotkey> --netuid 1

    Note:
        This command is critical for users who wish to delegate child hotkeys among different neurons (hotkeys) on the network.
        It allows for a strategic allocation of authority to enhance network participation and influence.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Set child hotkey."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            RevokeChildCommand._run(cli, subtensor)
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

        if not cli.config.is_set("child"):
            cli.config.child = Prompt.ask("Enter child hotkey (ss58)")

        if not cli.config.is_set("hotkey"):
            cli.config.hotkey = Prompt.ask("Enter parent hotkey (ss58)")

        # Parse from strings
        netuid = cli.config.netuid

        success, message = RevokeChildCommand.do_revoke_child_singular(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
            child=cli.config.child,
            hotkey=cli.config.hotkey,
            wait_for_inclusion=cli.config.wait_for_inclusion,
            wait_for_finalization=cli.config.wait_for_finalization,
            prompt=cli.config.prompt,
        )

        # Result
        if success:
            console.print(
                ":white_heavy_check_mark: [green]Revoked child hotkey.[/green]"
            )
        else:
            console.print(
                f":cross_mark:[red] Unable to revoke child hotkey.[/red] {message}"
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
        parser = parser.add_parser("revoke_child", help="""Revoke a child hotkey.""")
        parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        parser.add_argument("--child", dest="child", type=str, required=False)
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
    def do_revoke_child_singular(
        subtensor: "bittensor.subtensor",
        wallet: "bittensor.wallet",
        hotkey: str,
        child: str,
        netuid: int,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> Tuple[bool, str]:
        r"""
        Revokes child hotkey from subnet.

        Args:
            subtensor (bittensor.subtensor):
                Subtensor endpoint to use.
            wallet (bittensor.wallet):
                Bittensor wallet object.
            hotkey (str):
                Parent hotkey.
            child (str):
                Child hotkey.
            netuid (int):
                Unique identifier of for the subnet.
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
                "Do you want to revoke the child hotkey:\n[bold white]  child: {}\n [/bold white ]?".format(
                    child
                )
            ):
                return False, "Operation Cancelled"

        with bittensor.__console__.status(
            ":satellite: Revoking child hotkey on [white]{}[/white] ...".format(
                subtensor.network
            )
        ):
            try:
                call_module = "SubtensorModule"
                call_function = "revoke_child_singular"
                call_params = {
                    "hotkey": hotkey,
                    "child": child,
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
                        prefix="Revoked child hotkey",
                        suffix="<green>Finalized: </green>" + str(success),
                    )
                    return True, "Successfully revoked child hotkey and Finalized."
                else:
                    bittensor.__console__.print(
                        f":cross_mark: [red]Failed[/red]: {error_message}"
                    )
                    bittensor.logging.warning(
                        prefix="Revoked child hotkey",
                        suffix="<red>Failed: </red>" + str(error_message),
                    )
                    return False, error_message

            except Exception as e:
                bittensor.__console__.print(
                    ":cross_mark: [red]Failed[/red]: error:{}".format(e)
                )
                bittensor.logging.warning(
                    prefix="Revoked child hotkey", suffix="<red>Failed: </red>" + str(e)
                )
                return False, "Exception Occurred while revoking child hotkey."
