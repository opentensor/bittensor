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
import json
import time
import bittensor
import bittensor.utils.networking as net
from dataclasses import asdict
from rich.prompt import Confirm


def register_subnetwork_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
) -> bool:
    r"""Registers a new subnetwork
    Args:
        wallet (bittensor.wallet):
            bittensor wallet object.
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
            flag is true if extrinsic was finalized or included in the block.
            If we did not wait for finalization / inclusion, the response is true.
    """
    your_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
    burn_cost = bittensor.utils.balance.Balance(subtensor.get_subnet_burn_cost())
    if burn_cost > your_balance:
        bittensor.__console__.print(
            f"Your balance of: [green]{your_balance}[/green] is not enough to pay the subnet lock cost of: [green]{burn_cost}[/green]"
        )
        return False

    if prompt:
        bittensor.__console__.print(f"Your balance is: [green]{your_balance}[/green]")
        if not Confirm.ask(
            f"Do you want to register a subnet for [green]{ burn_cost }[/green]?"
        ):
            return False

    wallet.coldkey  # unlock coldkey

    with bittensor.__console__.status(":satellite: Registering subnet..."):
        with subtensor.substrate as substrate:
            # create extrinsic call
            call = substrate.compose_call(
                call_module="SubtensorModule",
                call_function="register_network",
                call_params={"immunity_period": 0, "reg_allowed": True},
            )
            extrinsic = substrate.create_signed_extrinsic(
                call=call, keypair=wallet.coldkey
            )
            response = substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            # process if registration successful
            response.process_events()
            if not response.is_success:
                bittensor.__console__.print(
                    ":cross_mark: [red]Failed[/red]: error:{}".format(
                        response.error_message
                    )
                )
                time.sleep(0.5)

            # Successful registration, final check for membership
            else:
                bittensor.__console__.print(
                    f":white_heavy_check_mark: [green]Registered subnetwork with netuid: {response.triggered_events[1].value['event']['attributes'][0]}[/green]"
                )
                return True


from ..commands.network import HYPERPARAMS


def set_hyperparameter_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    netuid: int,
    parameter: str,
    value,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
) -> bool:
    r"""Sets a hyperparameter for a specific subnetwork.
    Args:
        wallet (bittensor.wallet):
            bittensor wallet object.
        netuid (int):
            Subnetwork uid.
        parameter (str):
            Hyperparameter name.
        value (any):
            New hyperparameter value.
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
            flag is true if extrinsic was finalized or included in the block.
            If we did not wait for finalization / inclusion, the response is true.
    """
    if subtensor.get_subnet_owner(netuid) != wallet.coldkeypub.ss58_address:
        bittensor.__console__.print(
            ":cross_mark: [red]This wallet doesn't own the specified subnet.[/red]"
        )
        return False

    wallet.coldkey  # unlock coldkey

    extrinsic = HYPERPARAMS.get(parameter)
    if extrinsic == None:
        bittensor.__console__.print(
            ":cross_mark: [red]Invalid hyperparameter specified.[/red]"
        )
        return False

    with bittensor.__console__.status(
        f":satellite: Setting hyperparameter {parameter} to {value} on subnet: {netuid} ..."
    ):
        with subtensor.substrate as substrate:
            extrinsic_params = substrate.get_metadata_call_function(
                "SubtensorModule", extrinsic
            )
            value_argument = extrinsic_params["fields"][
                len(extrinsic_params["fields"]) - 1
            ]

            # create extrinsic call
            call = substrate.compose_call(
                call_module="SubtensorModule",
                call_function=extrinsic,
                call_params={"netuid": netuid, str(value_argument["name"]): value},
            )
            extrinsic = substrate.create_signed_extrinsic(
                call=call, keypair=wallet.coldkey
            )
            response = substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            # process if registration successful
            response.process_events()
            if not response.is_success:
                bittensor.__console__.print(
                    ":cross_mark: [red]Failed[/red]: error:{}".format(
                        response.error_message
                    )
                )
                time.sleep(0.5)

            # Successful registration, final check for membership
            else:
                bittensor.__console__.print(
                    f":white_heavy_check_mark: [green]Hyper parameter {parameter} changed to {value}[/green]"
                )
                return True
