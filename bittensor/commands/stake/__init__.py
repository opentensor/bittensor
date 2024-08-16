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

import typing
import bittensor as bt
from bittensor.commands.utils import DelegatesDetails, get_delegates_details
from rich.table import Table
from rich.console import Console
from rich.prompt import Prompt

"""
This module provides a function to select a delegate from a list of delegates in the Bittensor network.

Example usage:
    import bittensor as bt
    from bittensor.commands.staking.helpers import select_delegate

    subtensor = bt.subtensor(network="finney")
    netuid = 1  # The network UID you're interested in
    selected_delegate = select_delegate(subtensor, netuid)
    print(f"Selected delegate: {selected_delegate.hotkey_ss58}")
"""


def select_delegate(subtensor, netuid: int):
    # Get a list of delegates and sort them by total stake in descending order
    delegates: typing.List[bt.DelegateInfoLight] = (
        subtensor.get_delegates_by_netuid_light(netuid)
    )
    delegates.sort(key=lambda x: x.total_stake, reverse=True)

    # Get registered delegates details.
    registered_delegate_info: typing.Optional[DelegatesDetails] = get_delegates_details(
        url=bt.__delegates_details_url__
    )

    # Initialize Rich console for pretty printing
    console = Console()

    # Create a table to display delegate information
    table = Table(
        show_header=True,
        header_style="bold",
        border_style="rgb(7,54,66)",
        style="rgb(0,43,54)",
    )

    # Add columns to the table with specific styles
    table.add_column("Index", style="rgb(253,246,227)", no_wrap=True)
    table.add_column("Delegate Name", no_wrap=True)
    table.add_column("Hotkey SS58", style="rgb(211,54,130)", no_wrap=True)
    table.add_column("Owner SS58", style="rgb(133,153,0)", no_wrap=True)
    table.add_column("Take", style="rgb(181,137,0)", no_wrap=True)
    table.add_column(
        "Total Stake", style="rgb(38,139,210)", no_wrap=True, justify="right"
    )
    table.add_column(
        "Owner Stake", style="rgb(220,50,47)", no_wrap=True, justify="right"
    )
    # table.add_column("Return per 1000", style="rgb(108,113,196)", no_wrap=True, justify="right")
    # table.add_column("Total Daily Return", style="rgb(42,161,152)", no_wrap=True, justify="right")

    # List to store visible delegates
    visible_delegates = []

    # TODO: Add pagination to handle large number of delegates more efficiently
    # Iterate through delegates and display their information
    idx = 0
    done = False
    while not done:
        try:
            if idx < len(delegates):
                delegate = delegates[idx]

                # Add delegate to visible list
                visible_delegates.append(delegate)

                # Add a row to the table with delegate information
                table.add_row(
                    str(idx),
                    registered_delegate_info[delegate.hotkey_ss58].name
                    if delegate.hotkey_ss58 in registered_delegate_info
                    else "",
                    delegate.hotkey_ss58[:5]
                    + "..."
                    + delegate.hotkey_ss58[-5:],  # Show truncated hotkey
                    delegate.owner_ss58[:5]
                    + "..."
                    + delegate.owner_ss58[-5:],  # Show truncated owner address
                    f"{delegate.take:.6f}",
                    f"τ{delegate.total_stake.tao:,.4f}",
                    f"τ{delegate.owner_stake.tao:,.4f}",
                    # f"τ{delegate.return_per_1000.tao:,.4f}",
                    # f"τ{delegate.total_daily_return.tao:,.4f}",
                )

            # Clear console and print updated table
            console.clear()
            console.print(table)

            # Prompt user for input
            user_input = input(
                'Press Enter to scroll, enter a number (1-N) to select, or type "h" for help: '
            )

            # Add a help option to display information about each column
            if user_input.lower() == "h" or user_input.lower() == "help":
                console.print("\nColumn Information:")
                console.print(
                    "[rgb(253,246,227)]Index:[/rgb(253,246,227)] Position in the list of delegates"
                )
                console.print(
                    "[rgb(211,54,130)]Hotkey SS58:[/rgb(211,54,130)] Truncated public key of the delegate's hotkey"
                )
                console.print(
                    "[rgb(133,153,0)]Owner SS58:[/rgb(133,153,0)] Truncated public key of the delegate's owner"
                )
                console.print(
                    "[rgb(181,137,0)]Take:[/rgb(181,137,0)] Percentage of rewards the delegate takes"
                )
                console.print(
                    "[rgb(38,139,210)]Total Stake:[/rgb(38,139,210)] Total amount staked to this delegate"
                )
                console.print(
                    "[rgb(220,50,47)]Owner Stake:[/rgb(220,50,47)] Amount staked by the delegate owner"
                )
                console.print(
                    "[rgb(108,113,196)]Return per 1000:[/rgb(108,113,196)] Estimated return for 1000 Tao staked"
                )
                console.print(
                    "[rgb(42,161,152)]Total Daily Return:[/rgb(42,161,152)] Estimated total daily return for all stake"
                )
                console.print("\nPress Enter to continue...")
                input()
                continue

            # If user presses Enter, continue to next delegate
            if user_input:
                try:
                    # Try to convert user input to integer (delegate index)
                    selected_idx = int(user_input)
                    if 0 <= selected_idx < len(delegates):
                        # Exit loop if valid index is selected
                        done = True
                    else:
                        console.print(
                            f"[red]Invalid index. Please enter a number between 0 and {len(delegates) - 1}.[/red]"
                        )
                        continue
                except ValueError:
                    console.print(
                        "[red]Invalid input. Please enter a valid number.[/red]"
                    )
                except IndexError:
                    console.print(
                        f"[red]Invalid index. Please enter a number between 0 and {len(delegates) - 1}.[/red]"
                    )
                except Exception as e:
                    console.print(f"[red]An error occurred: {str(e)}[/red]")
                    continue  # If input is invalid, continue to next delegate
        except KeyboardInterrupt:
            # Allow user to exit the selection process
            raise KeyboardInterrupt

        if idx < len(delegates):
            idx += 1

    # TODO( const ): uncomment for check
    # Add a confirmation step before returning the selected delegate
    # console.print(f"\nSelected delegate: [rgb(211,54,130)]{visible_delegates[selected_idx].hotkey_ss58}[/rgb(211,54,130)]")
    # console.print(f"Take: [rgb(181,137,0)]{visible_delegates[selected_idx].take:.6f}[/rgb(181,137,0)]")
    # console.print(f"Total Stake: [rgb(38,139,210)]{visible_delegates[selected_idx].total_stake}[/rgb(38,139,210)]")

    # confirmation = Prompt.ask("Do you want to proceed with this delegate? (y/n)")
    # if confirmation.lower() != 'yes' and confirmation.lower() != 'y':
    #     return select_delegate( subtensor, netuid )

    # Return the selected delegate
    return delegates[selected_idx]
