

import sys
import argparse
import bittensor as bt
from . import select_delegate
from rich.prompt import Confirm, Prompt
from bittensor.utils.slippage import (Operation, show_slippage_warning_if_needed)

class RemoveStakeCommand:
    
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        unstake_parser = parser.add_parser("remove", help="""Remove stake from a specific hotkey on subnet `netuid` from your coldkey.""")
        unstake_parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        unstake_parser.add_argument("--all", dest="un_stake_all", action="store_true")
        unstake_parser.add_argument("--amount", dest="amount", type=float, required=False)
        unstake_parser.add_argument("--hotkey_ss58", required=False, type=str, help="""Specify the hotkey by name or ss58 address.""")
        unstake_parser.add_argument("--delegate", required=False, action='store_true', help="""Specify this flag to delegate stake""" )
        unstake_parser.add_argument("--no_prompt", "--y", "-y", dest = 'no_prompt', required=False, action='store_true', help="""Specify this flag to delegate stake""" )
        bt.wallet.add_args(unstake_parser)
        bt.subtensor.add_args(unstake_parser)
        
    @staticmethod
    def check_config(config: "bt.config"): pass
    
    @staticmethod
    def run(cli: "bt.cli"):
        
        # Get config and subtensor connection.
        config = cli.config.copy()
        subtensor = bt.subtensor(config=config, log_verbose=False)
        wallet = bt.wallet( config = config )

        # Get netuid
        netuid = config.get('netuid') 
        if not config.is_set("netuid") and not config.no_prompt:
            netuid = int( Prompt.ask("Enter netuid", default="0") )
        else:
            bt.logging.error("netuid is needed to proceed")
            sys.exit(1)

        # Get which hotkey we are staking to.
        staking_address_ss58 = config.get('hotkey_ss58')
        staking_address_name = staking_address_ss58

        # If no hotkey is specified, and no prompt is set, delegate to the selected delegate.
        if not staking_address_ss58 and not config.no_prompt:
            if Confirm.ask("Is this a delegation?"):
                delegate = select_delegate(subtensor, netuid)
                staking_address_ss58 = delegate.hotkey_ss58
                staking_address_name = delegate.hotkey_ss58
            else:
                hotkey_str = Prompt.ask("Enter hotkey name or ss58_address to stake to", default=bt.defaults.wallet.hotkey)
                if bt.utils.is_valid_ss58_address(hotkey_str):
                    staking_address_ss58 = str(hotkey_str)
                    staking_address_name = hotkey_str
                else:
                    wallet = bt.wallet(name=config.wallet.name, hotkey=hotkey_str)
                    staking_address_ss58 = wallet.hotkey.ss58_address
                    staking_address_name = hotkey_str
        elif not staking_address_ss58:
            bt.logging.error("--hotkey_ss58 must be specified when using --no_prompt or --y")
            sys.exit(1)

        # Check to see if the passed hotkey is valid.
        if not subtensor.is_hotkey_registered_any(hotkey_ss58=staking_address_ss58):
            bt.__console__.print(f"[red]Hotkey [bold]{staking_address_ss58}[/bold] is not registered on any subnets. Aborting.[/red]")
            sys.exit(1)

        # Check to see if the hotkey is a delegate.
        if wallet.coldkeypub.ss58_address != subtensor.get_hotkey_owner(staking_address_ss58):
            if not subtensor.is_hotkey_delegate(staking_address_ss58):
                bt.__console__.print(f"[red]Hotkey [bold]{staking_address_ss58}[/bold] is not a delegate. Aborting.[/red]")
                sys.exit(1)

        # Get old staking balance.
        current_stake_balance: bt.Balance = subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
            coldkey_ss58=wallet.coldkeypub.ss58_address,
            hotkey_ss58=staking_address_ss58,
            netuid=netuid,
        )
        # Get current wallet balance
        current_wallet_balance: bt.Balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)

        # Determine the amount we are staking.
        amount_to_unstake_as_balance = None
        if config.get("amount"):
            amount_to_unstake_as_balance = bt.Balance.from_tao(config.amount)
        elif config.get("un_stake_all"):
            amount_to_unstake_as_balance = current_stake_balance
        elif not config.get("amount") and not config.get("max_stake"):
            if Confirm.ask(f"Un stake all: [bold]{current_stake_balance}[/bold]?"):
                amount_to_unstake_as_balance = current_stake_balance
            else:
                try:
                    # TODO add unit.
                    amount = float(Prompt.ask("Enter amount to unstake in unit TODO"))
                    amount_to_unstake_as_balance = bt.Balance.from_tao(amount)
                except ValueError:
                    bt.__console__.print(f":cross_mark:[red]Invalid amount: {amount}[/red]")
                    sys.exit(1)

        # Check enough to stake.
        if amount_to_unstake_as_balance > current_stake_balance:
            bt.__console__.print(f"[red]Not enough stake to remove[/red]:[bold white]\n stake balance:{current_stake_balance} < unstaking amount: {amount_to_unstake_as_balance}[/bold white]")
            sys.exit(1)

        # Slippage warning
        if not config.no_prompt:
            dynamic_info = subtensor.get_dynamic_info_for_netuid( netuid )
            received_amount, slippage = dynamic_info.alpha_to_tao_with_slippage( amount_to_unstake_as_balance )
            slippage_pct = 100 * float(slippage) / float(slippage + received_amount) if slippage + received_amount != 0 else 0
            message = (
                f"Would you like to continue?\n\n"
                f"     [rgb(133,153,0)]netuid:[/rgb(133,153,0)] {netuid}\n"
                f"     [rgb(220,50,47)]stake removed:[/rgb(220,50,47)] { amount_to_unstake_as_balance }\n"
                f"     [rgb(38,139,210)]tao recieved:[/rgb(38,139,210)] { received_amount}\n"
                f"     [rgb(181,137,0)]slippage:[/rgb(181,137,0)] {slippage_pct}%\n"
                f"     [rgb(181,137,0)]account:[/rgb(181,137,0)] {staking_address_name}%\n"
            )
            if slippage_pct > 0.05:
                bt.__console__.warning(f"[rgb(181,137,0)]Slippage is high: {slippage_pct}%, this may result in a loss of funds.[/rgb(181,137,0)]")
            if not Confirm.ask(message):
                sys.exit(1)
                                            
        # Perform staking operation.
        with bt.__console__.status(f"\n:satellite:"):
            call = subtensor.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="remove_stake",
                call_params={
                    "hotkey": staking_address_ss58,
                    "netuid": netuid,
                    "amount_unstaked": amount_to_unstake_as_balance.rao,
                },
            )
            extrinsic = subtensor.substrate.create_signed_extrinsic(call=call, keypair=wallet.coldkey)
            response = subtensor.substrate.submit_extrinsic(extrinsic, wait_for_finalization=not config.no_prompt)
            if config.no_prompt:
                bt.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                return
            else:
                response.process_events()
                if not response.is_success:
                    bt.__console__.print(f":cross_mark: [red]Failed[/red] with error: {response.error_message}")
                    return
                else:
                    new_balance = subtensor.get_balance(address=wallet.coldkeypub.ss58_address)
                    new_stake = subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                        coldkey_ss58=wallet.coldkeypub.ss58_address,
                        hotkey_ss58=staking_address_ss58,
                        netuid=netuid,
                    )
                    bt.__console__.print(f"Balance:\n  [blue]{current_wallet_balance}[/blue] :arrow_right: [green]{new_balance}[/green]")
                    bt.__console__.print(f"Stake:\n  [blue]{current_stake_balance}[/blue] :arrow_right: [green]{new_stake}[/green]")
                    return
