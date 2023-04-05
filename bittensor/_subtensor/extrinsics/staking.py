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

from rich.prompt import Confirm
from time import sleep
from typing import List, Dict, Union, Optional
from bittensor.utils.balance import Balance
from ..errors import *

def add_stake_extrinsic(
        subtensor: 'bittensor.Subtensor', 
        wallet: 'bittensor.wallet',
        hotkey_ss58: Optional[str] = None,
        amount: Union[Balance, float] = None, 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
    r""" Adds the specified amount of stake to passed hotkey uid.
    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        hotkey_ss58 (Optional[str]):
            ss58 address of the hotkey account to stake to
            defaults to the wallet's hotkey.
        amount (Union[Balance, float]):
            Amount to stake as bittensor balance, or float interpreted as Tao.
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
            flag is true if extrinsic was finalized or uncluded in the block. 
            If we did not wait for finalization / inclusion, the response is true.

    Raises:
        NotRegisteredError:
            If the wallet is not registered on the chain.
        NotDelegateError:
            If the hotkey is not a delegate on the chain.
    """
    # Decrypt keys,
    wallet.coldkey

    # Default to wallet's own hotkey if the value is not passed.
    if hotkey_ss58 is None:
        hotkey_ss58 = wallet.hotkey.ss58_address 

    # Flag to indicate if we are using the wallet's own hotkey.
    own_hotkey: bool

    with bittensor.__console__.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(subtensor.network)):
        old_balance = subtensor.get_balance( wallet.coldkeypub.ss58_address )
        # Get hotkey owner
        hotkey_owner = subtensor.get_hotkey_owner( hotkey_ss58 )
        own_hotkey = (wallet.coldkeypub.ss58_address == hotkey_owner)
        if not own_hotkey:
            # This is not the wallet's own hotkey so we are delegating.
            if not subtensor.is_hotkey_delegate( hotkey_ss58 ):
                raise NotDelegateError("Hotkey: {} is not a delegate.".format(hotkey_ss58))
            
            # Get hotkey take
            hotkey_take = subtensor.get_delegate_take( hotkey_ss58 )
        
        # Get current stake
        old_stake = subtensor.get_stake_for_coldkey_and_hotkey( coldkey_ss58=wallet.coldkeypub.ss58_address, hotkey_ss58=hotkey_ss58 )

    # Convert to bittensor.Balance
    if amount == None:
        # Stake it all.
        staking_balance = bittensor.Balance.from_tao( old_balance.tao )
    elif not isinstance(amount, bittensor.Balance ):
        staking_balance = bittensor.Balance.from_tao( amount )
    else:
        staking_balance = amount

    # Remove existential balance to keep key alive.
    if staking_balance > bittensor.Balance.from_rao( 1000 ):
        staking_balance = staking_balance - bittensor.Balance.from_rao( 1000 )
    else:
        staking_balance = staking_balance

    # Check enough to stake.
    if staking_balance > old_balance:
        bittensor.__console__.print(":cross_mark: [red]Not enough stake[/red]:[bold white]\n  balance:{}\n  amount: {}\n  coldkey: {}[/bold white]".format(old_balance, staking_balance, wallet.name))
        return False
            
    # Ask before moving on.
    if prompt:
        if not own_hotkey:
            # We are delegating.
            if not Confirm.ask("Do you want to delegate:[bold white]\n  amount: {}\n  to: {}\n  take: {}\n  owner: {}[/bold white]".format( staking_balance, wallet.hotkey_str, hotkey_take, hotkey_owner) ):
                return False
        else:
            if not Confirm.ask("Do you want to stake:[bold white]\n  amount: {}\n  to: {}[/bold white]".format( staking_balance, wallet.hotkey_str) ):
                return False

    try:
        with bittensor.__console__.status(":satellite: Staking to: [bold white]{}[/bold white] ...".format(subtensor.network)):
            staking_response: bool = __do_add_stake_single(
                subtensor = subtensor,
                wallet = wallet,
                hotkey_ss58 = hotkey_ss58,
                amount = staking_balance,
                wait_for_inclusion = wait_for_inclusion,
                wait_for_finalization = wait_for_finalization,
            )

        if staking_response: # If we successfully staked.
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                return True

            bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")
            with bittensor.__console__.status(":satellite: Checking Balance on: [white]{}[/white] ...".format(subtensor.network)):
                new_balance = subtensor.get_balance( address = wallet.coldkeypub.ss58_address )
                block = subtensor.get_current_block()
                new_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58= wallet.hotkey.ss58_address,
                    block=block
                ) # Get current stake

                bittensor.__console__.print("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_balance, new_balance ))
                bittensor.__console__.print("Stake:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_stake, new_stake ))
                return True
        else:
            bittensor.__console__.print(":cross_mark: [red]Failed[/red]: Error unknown.")
            return False

    except NotRegisteredError as e:
        bittensor.__console__.print(":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(wallet.hotkey_str))
        return False
    except StakeError as e:
        bittensor.__console__.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
        return False


def add_stake_multiple_extrinsic (
        subtensor: 'bittensor.Subtensor', 
        wallet: 'bittensor.wallet',
        hotkey_ss58s: List[str],
        amounts: List[Union[Balance, float]] = None, 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
    r""" Adds stake to each hotkey_ss58 in the list, using each amount, from a common coldkey.
    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object for the coldkey.
        hotkey_ss58s (List[str]):
            List of hotkeys to stake to.
        amounts (List[Union[Balance, float]]):
            List of amounts to stake. If None, stake all to the first hotkey.
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
            flag is true if extrinsic was finalized or included in the block.
            flag is true if any wallet was staked.
            If we did not wait for finalization / inclusion, the response is true.
    """
    if not isinstance(hotkey_ss58s, list) or not all(isinstance(hotkey_ss58, str) for hotkey_ss58 in hotkey_ss58s):
        raise TypeError("hotkey_ss58s must be a list of str")
    
    if len(hotkey_ss58s) == 0:
        return True

    if amounts is not None and len(amounts) != len(hotkey_ss58s):
        raise ValueError("amounts must be a list of the same length as hotkey_ss58s")

    if amounts is not None and not all(isinstance(amount, (Balance, float)) for amount in amounts):
        raise TypeError("amounts must be a [list of bittensor.Balance or float] or None")

    if amounts is None:
        amounts = [None] * len(hotkey_ss58s)
    else:
        # Convert to Balance
        amounts = [bittensor.Balance.from_tao(amount) if isinstance(amount, float) else amount for amount in amounts ]

        if sum(amount.tao for amount in amounts) == 0:
            # Staking 0 tao
            return True

    # Decrypt coldkey.
    wallet.coldkey

    old_stakes = []
    with bittensor.__console__.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(subtensor.network)):
        old_balance = subtensor.get_balance( wallet.coldkeypub.ss58_address )

        # Get the old stakes.
        for hotkey_ss58 in hotkey_ss58s:
            old_stakes.append( subtensor.get_stake_for_coldkey_and_hotkey( coldkey_ss58 = wallet.coldkeypub.ss58_address, hotkey_ss58 = hotkey_ss58 ) )

    # Remove existential balance to keep key alive.
    ## Keys must maintain a balance of at least 1000 rao to stay alive.
    total_staking_rao = sum([amount.rao if amount is not None else 0 for amount in amounts])
    if total_staking_rao == 0:
        # Staking all to the first wallet.
        if old_balance.rao > 1000:
            old_balance -= bittensor.Balance.from_rao(1000)

    elif total_staking_rao < 1000:
        # Staking less than 1000 rao to the wallets.
        pass
    else:
        # Staking more than 1000 rao to the wallets.
        ## Reduce the amount to stake to each wallet to keep the balance above 1000 rao.
        percent_reduction = 1 - (1000 / total_staking_rao)
        amounts = [Balance.from_tao(amount.tao * percent_reduction) for amount in amounts]
    
    successful_stakes = 0
    for idx, (hotkey_ss58, amount, old_stake) in enumerate(zip(hotkey_ss58s, amounts, old_stakes)):
        staking_all = False
        # Convert to bittensor.Balance
        if amount == None:
            # Stake it all.
            staking_balance = bittensor.Balance.from_tao( old_balance.tao )
            staking_all = True
        else:
            # Amounts are cast to balance earlier in the function
            assert isinstance(amount, bittensor.Balance)
            staking_balance = amount

        # Check enough to stake
        if staking_balance > old_balance:
            bittensor.__console__.print(":cross_mark: [red]Not enough balance[/red]: [green]{}[/green] to stake: [blue]{}[/blue] from coldkey: [white]{}[/white]".format(old_balance, staking_balance, wallet.name))
            continue
                        
        # Ask before moving on.
        if prompt:
            if not Confirm.ask("Do you want to stake:\n[bold white]  amount: {}\n  hotkey: {}[/bold white ]?".format( staking_balance, wallet.hotkey_str) ):
                continue

        try:
            staking_response: bool = __do_add_stake_single(
                subtensor = subtensor,
                wallet = wallet,
                hotkey_ss58 = hotkey_ss58,
                amount = staking_balance,
                wait_for_inclusion = wait_for_inclusion,
                wait_for_finalization = wait_for_finalization,
            )
        
            if staking_response: # If we successfully staked.
                # We only wait here if we expect finalization.

                if idx < len(hotkey_ss58s) - 1:
                    # Wait for tx rate limit.
                    tx_rate_limit_blocks = subtensor.tx_rate_limit()
                    if tx_rate_limit_blocks > 0:
                        bittensor.__console__.print(":hourglass: [yellow]Waiting for tx rate limit: [white]{}[/white] blocks[/yellow]".format(tx_rate_limit_blocks))
                        sleep( tx_rate_limit_blocks * 12 ) # 12 seconds per block

                if not wait_for_finalization and not wait_for_inclusion:
                    bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                    old_balance -= staking_balance
                    successful_stakes += 1
                    if staking_all:
                        # If staked all, no need to continue
                        break

                    continue

                bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")

                block = subtensor.get_current_block()
                new_stake = subtensor.get_stake_for_coldkey_and_hotkey( coldkey_ss58 = wallet.coldkeypub.ss58_address, hotkey_ss58 = hotkey_ss58, block = block )
                new_balance = subtensor.get_balance( wallet.coldkeypub.ss58_address, block = block )
                bittensor.__console__.print("Stake ({}): [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( hotkey_ss58, old_stake, new_stake ))
                old_balance = new_balance
                successful_stakes += 1
                if staking_all:
                    # If staked all, no need to continue
                    break

            else:
                bittensor.__console__.print(":cross_mark: [red]Failed[/red]: Error unknown.")
                continue

        except NotRegisteredError as e:
            bittensor.__console__.print(":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(hotkey_ss58))
            continue
        except StakeError as e:
            bittensor.__console__.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
            continue
            
    
    if successful_stakes != 0:
        with bittensor.__console__.status(":satellite: Checking Balance on: ([white]{}[/white] ...".format(subtensor.network)):
            new_balance = subtensor.get_balance( wallet.coldkeypub.ss58_address )
        bittensor.__console__.print("Balance: [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_balance, new_balance ))
        return True

    return False

def __do_add_stake_single(
        subtensor: 'bittensor.Subtensor', 
        wallet: 'bittensor.wallet',
        hotkey_ss58: str,
        amount: 'bittensor.Balance', 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
    r"""
    Executes a stake call to the chain using the wallet and amount specified.
    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        hotkey_ss58 (str):
            Hotkey to stake to.
        amount (bittensor.Balance):
            Amount to stake as bittensor balance object.
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
            flag is true if extrinsic was finalized or uncluded in the block. 
            If we did not wait for finalization / inclusion, the response is true.
    Raises:
        StakeError:
            If the extrinsic fails to be finalized or included in the block.
        NotDelegateError:
            If the hotkey is not a delegate.
        NotRegisteredError:
            If the hotkey is not registered in any subnets.

    """
    # Decrypt keys,
    wallet.coldkey
    
    hotkey_owner = subtensor.get_hotkey_owner( hotkey_ss58 )
    own_hotkey = (wallet.coldkeypub.ss58_address == hotkey_owner)
    if not own_hotkey:
        # We are delegating.
        # Verify that the hotkey is a delegate.
        if not subtensor.is_hotkey_delegate( hotkey_ss58 = hotkey_ss58 ):
            raise NotDelegateError("Hotkey: {} is not a delegate.".format(hotkey_ss58))

    with subtensor.substrate as substrate:
        call = substrate.compose_call(
        call_module='SubtensorModule', 
        call_function='add_stake',
        call_params={
            'hotkey': hotkey_ss58,
            'amount_staked': amount.rao
            }
        )
        extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
        response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
        # We only wait here if we expect finalization.
        if not wait_for_finalization and not wait_for_inclusion:
            return True

        response.process_events()
        if response.is_success:
            return True
        else:
            raise StakeError(response.error_message)