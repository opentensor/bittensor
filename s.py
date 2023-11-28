import bittensor as bt
import typing
import os
import rich
from rich.prompt import Confirm


def _get_coldkey_wallets_for_path(path: str) -> typing.List["bt.wallet"]:
    try:
        wallet_names = next(os.walk(os.path.expanduser(path)))[1]
        return [bt.wallet(path=path, name=name) for name in wallet_names]
    except StopIteration:
        # No wallet files found.
        wallets = []
    return wallets


registered_delegate_info = bt.commands.utils.get_delegates_details(url=bt.__delegates_details_url__)
subtensor = bt.subtensor(network = 'finney')
wallets = _get_coldkey_wallets_for_path('~/.bittensor/wallets')
for wallet in wallets:
    if not Confirm.ask(f"Unstake from {wallet.name}?"): continue
    if not wallet.coldkeypub_file.exists_on_device(): continue

    delegates = subtensor.get_delegated( coldkey_ss58 = wallet.coldkeypub.ss58_address)

    my_delegates = {}  # hotkey, amount
    for delegate in delegates:
        for coldkey_addr, staked in delegate[0].nominators:
            if (
                coldkey_addr == wallet.coldkeypub.ss58_address
                and staked.tao > 0
            ):
                my_delegates[delegate[0].hotkey_ss58] = staked

    for delegate_ss58, amount in my_delegates.items():
        if delegate_ss58 in registered_delegate_info:
            delegate_name = registered_delegate_info[ delegate_ss58 ].name
        else: delegate_name = 'unknown'
        if not Confirm.ask(f"Unstake from {wallet.name} to {delegate_name}:{delegate_ss58} amount {amount} ?"): continue
        try:
            call = subtensor.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="remove_stake",
                call_params={
                    "hotkey": delegate_ss58,
                    "amount_unstaked": amount.rao,
                },
            )
            extrinsic = subtensor.substrate.create_signed_extrinsic(
                call=call, keypair=wallet.coldkey
            )
            response = subtensor.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            response.process_events()
            if response.is_success:
                print ('unstaked successfully')
            else:
                print( response.error_message )
        except Exception as e:
            print (e)
            print ('failed to unstake')