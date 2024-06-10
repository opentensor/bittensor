from bittensor.commands.delegates import DelegateStakeCommand, DelegateUnstakeCommand
from bittensor.commands.stake import StakeCommand
from bittensor.commands.delegates import SetTakeCommand
from bittensor.commands.network import RegisterSubnetworkCommand
from bittensor.commands.register import RegisterCommand

from ...utils import (
    new_wallet,
    call_add_proposal,
    sudo_call_set_network_limit,
    sudo_call_set_target_stakes_per_interval,
)


# delegate seems hard code the network config
def test_root_delegate_stake(local_chain, capsys):
    (wallet, exec_command) = new_wallet("//Alice", "//Bob")

    stakes = local_chain.query(
        "SubtensorModule",
        "Stake",
        [wallet.hotkey.ss58_address, wallet.coldkey.ss58_address],
    )
    assert stakes == 0

    assert sudo_call_set_network_limit(local_chain, wallet)
    assert sudo_call_set_target_stakes_per_interval(local_chain, wallet)

    exec_command(RegisterSubnetworkCommand, ["s", "create"])
    exec_command(RegisterCommand, ["s", "register", "--neduid", "1"])

    stake_amount = 2
    exec_command(StakeCommand, ["stake", "add", "--amount", str(stake_amount)])

    stakes = local_chain.query(
        "SubtensorModule",
        "Stake",
        [wallet.hotkey.ss58_address, wallet.coldkey.ss58_address],
    )

    assert stakes > 1_000_000_000

    delegates = local_chain.query(
        "SubtensorModule",
        "Delegates",
        [wallet.hotkey.ss58_address],
    )

    assert delegates == 11796

    # stake 1 TAO
    # exec_command(
    #     DelegateStakeCommand,
    #     [
    #         "root",
    #         "delegate",
    #         "--delegate_ss58key",
    #         wallet.hotkey.ss58_address,
    #         "--amount",
    #         "1",
    #     ],
    # )

    # new_stakes = local_chain.query(
    #     "SubtensorModule",
    #     "Stake",
    #     [wallet.hotkey.ss58_address, wallet.coldkey.ss58_address],
    # )

    # tolerance = 10000

    # assert (
    #     stakes.serialize() + 1_000_000_000 - tolerance
    #     < new_stakes.serialize()
    #     < stakes.serialize() + 1_000_000_000 + tolerance
    # )

    # unstake 1 TAO
    # exec_command(
    #     DelegateUnstakeCommand,
    #     [
    #         "root",
    #         "delegate",
    #         "--delegate_ss58key",
    #         wallet.hotkey.ss58_address,
    #         "--amount",
    #         "1",
    #     ],
    # )

    # stakes = local_chain.query(
    #     "SubtensorModule",
    #     "Stake",
    #     [wallet.hotkey.ss58_address, wallet.coldkey.ss58_address],
    # )

    # assert (
    #     stakes.serialize() + 1_000_000_000 - tolerance
    #     < new_stakes.serialize()
    #     < stakes.serialize() + 1_000_000_000 + tolerance
    # )
