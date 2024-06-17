from bittensor.commands.stake import StakeCommand
from bittensor.commands.unstake import UnStakeCommand
from bittensor.commands.network import RegisterSubnetworkCommand
from bittensor.commands.register import RegisterCommand
from ...utils import (
    setup_wallet,
    sudo_call_set_network_limit,
    sudo_call_set_target_stakes_per_interval,
)


def test_stake_add(local_chain):
    alice_keypair, exec_command, wallet = setup_wallet("//Alice")
    assert sudo_call_set_network_limit(local_chain, wallet)
    assert sudo_call_set_target_stakes_per_interval(local_chain, wallet)

    assert not (local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize())

    exec_command(RegisterSubnetworkCommand, ["s", "create"])
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1])

    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    assert (
        local_chain.query(
            "SubtensorModule", "LastTxBlock", [wallet.hotkey.ss58_address]
        ).serialize()
        == 0
    )

    assert (
        local_chain.query(
            "SubtensorModule", "LastTxBlockDelegateTake", [wallet.hotkey.ss58_address]
        ).serialize()
        == 0
    )

    exec_command(RegisterCommand, ["s", "register", "--neduid", "1"])

    assert (
        local_chain.query(
            "SubtensorModule", "TotalHotkeyStake", [wallet.hotkey.ss58_address]
        ).serialize()
        == 0
    )

    stake_amount = 2
    exec_command(StakeCommand, ["stake", "add", "--amount", str(stake_amount)])
    exact_stake = local_chain.query(
        "SubtensorModule", "TotalHotkeyStake", [wallet.hotkey.ss58_address]
    ).serialize()
    withdraw_loss = 1_000_000
    stake_amount_in_rao = stake_amount * 1_000_000_000

    assert stake_amount_in_rao - withdraw_loss < exact_stake <= stake_amount_in_rao

    # we can test remove after set the stake rate limit larger than 1
    remove_amount = 1
    exec_command(UnStakeCommand, ["stake", "remove", "--amount", str(remove_amount)])
    assert (
        local_chain.query(
            "SubtensorModule", "TotalHotkeyStake", [wallet.hotkey.ss58_address]
        ).serialize()
        == exact_stake - remove_amount * 1_000_000_000
    )
