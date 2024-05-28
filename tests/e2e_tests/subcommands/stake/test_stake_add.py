from bittensor.commands.stake import StakeCommand
from bittensor.commands.network import RegisterSubnetworkCommand
from bittensor.commands.register import RegisterCommand
from ...utils import new_wallet, sudo_call_set_network_limit


# Example test using the local_chain fixture
def test_stake_add(local_chain):
    (wallet, exec_command) = new_wallet("//Alice", "//Bob")
    assert sudo_call_set_network_limit(local_chain, wallet)

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

    assert (
        exact_stake > stake_amount_in_rao - withdraw_loss
        and exact_stake <= stake_amount_in_rao
    )

    # we can test remove after set the stake rate limit larger than 1
    # remove_amount = 1
    # exec_command(StakeCommand, ["stake", "remove", "--amount", str(remove_amount)])
    # assert (
    #     local_chain.query(
    #         "SubtensorModule", "TotalHotkeyStake", [wallet.hotkey.ss58_address]
    #     ).serialize()
    #     == exact_stake - remove_amount * 1_000_000_000
    # )
