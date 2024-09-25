from bittensor import logging
from bittensor.commands.network import RegisterSubnetworkCommand
from bittensor.commands.register import RegisterCommand
from bittensor.commands.stake import StakeCommand
from bittensor.commands.unstake import UnStakeCommand

from ...utils import (
    setup_wallet,
    sudo_call_set_network_limit,
    sudo_call_set_target_stakes_per_interval,
)


def test_stake_add(local_chain):
    logging.info("Testing test_stake_add")
    alice_keypair, exec_command, wallet = setup_wallet("//Alice")
    assert sudo_call_set_network_limit(
        local_chain, wallet
    ), "Unable to set network limit"
    assert sudo_call_set_target_stakes_per_interval(
        local_chain, wallet
    ), "Unable to set target stakes per interval"

    assert not (
        local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()
    ), "Subnet was found in netuid 1"

    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [1]
    ).serialize(), "Subnet 1 was successfully added"

    assert (
        local_chain.query(
            "SubtensorModule", "LastTxBlock", [wallet.hotkey.ss58_address]
        ).serialize()
        == 0
    ), "LastTxBlock is not 0"

    assert (
        local_chain.query(
            "SubtensorModule", "LastTxBlockDelegateTake", [wallet.hotkey.ss58_address]
        ).serialize()
        == 0
    ), "LastTxBlockDelegateTake is not 0"

    exec_command(RegisterCommand, ["s", "register", "--netuid", "1"])

    assert (
        local_chain.query(
            "SubtensorModule", "TotalHotkeyStake", [wallet.hotkey.ss58_address]
        ).serialize()
        == 0
    ), "TotalHotkeyStake is not 0"

    # Check coldkey balance before adding stake
    acc_before = local_chain.query("System", "Account", [wallet.coldkey.ss58_address])
    print("================= Balance before: ", acc_before.value["data"]["free"])

    stake_amount = 2
    exec_command(StakeCommand, ["stake", "add", "--amount", str(stake_amount)])
    exact_stake = local_chain.query(
        "SubtensorModule", "TotalHotkeyStake", [wallet.hotkey.ss58_address]
    ).serialize()
    withdraw_loss = 1_000_000
    stake_amount_in_rao = stake_amount * 1_000_000_000

    assert (
        stake_amount_in_rao - withdraw_loss < exact_stake <= stake_amount_in_rao
    ), f"Stake amount mismatch: expected {exact_stake} to be between {stake_amount_in_rao - withdraw_loss} and {stake_amount_in_rao}"

    # Ensure fees are not withdrawn for the add_stake extrinsic, i.e. balance is exactly lower by stake amount
    acc_after = local_chain.query("System", "Account", [wallet.coldkey.ss58_address])
    assert (
        acc_before.value["data"]["free"] - acc_after.value["data"]["free"]
        == stake_amount * 1_000_000_000
    ), f"Expected no transaction fees for add_stake"

    # we can test remove after set the stake rate limit larger than 1
    remove_amount = 1

    exec_command(UnStakeCommand, ["stake", "remove", "--amount", str(remove_amount)])
    total_hotkey_stake = local_chain.query(
        "SubtensorModule", "TotalHotkeyStake", [wallet.hotkey.ss58_address]
    ).serialize()
    expected_stake = exact_stake - remove_amount * 1_000_000_000
    assert (
        total_hotkey_stake == expected_stake
    ), f"Unstake amount mismatch: expected {expected_stake}, but got {total_hotkey_stake}"

    logging.info("Passed test_stake_add")
