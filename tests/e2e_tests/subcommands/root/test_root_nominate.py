from bittensor.commands.delegates import NominateCommand
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
def test_root_nominate(local_chain, capsys):
    (wallet, exec_command) = new_wallet("//Alice", "//Bob")

    delegates = local_chain.query(
        "SubtensorModule",
        "Delegates",
        [wallet.hotkey.ss58_address],
    )

    assert delegates == 11796

    assert sudo_call_set_network_limit(local_chain, wallet)
    assert sudo_call_set_target_stakes_per_interval(local_chain, wallet)

    exec_command(RegisterSubnetworkCommand, ["s", "create"])
    exec_command(RegisterCommand, ["s", "register", "--neduid", "1"])

    exec_command(
        NominateCommand,
        [
            "root",
            "nominate",
        ],
    )

    delegates = local_chain.query(
        "SubtensorModule",
        "Delegates",
        [wallet.hotkey.ss58_address],
    )

    assert delegates == 11796
