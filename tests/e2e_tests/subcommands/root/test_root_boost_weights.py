from bittensor.commands.root import (
    RootSetBoostCommand,
    RootSetWeightsCommand,
    RootRegisterCommand,
)
from bittensor.commands.stake import StakeCommand
from bittensor.commands.network import RegisterSubnetworkCommand
from bittensor.commands.register import RegisterCommand
from ...utils import (
    new_wallet,
    sudo_call_set_network_limit,
    sudo_call_set_weight_limit,
    sudo_call_set_min_stake,
)
import bittensor


# Test case to set weights for root network
def test_root_get_set_weights(local_chain, capsys):
    (wallet, exec_command) = new_wallet("//Alice", "//Bob")
    assert sudo_call_set_network_limit(local_chain, wallet)
    root_netuid = 0
    assert sudo_call_set_weight_limit(local_chain, wallet, root_netuid)

    assert not local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    exec_command(RegisterSubnetworkCommand, ["s", "create"])
    exec_command(RegisterSubnetworkCommand, ["s", "create"])
    exec_command(RegisterSubnetworkCommand, ["s", "create"])
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    assert (
        local_chain.query("SubtensorModule", "Uids", [1, wallet.hotkey.ss58_address])
        == None
    )

    exec_command(
        RootRegisterCommand,
        ["root", "register"],
    )

    exec_command(RegisterCommand, ["subnets", "register", "--netuid", "0"])

    netuids = "1,2,4"
    weights = "0.1,0.3,0.6"
    exec_command(
        RootSetWeightsCommand,
        ["root", "weights", "--netuids", netuids, "--weights", weights],
    )

    netuid = "1"
    increase = "0.01"

    exec_command(
        RootSetBoostCommand,
        ["root", "boost", "--netuid", netuid, "--increase", increase],
    )

    stake_amount = 2
    exec_command(StakeCommand, ["stake", "add", "--amount", str(stake_amount)])

    # set the min stake for the account to set weights
    assert sudo_call_set_min_stake(local_chain, wallet, stake_amount)

    exec_command(
        RootSetBoostCommand,
        ["root", "boost", "--netuid", netuid, "--increase", increase],
    )
