from bittensor.commands.root import (
    RootSetBoostCommand,
    RootSetWeightsCommand,
    RootRegisterCommand,
)
from bittensor.commands.stake import StakeCommand
from bittensor.commands.network import RegisterSubnetworkCommand
from bittensor.commands.register import RegisterCommand
from bittensor.commands import SubnetSudoCommand

from ...utils import (
    setup_wallet,
)

"""
Test the root set/get weights mechanism. 

Verify that:
* Fill in the test parameters here
* 
* 
* 

"""


def test_root_get_set_weights(local_chain, capsys):
    """Test case to set weights for root network"""

    keypair, exec_command, wallet = setup_wallet("//Alice")

    exec_command(
        SubnetSudoCommand,
        [
            "sudo",
            "set",
            "hyperparameters",
            "--param",
            "network_rate_limit",
            "--value",
            "1",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    output = capsys.readouterr().out
    assert "" in output  # assert the correct output is returned to user

    exec_command(
        SubnetSudoCommand,
        [
            "sudo",
            "set",
            "hyperparameters",
            "--param",
            "weights_set_rate_limit",
            "--value",
            "1",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    output = capsys.readouterr().out
    assert "" in output  # assert the correct output is returned to user

    assert not local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    exec_command(RegisterSubnetworkCommand, ["s", "create"])
    exec_command(RegisterSubnetworkCommand, ["s", "create"])
    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    output = capsys.readouterr().out
    assert "" in output  # assert no errors or failures after creating networks

    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()
    assert (
        local_chain.query(
            "SubtensorModule", "Uids", [0, wallet.hotkey.ss58_address]
        ).value
        is None
    )

    exec_command(
        RootRegisterCommand,
        ["root", "register"],
    )

    output = capsys.readouterr().out
    assert "" in output  # assert the correct output is returned to user

    exec_command(RegisterCommand, ["subnets", "register", "--netuid", "0"])

    output = capsys.readouterr().out
    assert "" in output  # assert the correct output is returned to user

    assert (
        local_chain.query("SubtensorModule", "Uids", [0, wallet.hotkey.ss58_address])
        == 0
    )

    assert not local_chain.query("SubtensorModule", "Weights", [0, 0])

    netuids = "1,2,4"
    weights = "0.1,0.3,0.6"
    exec_command(
        RootSetWeightsCommand,
        ["root", "weights", "--netuids", netuids, "--weights", weights],
    )

    output = capsys.readouterr().out
    assert "" in output  # assert the correct output is returned to user

    first_weight_vec = local_chain.query("SubtensorModule", "Weights", [0, 0])[0]
    assert first_weight_vec[0] == 1
    first_weight = first_weight_vec[1]

    netuid = "1"
    increase = "0.01"

    exec_command(
        RootSetBoostCommand,
        ["root", "boost", "--netuid", netuid, "--increase", increase],
    )

    output = capsys.readouterr().out
    assert "" in output  # assert the correct output is returned to user

    first_weight_vec = local_chain.query("SubtensorModule", "Weights", [0, 0])[0]
    assert first_weight_vec[0] == 1
    new_first_weight = first_weight_vec[1]

    assert new_first_weight > first_weight
    first_weight = new_first_weight

    stake_amount = 2
    exec_command(StakeCommand, ["stake", "add", "--amount", str(stake_amount)])

    output = capsys.readouterr().out
    assert "" in output  # assert the correct output is returned to user

    # set the min stake for the account to set weights
    exec_command(
        SubnetSudoCommand,
        [
            "sudo",
            "set",
            "hyperparameters",
            "--param",
            "weights_min_stake",
            "--value",
            str(stake_amount),
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    output = capsys.readouterr().out
    assert "" in output  # assert the correct output is returned to user

    exec_command(
        RootSetBoostCommand,
        ["root", "boost", "--netuid", netuid, "--increase", increase],
    )

    output = capsys.readouterr().out
    assert "" in output

    first_weight_vec = local_chain.query("SubtensorModule", "Weights", [0, 0])[0]
    assert first_weight_vec[0] == 1
    new_first_weight = first_weight_vec[1]

    assert new_first_weight > first_weight
