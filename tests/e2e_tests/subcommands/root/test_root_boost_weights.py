from bittensor.commands.root import RootSetBoostCommand
from bittensor.commands.stake import StakeCommand
from bittensor.commands.unstake import UnStakeCommand
from bittensor.commands.network import RegisterSubnetworkCommand
from bittensor.commands.register import RegisterCommand
from ...utils import new_wallet, sudo_call_set_network_limit
import bittensor


# Example test using the local_chain fixture
def test_root_get_set_weights(local_chain, capsys):
    (wallet, exec_command) = new_wallet("//Alice", "//Bob")
    assert sudo_call_set_network_limit(local_chain, wallet)

    assert not local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    exec_command(RegisterSubnetworkCommand, ["s", "create"])
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    assert (
        local_chain.query("SubtensorModule", "Uids", [1, wallet.hotkey.ss58_address])
        == None
    )

    exec_command(RegisterCommand, ["subnets", "register", "--netuid", "1"])

    # netuids = "1,2,4"
    # weights = "0.1,0.3,0.6"
    # exec_command(
    #     RootSetWeightsCommand,
    #     ["root", "weights", "--netuids", netuids, "--weights", weights],
    # )

    # weights = local_chain.query_map(
    #     "SubtensorModule", "Weights", [wallet.hotkey.ss58_address]
    # )

    netuid = "1"
    increase = "0.01"

    exec_command(
        RootSetBoostCommand,
        ["root", "boost", "--netuid", netuid, "--increase", increase],
    )

    # weights = local_chain.query("SubtensorModule", "Weights", [1])
    # assert weights == 1

    # captured = capsys.readouterr()
    # lines = captured.out.splitlines()

    # for line in lines:
    #     bittensor.logging.info(line)

    # assert len(lines) == 4
