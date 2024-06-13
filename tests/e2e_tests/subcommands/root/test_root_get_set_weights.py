from bittensor.commands.root import RootSetWeightsCommand, RootGetWeightsCommand
from bittensor.commands.network import RegisterSubnetworkCommand
from ...utils import new_wallet, sudo_call_set_network_limit


# we can't test it now since the root network weights can't be set
# Test case to set weights for root network and get the weights
def test_root_get_set_weights(local_chain, capsys):
    (wallet, exec_command) = new_wallet("//Alice", "//Bob")
    assert sudo_call_set_network_limit(local_chain, wallet)
    assert not local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    exec_command(RegisterSubnetworkCommand, ["s", "create"])
    exec_command(RegisterSubnetworkCommand, ["s", "create"])
    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()
    assert local_chain.query("SubtensorModule", "NetworksAdded", [2]).serialize()
    assert local_chain.query("SubtensorModule", "NetworksAdded", [4]).serialize()

    netuids = "1,2,4"
    weights = "0.1,0.3,0.6"

    # this command need update, should set the netuid. subtensor not accept the weight set for root network
    exec_command(
        RootSetWeightsCommand,
        ["root", "weights", "--netuids", netuids, "--weights", weights],
    )

    weights = local_chain.query("SubtensorModule", "Weights", [1, 0])

    exec_command(
        RootGetWeightsCommand,
        ["root", "get_weights"],
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
