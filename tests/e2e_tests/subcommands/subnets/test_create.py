from bittensor.commands.root import RootRegisterCommand
from bittensor.commands.network import RegisterSubnetworkCommand
from tests.e2e_tests.utils import setup_wallet


def test_list(local_chain):
    # Register root as Alice
    (_, exec_command) = setup_wallet("//Alice")

    assert (
        local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize() == False
    )

    exec_command(RootRegisterCommand, ["root", "register"])
    exec_command(RegisterSubnetworkCommand, ["subnet", "create"])

    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]) == True
