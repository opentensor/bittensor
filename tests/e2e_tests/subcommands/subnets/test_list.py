from bittensor.commands.root import RootRegisterCommand
from bittensor.commands.network import RegisterSubnetworkCommand
from bittensor.commands.network import SubnetListCommand
from tests.e2e_tests.utils import setup_wallet


def test_list(local_chain):
    # Register root as Alice
    (_, exec_command) = setup_wallet("//Alice")

    # Can call when no subnets
    exec_command(SubnetListCommand, ["subnet", "list"])

    # Can call after adding a subnet
    exec_command(RootRegisterCommand, ["root", "register"])
    exec_command(RegisterSubnetworkCommand, ["subnet", "create"])
    exec_command(SubnetListCommand, ["subnet", "list"])
