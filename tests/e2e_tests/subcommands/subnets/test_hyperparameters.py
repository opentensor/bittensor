from bittensor.commands.root import RootRegisterCommand
from bittensor.commands.network import SubnetHyperparamsCommand
from bittensor.commands.network import RegisterSubnetworkCommand
from bittensor.commands.register import RegisterCommand

from tests.e2e_tests.utils import setup_wallet


def test_metagraph(local_chain):
    # Register root as Alice
    (_, exec_command) = setup_wallet("//Alice")

    # Can call after adding a subnet
    exec_command(RootRegisterCommand, ["root", "register"])
    exec_command(RegisterSubnetworkCommand, ["subnet", "create"])
    exec_command(RegisterCommand, ["subnet", "register", "--netuid", "1"])
    exec_command(
        SubnetHyperparamsCommand, ["subnet", "hyperparameters", "--netuid", "1"]
    )
