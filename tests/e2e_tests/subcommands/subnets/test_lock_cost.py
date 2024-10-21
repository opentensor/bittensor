from bittensor.commands.root import RootRegisterCommand
from bittensor.commands.network import RegisterSubnetworkCommand
from bittensor.commands.network import SubnetLockCostCommand
from bittensor.commands.register import RegisterCommand

from tests.e2e_tests.utils import setup_wallet


def test_lock_cost(local_chain):
    # Register root as Alice
    (_, exec_command) = setup_wallet("//Alice")

    # Can call when no subnets
    exec_command(SubnetLockCostCommand, ["subnet", "lock_cost"])

    # Can call after adding a subnet
    exec_command(RootRegisterCommand, ["root", "register"])
    exec_command(RegisterSubnetworkCommand, ["subnet", "create"])
    exec_command(RegisterCommand, ["s", "register", "--netuid", "1"])
    exec_command(SubnetLockCostCommand, ["subnet", "lock_cost"])
