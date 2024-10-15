from bittensor.commands.root import RootRegisterCommand
from bittensor.commands.network import RegisterSubnetworkCommand
from bittensor.commands.register import RegisterCommand

from tests.e2e_tests.utils import setup_wallet


def test_register(local_chain):
    # Register root as Alice
    (_, exec_command) = setup_wallet("//Alice")
    exec_command(RootRegisterCommand, ["root", "register"])

    # Register Bob subnet
    (keypair, exec_command) = setup_wallet("//Bob")
    exec_command(RegisterSubnetworkCommand, ["subnet", "create"])
    exec_command(RegisterCommand, ["subnet", "register", "--netuid", "1"])
    assert (
        local_chain.query("SubtensorModule", "SubnetOwner", [1]).serialize()
        == keypair.ss58_address
    )
