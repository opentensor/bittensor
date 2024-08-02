import bittensor
from bittensor.commands import RegisterSubnetworkCommand
from tests.e2e_tests.utils import setup_wallet

"""
Test the list command before and after registering subnets. 

Verify that:
* list of subnets gets displayed
-------------------------
* Register a subnets
* Ensure is visible in list cmd
"""


def test_list_command(local_chain, capsys):
    # Register root as Alice
    keypair, exec_command, wallet = setup_wallet("//Alice")

    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    subnets = subtensor.get_subnets()

    assert len(subnets) == 2

    exec_command(RegisterSubnetworkCommand, ["s", "create"])
    # Verify subnet 1 created successfully
    subnets = subtensor.get_subnets()
    assert len(subnets) == 3
