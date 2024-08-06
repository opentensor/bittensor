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

    netuid = 0

    assert local_chain.query("SubtensorModule", "NetworksAdded", [netuid]).serialize()

    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    netuid - 1

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [netuid]).serialize()
