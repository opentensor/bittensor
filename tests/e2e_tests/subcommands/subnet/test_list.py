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

    netuids = [0, 3]

    assert local_chain.query("SubtensorModule", "NetworksAdded", netuids).serialize()

    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    netuids.append(1)
    netuids.sort()

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", netuids).serialize()
