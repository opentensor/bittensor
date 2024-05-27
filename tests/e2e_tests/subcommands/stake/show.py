# test stake show add and remove

import bittensor
from bittensor.commands.network import RegisterSubnetworkCommand, SubnetSudoCommand
from bittensor.commands.stake import StakeCommand, StakeShow
from bittensor.commands.transfer import TransferCommand
from bittensor.subtensor import subtensor

from ...utils import get_wallet, setup_wallet


# Example test using the local_chain fixture
def test_stake_show(local_chain: subtensor):
    netuid = 1
    wallet = get_wallet("//Alice", "//Bob")

    assert local_chain.set_network_hyperparameter(
        wallet,
        "network_rate_limit",
        2,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )

    assert local_chain.register_subnetwork(
        wallet, wait_for_finalization=True, wait_for_inclusion=True
    )

    subnet_list = local_chain.get_all_subnet_netuids()
    assert netuid in subnet_list
