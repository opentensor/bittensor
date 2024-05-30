from bittensor.commands.senate import SenateCommand
from bittensor.commands.stake import StakeCommand
from bittensor.commands.unstake import UnStakeCommand
from bittensor.commands.network import RegisterSubnetworkCommand
from bittensor.commands.register import RegisterCommand
from ...utils import new_wallet, sudo_call_add_senate_member
import bittensor


# Example test using the local_chain fixture
def test_root_get_set_weights(local_chain, capsys)
    (wallet, exec_command) = new_wallet("//Alice", "//Bob")

    members = local_chain.query("SenateMembers", "Members").serialize()
    assert len(members) == 3

    exec_command(
        SenateCommand,
        ["root", "senate"],
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    assert len(lines) == 7

    sudo_call_add_senate_member(local_chain, wallet)

    members = local_chain.query("SenateMembers", "Members").serialize()
    bittensor.logging.info(members)
    assert len(members) == 4

    exec_command(
        SenateCommand,
        ["root", "senate"],
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    assert len(lines) == 8
