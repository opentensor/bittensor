from bittensor.commands.root import RootList
from ...utils import new_wallet
from bittensor.commands.network import RegisterSubnetworkCommand
import bittensor


# test case to list the root network
def test_root_list(local_chain, capsys):
    (wallet, exec_command) = new_wallet("//Alice", "//Bob")

    exec_command(RootList, ["root", "list"])
    captured = capsys.readouterr()
    lines = captured.out.split("\n")

    assert len(lines) >= 4
    bittensor.logging.info(lines)

    # assert "Root Network" in lines[0]
    # assert "UID  NAME  ADDRESS  STAKE" in lines[1]

    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    exec_command(RootList, ["root", "list"])
    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    assert len(lines) >= 4
    # assert "Root Network" in lines[0]
    # assert "UID  NAME  ADDRESS  STAKE" in lines[1]
    # assert "1" in lines[2]
