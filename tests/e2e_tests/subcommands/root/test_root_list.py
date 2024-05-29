from bittensor.commands.root import RootList
from ...utils import new_wallet, sudo_call_set_network_limit
from bittensor.commands.network import RegisterSubnetworkCommand
import bittensor


# Example test using the local_chain fixture
def test_root_list(local_chain, capsys):
    (wallet, exec_command) = new_wallet("//Alice", "//Bob")

    # exec_command(RootList, ["root", "list"])
    # captured = capsys.readouterr()
    # lines = captured.out.split("\n")

    # assert len(lines) == 4
    # assert "Root Network" in lines[0]
    # assert "UID  NAME  ADDRESS  STAKE" in lines[1]

    # exec_command(RegisterSubnetworkCommand, ["s", "create"])

    exec_command(RootList, ["root", "list"])
    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    for line in lines:
        bittensor.logging.info(line)

    assert len(lines) == 4
    assert "Root Network" in lines[0]
    assert "UID  NAME  ADDRESS  STAKE" in lines[1]
    assert "1" in lines[2]
