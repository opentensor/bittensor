from bittensor.commands.root import RootList
from ...utils import setup_wallet


# Example test using the local_chain fixture
def test_stake_show(local_chain, capsys):
    (keypair, exec_command) = setup_wallet("//Alice")

    exec_command(RootList, ["root", "list"])
    captured = capsys.readouterr()
    lines = captured.out.split("\n")

    assert len(lines) == 4
    assert "Root Network" in lines[0]
    assert "UID  NAME  ADDRESS  STAKE" in lines[1]
