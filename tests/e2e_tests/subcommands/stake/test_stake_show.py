from bittensor.commands.stake import StakeShow
from ...utils import setup_wallet


# Example test using the local_chain fixture
def test_stake_show(local_chain, capsys):
    (keypair, exec_command) = setup_wallet("//Alice")

    exec_command(StakeShow, ["stake", "show"])
    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    assert len(lines) >= 5
    # assert "Coldkey" in lines[0]
    # assert "default" in lines[1]
    # assert "default" in lines[2]
