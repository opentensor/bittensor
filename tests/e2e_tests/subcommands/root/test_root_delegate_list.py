from bittensor.commands.delegates import ListDelegatesCommand
from ...utils import setup_wallet


# delegate seems hard code the network config
def test_root_delegate_list(local_chain, capsys):
    alice_keypair, exec_command, wallet = setup_wallet("//Alice")

    # 1200 hardcoded block gap
    exec_command(
        ListDelegatesCommand,
        ["root", "list_delegates"],
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    # the command print too many lines
    assert len(lines) > 200
