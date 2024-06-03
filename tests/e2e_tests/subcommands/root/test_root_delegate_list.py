from bittensor.commands.delegates import ListDelegatesCommand
from bittensor.commands.root import RootRegisterCommand
from bittensor.commands.delegates import SetTakeCommand
from ...utils import (
    new_wallet,
    call_add_proposal,
)


# delegate seems hard code the network config
def test_root_delegate_list(local_chain, capsys):
    (wallet, exec_command) = new_wallet("//Alice", "//Bob")

    # 1200 hardcoded block gap
    exec_command(
        ListDelegatesCommand,
        ["root", "list_delegates"],
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    # the command print too many lines
    assert len(lines) > 200
