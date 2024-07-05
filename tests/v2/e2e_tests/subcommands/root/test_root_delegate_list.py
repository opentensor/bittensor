import pytest

from bittensor.v2.commands.delegates import ListDelegatesCommand
from ...utils import setup_wallet


# delegate seems hard code the network config
@pytest.mark.asyncio
async def test_root_delegate_list(local_chain, capsys):
    alice_keypair, exec_command, wallet = await setup_wallet("//Alice")

    # 1200 hardcoded block gap
    await exec_command(
        ListDelegatesCommand,
        ["root", "list_delegates"],
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    # the command print too many lines
    assert len(lines) > 200
