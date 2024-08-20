from bittensor import logging
from bittensor.commands.delegates import ListDelegatesCommand

from ...utils import setup_wallet


# delegate seems hard code the network config
def test_root_delegate_list(local_chain, capsys):
    logging.info("Testing test_root_delegate_list")
    alice_keypair, exec_command, wallet = setup_wallet("//Alice")

    # 1200 hardcoded block gap
    exec_command(
        ListDelegatesCommand,
        ["root", "list_delegates"],
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    # the command print too many lines
    # To:do - Find a better to validate list delegates
    assert len(lines) > 200
    logging.info("Passed test_root_delegate_list")
