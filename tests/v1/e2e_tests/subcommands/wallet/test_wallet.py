from bittensor.v1.commands.list import ListCommand
from bittensor.v1.subtensor import subtensor

from ...utils import setup_wallet


def test_wallet_list(local_chain: subtensor, capsys):
    keypair, exec_command, wallet = setup_wallet("//Alice")

    exec_command(
        ListCommand,
        [
            "wallet",
            "list",
        ],
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    # can't check the output now since there is a info about bittensor version
    assert len(lines) >= 4
    # assert "└──" in lines[1]
    # assert "default (5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY)" in lines[2]
    # assert "└── default (5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY)" in lines[3]
