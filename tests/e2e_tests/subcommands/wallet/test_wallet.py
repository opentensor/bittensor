from bittensor.commands.list import ListCommand
from ...utils import setup_wallet

def test_wallet_list(local_chain, capsys):
    (keypair, exec_command) = setup_wallet("//Alice")

    exec_command(
        ListCommand,
        [
            "wallet",
            "list",
        ],
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert(len(lines) == 4)
    assert("└──" in lines[1])
    assert("default (5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY)" in lines[2])
    assert("└── default (5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY)" in lines[3])
