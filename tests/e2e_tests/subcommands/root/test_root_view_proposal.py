from bittensor.commands.senate import ProposalsCommand

from ...utils import (
    new_wallet,
    call_add_proposal,
)
import bittensor


# Example test using the local_chain fixture
def test_root_view_proposal(local_chain, capsys):
    (wallet, exec_command) = new_wallet("//Alice", "//Bob")

    proposals = local_chain.query("Triumvirate", "Proposals").serialize()

    assert len(proposals) == 0

    call_add_proposal(local_chain, wallet)

    proposals = local_chain.query("Triumvirate", "Proposals").serialize()

    assert len(proposals) == 1

    exec_command(
        ProposalsCommand,
        ["root", "proposals"],
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    for line in lines:
        bittensor.logging.info(line)

    assert len(lines) == 6
