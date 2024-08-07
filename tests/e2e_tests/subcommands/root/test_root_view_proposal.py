from bittensor.commands.senate import ProposalsCommand

from ...utils import (
    setup_wallet,
    call_add_proposal,
)


def test_root_view_proposal(local_chain, capsys):
    keypair, exec_command, wallet = setup_wallet("//Alice")

    proposals = local_chain.query("Triumvirate", "Proposals").serialize()

    assert len(proposals) == 0

    call_add_proposal(local_chain, wallet)

    proposals = local_chain.query("Triumvirate", "Proposals").serialize()

    assert len(proposals) == 1

    exec_command(
        ProposalsCommand,
        ["root", "proposals"],
    )

    simulated_output = [
        "📡 Syncing with chain: local ...",
        "     Proposals               Active Proposals: 1             Senate Size: 3     ",
        "HASH                                                                          C…",
        "0x78b8a348690f565efe3730cd8189f7388c0a896b6fd090276639c9130c0eba47            r…",
        "                                                                              \x00) ",
        "                                                                                ",
    ]

    captured = capsys.readouterr()
    output = captured.out

    for expected_line in simulated_output:
        assert (
            expected_line in output
        ), f"Expected '{expected_line}' to be in the output"
