import bittensor
from bittensor import logging
from bittensor.commands.senate import ProposalsCommand

from ...utils import (
    call_add_proposal,
    setup_wallet,
)


def test_root_view_proposal(local_chain, capsys):
    logging.info("Testing test_root_view_proposal")
    keypair, exec_command, wallet = setup_wallet("//Alice")

    proposals = local_chain.query("Triumvirate", "Proposals").serialize()

    assert len(proposals) == 0, "Proposals are not 0"

    call_add_proposal(local_chain, wallet)

    proposals = local_chain.query("Triumvirate", "Proposals").serialize()

    assert len(proposals) == 1, "Added proposal not found"

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
    lines = captured.out.splitlines()
    for line in lines:
        bittensor.logging.info(line)

    # Assert that the length of the lines is as expected
    assert len(lines) == 6

    # Check each line for expected content
    assert (
        lines[0] == "📡 Syncing with chain: local ..."
    ), f"Expected '📡 Syncing with chain: local ...', got {lines[0]}"
    assert (
        lines[1].strip()
        == "Proposals               Active Proposals: 1             Senate Size: 3"
    ), f"Expected 'Proposals               Active Proposals: 1             Senate Size: 3', got {lines[1].strip()}"
    assert (
        lines[2].strip().startswith("HASH")
    ), f"Expected line starting with 'HASH', got {lines[2].strip()}"
    assert (
        lines[3]
        .strip()
        .startswith(
            "0x78b8a348690f565efe3730cd8189f7388c0a896b6fd090276639c9130c0eba47"
        )
    ), f"Expected line starting with '0x78b8a348690f565efe3730cd8189f7388c0a896b6fd090276639c9130c0eba47', got {lines[3].strip()}"
    assert lines[4].strip() == "\x00)", f"Expected '\x00)', got {lines[4].strip()}"
    assert lines[5].strip() == "", f"Expected empty line, got {lines[5].strip()}"
    logging.info("Passed test_root_view_proposal")
