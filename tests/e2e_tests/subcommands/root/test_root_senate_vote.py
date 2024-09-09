from bittensor import logging
from bittensor.commands.root import RootRegisterCommand
from bittensor.commands.senate import VoteCommand

from ...utils import (
    call_add_proposal,
    setup_wallet,
)


def test_root_senate_vote(local_chain, capsys, monkeypatch):
    logging.info("Testing test_root_senate_vote")
    keypair, exec_command, wallet = setup_wallet("//Alice")
    monkeypatch.setattr("rich.prompt.Confirm.ask", lambda self: True)

    exec_command(
        RootRegisterCommand,
        ["root", "register"],
    )

    members = local_chain.query("Triumvirate", "Members")
    proposals = local_chain.query("Triumvirate", "Proposals").serialize()

    assert len(members) == 3, f"Expected 3 Triumvirate members, found {len(members)}"
    assert (
        len(proposals) == 0
    ), f"Expected 0 initial Triumvirate proposals, found {len(proposals)}"

    call_add_proposal(local_chain, wallet)

    proposals = local_chain.query("Triumvirate", "Proposals").serialize()

    assert (
        len(proposals) == 1
    ), f"Expected 1 proposal in the Triumvirate after addition, found {len(proposals)}"
    proposal_hash = proposals[0]

    exec_command(
        VoteCommand,
        ["root", "senate_vote", "--proposal", proposal_hash],
    )

    voting = local_chain.query("Triumvirate", "Voting", [proposal_hash]).serialize()

    assert len(voting["ayes"]) == 1, f"Expected 1 ayes, found {len(voting['ayes'])}"
    assert (
        voting["ayes"][0] == wallet.hotkey.ss58_address
    ), "wallet hotkey address doesn't match 'ayes' address"
    logging.info("Passed test_root_senate_vote")
