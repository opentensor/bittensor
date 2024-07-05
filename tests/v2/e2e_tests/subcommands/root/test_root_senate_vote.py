import pytest

from bittensor.v2.commands.senate import VoteCommand
from bittensor.v2.commands.root import RootRegisterCommand

from ...utils import (
    setup_wallet,
    call_add_proposal,
)


@pytest.mark.asyncio
async def test_root_senate_vote(local_chain, capsys, monkeypatch):
    keypair, exec_command, wallet = await setup_wallet("//Alice")
    monkeypatch.setattr("rich.prompt.Confirm.ask", lambda self: True)

    await exec_command(
        RootRegisterCommand,
        ["root", "register"],
    )

    members = local_chain.query("Triumvirate", "Members")
    proposals = local_chain.query("Triumvirate", "Proposals").serialize()

    assert len(members) == 3
    assert len(proposals) == 0

    call_add_proposal(local_chain, wallet)

    proposals = local_chain.query("Triumvirate", "Proposals").serialize()

    assert len(proposals) == 1
    proposal_hash = proposals[0]

    await exec_command(
        VoteCommand,
        ["root", "senate_vote", "--proposal", proposal_hash],
    )

    voting = local_chain.query("Triumvirate", "Voting", [proposal_hash]).serialize()

    assert len(voting["ayes"]) == 1
    assert voting["ayes"][0] == wallet.hotkey.ss58_address
