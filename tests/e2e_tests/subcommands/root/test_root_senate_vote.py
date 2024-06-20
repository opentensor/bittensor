# from bittensor.commands.senate import VoteCommand
# from bittensor.commands.root import RootRegisterCommand
#
# from ...utils import (
#     setup_wallet,
#     call_add_proposal,
# )


# TODO: fix after commands and cli are async migrated
# def test_root_senate_vote(local_chain, capsys, monkeypatch):
#     keypair, exec_command, wallet = setup_wallet("//Alice")
#     monkeypatch.setattr("rich.prompt.Confirm.ask", lambda self: True)
#
#     exec_command(
#         RootRegisterCommand,
#         ["root", "register"],
#     )
#
#     members = local_chain.query("Triumvirate", "Members")
#     proposals = local_chain.query("Triumvirate", "Proposals").serialize()
#
#     assert len(members) == 3
#     assert len(proposals) == 0
#
#     call_add_proposal(local_chain, wallet)
#
#     proposals = local_chain.query("Triumvirate", "Proposals").serialize()
#
#     assert len(proposals) == 1
#     proposal_hash = proposals[0]
#
#     exec_command(
#         VoteCommand,
#         ["root", "senate_vote", "--proposal", proposal_hash],
#     )
#
#     voting = local_chain.query("Triumvirate", "Voting", [proposal_hash]).serialize()
#
#     assert len(voting["ayes"]) == 1
#     assert voting["ayes"][0] == wallet.hotkey.ss58_address
