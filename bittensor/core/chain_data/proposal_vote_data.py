from typing import List, TypedDict


# Senate / Proposal data
class ProposalVoteData(TypedDict):
    """
    This TypedDict represents the data structure for a proposal vote in the Senate.

    Attributes:
        index (int): The index of the proposal.
        threshold (int): The threshold required for the proposal to pass.
        ayes (List[str]): List of senators who voted 'aye'.
        nays (List[str]): List of senators who voted 'nay'.
        end (int): The ending timestamp of the voting period.
    """

    index: int
    threshold: int
    ayes: List[str]
    nays: List[str]
    end: int
