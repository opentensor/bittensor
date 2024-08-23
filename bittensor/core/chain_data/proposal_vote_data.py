from typing import List, TypedDict


# Senate / Proposal data
class ProposalVoteData(TypedDict):
    index: int
    threshold: int
    ayes: List[str]
    nays: List[str]
    end: int


