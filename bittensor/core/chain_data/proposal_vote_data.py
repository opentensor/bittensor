from dataclasses import dataclass

from bittensor.core.chain_data.info_base import InfoBase


@dataclass
class ProposalVoteData(InfoBase):
    """
    Senate / Proposal data
    """

    index: int
    threshold: int
    ayes: list[str]
    nays: list[str]
    end: int

    @classmethod
    def from_dict(cls, proposal_dict: dict) -> "ProposalVoteData":
        return cls(
            ayes=proposal_dict["ayes"],
            end=proposal_dict["end"],
            index=proposal_dict["index"],
            nays=proposal_dict["nays"],
            threshold=proposal_dict["threshold"],
        )
