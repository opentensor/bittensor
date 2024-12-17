from bittensor.core.chain_data.utils import decode_account_id


# Senate / Proposal data
class ProposalVoteData:
    index: int
    threshold: int
    ayes: list[str]
    nays: list[str]
    end: int

    def __init__(self, proposal_dict: dict) -> None:
        self.index = proposal_dict["index"]
        self.threshold = proposal_dict["threshold"]
        self.ayes = self.decode_ss58_tuples(proposal_dict["ayes"])
        self.nays = self.decode_ss58_tuples(proposal_dict["nays"])
        self.end = proposal_dict["end"]

    @staticmethod
    def decode_ss58_tuples(line: tuple):
        """Decodes a tuple of ss58 addresses formatted as bytes tuples."""
        return [decode_account_id(line[x][0]) for x in range(len(line))]
