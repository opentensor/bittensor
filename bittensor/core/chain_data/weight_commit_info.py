from dataclasses import dataclass


@dataclass
class WeightCommitInfo:
    """
    Data class representing weight commit information.

    Attributes:
        ss58 (str): The SS58 address of the committer
        commit_hex (str): The serialized weight commit data as hex string
        reveal_round (int): The round number for reveal
    """

    ss58: str
    commit_hex: str
    reveal_round: int

    @classmethod
    def from_vec_u8(cls, data: tuple) -> tuple[str, str, int]:
        """
        Creates a WeightCommitInfo instance

        Args:
            data (tuple): Tuple containing ((AccountId,), (commit_data,), round_number)

        Returns:
            WeightCommitInfo: A new instance with the decoded data
        """
        account_id, commit_data, round_number = data

        account_id_ = account_id[0] if isinstance(account_id, tuple) else account_id
        commit_data = commit_data[0] if isinstance(commit_data, tuple) else commit_data
        commit_hex = "0x" + "".join(format(x, "02x") for x in commit_data)

        return account_id_, commit_hex, round_number
