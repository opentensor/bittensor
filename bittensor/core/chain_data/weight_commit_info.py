from dataclasses import dataclass
from bittensor.core.chain_data.utils import decode_account_id
from typing import Optional


@dataclass
class WeightCommitInfo:
    """
    Data class representing weight commit information.

    Attributes:
        ss58: The SS58 address of the committer
        commit_block: The block number of the commitment.
        commit_hex: The serialized weight commit data as hex string
        reveal_round: The round number for reveal
    """

    ss58: str
    commit_block: Optional[int]
    commit_hex: str
    reveal_round: int

    @classmethod
    def from_vec_u8(cls, data: tuple) -> tuple[str, str, int]:
        """
        Creates a WeightCommitInfo instance

        Parameters:
            data: Tuple containing ((AccountId,), (commit_data,), round_number)

        Returns:
            WeightCommitInfo: A new instance with the decoded data

        Note:
            This method is used when querying a block or block hash where storage functions `CRV3WeightCommitsV2` does
            not exist in Subtensor module.
        """
        account_id, commit_block, commit_data, round_number = data

        account_id_ = account_id[0] if isinstance(account_id, tuple) else account_id
        commit_data = commit_data[0] if isinstance(commit_data, tuple) else commit_data
        commit_hex = "0x" + "".join(format(x, "02x") for x in commit_data)

        return decode_account_id(account_id_), commit_hex, round_number

    @classmethod
    def from_vec_u8_v2(cls, data: tuple) -> tuple[str, int, str, int]:
        """
        Creates a WeightCommitInfo instance

        Parameters:
            data: Tuple containing ((AccountId,), (commit_block, ) (commit_data,), round_number)

        Returns:
            WeightCommitInfo: A new instance with the decoded data
        """
        account_id, commit_block, commit_data, round_number = data

        account_id_ = account_id[0] if isinstance(account_id, tuple) else account_id
        commit_block = (
            commit_block[0] if isinstance(commit_block, tuple) else commit_block
        )
        commit_data = commit_data[0] if isinstance(commit_data, tuple) else commit_data
        commit_hex = "0x" + "".join(format(x, "02x") for x in commit_data)

        return decode_account_id(account_id_), commit_block, commit_hex, round_number
