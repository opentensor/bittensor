from typing import Optional
from dataclasses import asdict, dataclass, fields

from bittensor.core.chain_data.info_base import InfoBase
from bittensor_wallet.utils import SS58_FORMAT
from scalecodec.utils.ss58 import ss58_encode


@dataclass
class ColdkeySwapAnnouncementInfo(InfoBase):
    """
    Information about a coldkey swap announcement.

    This class contains information about a pending coldkey swap announcement. Announcements are used when a coldkey
    wants to declare its intention to swap to a new coldkey address. The announcement must be made before the actual
    swap can be executed, allowing time for verification and security checks.

    Attributes:
        coldkey: The SS58 address of the coldkey that made the announcement.
        execution_block: The block number when the swap can be executed (after the delay period has passed).
        new_coldkey_hash: The BlakeTwo256 hash of the new coldkey AccountId (hex string with 0x prefix). This hash
            must match the actual new coldkey when the swap is executed.

    Notes:
        - The announcement is stored on-chain and can be queried via `get_coldkey_swap_announcement()`.
        - After making an announcement, all transactions from coldkey are blocked except for `swap_coldkey_announced`.
        - The swap can only be executed after the `execution_block` has been reached.
        - See: <https://docs.learnbittensor.org/keys/coldkey-swap>
    """

    coldkey: str
    execution_block: int
    new_coldkey_hash: str

    @classmethod
    def from_query(cls, query) -> Optional["ColdkeySwapAnnouncementInfo"]:
        """
        Creates a ColdkeySwapAnnouncementInfo object from a Substrate query result.

        Parameters:
            query: Query result from Substrate `query()` call to `ColdkeySwapAnnouncements` storage function.

        Returns:
            ColdkeySwapAnnouncementInfo if announcement exists, None otherwise.
        """
        if query.value is None:
            return None

        # Decode the coldkey from the query params (if available)
        # The query result contains (BlockNumber, Hash)
        execution_block = query.value[0]
        new_coldkey_hash = "0x" + bytes(query.value[1]).hex()

        # Note: The coldkey SS58 address should be provided separately when calling this method
        # as it's the key in the storage map, not part of the value
        return cls(
            coldkey="",  # Will be set by caller
            execution_block=execution_block,
            new_coldkey_hash=new_coldkey_hash,
        )

    @classmethod
    def from_query_map_record(
        cls, record: tuple
    ) -> tuple[str, "ColdkeySwapAnnouncementInfo"]:
        """
        Creates a ColdkeySwapAnnouncementInfo object from a query_map record.

        Parameters:
            record: Data item from query_map records call to ColdkeySwapAnnouncements storage function. Structure is
                [key, value] where key is the coldkey AccountId and value contains (BlockNumber, Hash) tuple.

        Returns:
            Tuple containing:
                - SS58 address of the coldkey that made the announcement.
                - ColdkeySwapAnnouncementInfo object with announcement details.
        """
        # record[0] is the coldkey AccountId (key from storage)
        # record[1] is the value containing (BlockNumber, Hash)
        coldkey_ss58 = ss58_encode(record[0], SS58_FORMAT)
        value = record[1].value
        execution_block = value[0]
        new_coldkey_hash = "0x" + bytes(value[1]).hex()

        return coldkey_ss58, cls(
            coldkey=coldkey_ss58,
            execution_block=execution_block,
            new_coldkey_hash=new_coldkey_hash,
        )


@dataclass
class ColdkeySwapConstants:
    """
    Represents all runtime constants defined for coldkey swap operations in the SubtensorModule.

    These attributes correspond directly to on-chain configuration constants exposed by the SubtensorModule pallet.
    They define delay periods and cost requirements that govern how coldkey swap operations work within the Subtensor
    network.

    Each attribute is fetched directly from the runtime via `Subtensor.query_constant("SubtensorModule", <name>)` and
    reflects the current chain configuration at the time of retrieval.

    Attributes:
        ColdkeySwapAnnouncementDelay: The number of blocks that must elapse after making an announcement before the swap
            can be executed. This delay provides security and allows time for verification.
        ColdkeySwapReannouncementDelay: The number of blocks that must elapse between the original announcement and a
            reannouncement. This prevents spam and allows time for the original announcement to be processed.
        KeySwapCost: The cost in RAO required to make a coldkey swap announcement. This cost is charged when making the
            first announcement (not when reannouncing).

    Notes:
        - All amounts are in RAO.
        - Constants reflect the current chain configuration at the time of retrieval.
        - See: <https://docs.learnbittensor.org/keys/coldkey-swap>
    """

    ColdkeySwapAnnouncementDelay: Optional[int]
    ColdkeySwapReannouncementDelay: Optional[int]
    KeySwapCost: Optional[int]

    @classmethod
    def constants_names(cls) -> list[str]:
        """Returns the list of all constant field names defined in this dataclass.

        Returns:
            List of constant field names as strings.
        """
        return [f.name for f in fields(cls)]

    @classmethod
    def from_dict(cls, data: dict) -> "ColdkeySwapConstants":
        """
        Creates a ColdkeySwapConstants instance from a dictionary of decoded chain constants.

        Parameters:
            data: Dictionary mapping constant names to their decoded values (returned by `Subtensor.query_constant()`).

        Returns:
            ColdkeySwapConstants object with constants filled in. Fields not found in data will be set to `None`.
        """
        return cls(**{name: data.get(name) for name in cls.constants_names()})

    def to_dict(self) -> dict:
        """Converts the ColdkeySwapConstants instance to a dictionary.

        Returns:
            Dictionary mapping constant names to their values.
        """
        return asdict(self)
