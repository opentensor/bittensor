from dataclasses import asdict, dataclass, fields
from typing import Optional

from async_substrate_interface.types import ScaleObj

from bittensor.core.chain_data.utils import decode_account_id


@dataclass
class ColdkeySwapAnnouncementInfo:
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
    def from_query(
        cls, coldkey_ss58: str, query: "ScaleObj"
    ) -> Optional["ColdkeySwapAnnouncementInfo"]:
        """
        Creates a ColdkeySwapAnnouncementInfo object from a Substrate query result.

        Parameters:
            coldkey_ss58: The SS58 address of the coldkey that made the announcement.
            query: Query result from Substrate `query()` call to `ColdkeySwapAnnouncements` storage function.

        Returns:
            ColdkeySwapAnnouncementInfo if announcement exists, None otherwise.
        """
        if not getattr(query, "value", None):
            return None

        execution_block = query.value[0]
        new_coldkey_hash = "0x" + bytes(query.value[1][0]).hex()
        return cls(
            coldkey=coldkey_ss58,
            execution_block=execution_block,
            new_coldkey_hash=new_coldkey_hash,
        )

    @classmethod
    def from_record(cls, record: tuple) -> "ColdkeySwapAnnouncementInfo":
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
        coldkey_ss58 = decode_account_id(record[0])
        announcement_data = record[1].value
        execution_block = announcement_data[0]
        new_coldkey_hash = "0x" + bytes(announcement_data[1][0]).hex()

        return cls(
            coldkey=coldkey_ss58,
            execution_block=execution_block,
            new_coldkey_hash=new_coldkey_hash,
        )


@dataclass
class ColdkeySwapConstants:
    """
    Represents runtime constants for coldkey swap operations in the SubtensorModule.

    This class contains runtime constants that define cost requirements for coldkey swap operations.
    Note: For delay values (ColdkeySwapAnnouncementDelay and ColdkeySwapReannouncementDelay), use the dedicated
    query methods `get_coldkey_swap_announcement_delay()` and `get_coldkey_swap_reannouncement_delay()` instead,
    as these are storage values, not runtime constants.

    Attributes:
        KeySwapCost: The cost in RAO required to make a coldkey swap announcement. This cost is charged when making the
            first announcement (not when reannouncing). This is a runtime constant (queryable via constants).

    Notes:
        - All amounts are in RAO.
        - Values reflect the current chain configuration at the time of retrieval.
        - See: <https://docs.learnbittensor.org/keys/coldkey-swap>
    """

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
