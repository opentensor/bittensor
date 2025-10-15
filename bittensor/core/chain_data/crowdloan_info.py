from dataclasses import dataclass
from typing import Optional

from bittensor.core.chain_data.utils import decode_account_id
from bittensor.utils.balance import Balance


@dataclass
class CrowdloanInfo:
    """
    Represents a single on-chain crowdloan campaign from the `pallet-crowdloan`.

    Each instance reflects the current state of a specific crowdloan as stored in chain storage. It includes funding
    details, creator information, contribution totals, and optional call/target data that define what happens upon
    successful finalization.

    Attributes:
        id: The unique identifier (index) of the crowdloan.
        creator: The SS58 address of the creator (campaign initiator).
        deposit: The creator's initial deposit locked to open the crowdloan.
        min_contribution: The minimum contribution amount allowed per participant.
        end: The block number when the campaign ends.
        cap: The maximum amount to be raised (funding cap).
        funds_account: The account ID holding the crowdloan’s funds.
        raised: The total amount raised so far.
        target_address: Optional SS58 address to which funds are transferred upon success.
        call: Optional encoded runtime call (e.g., a `register_leased_network` extrinsic) to execute on finalize.
        finalized: Whether the crowdloan has been finalized on-chain.
        contributors_count: Number of unique contributors currently participating.
    """

    id: int
    creator: str
    deposit: Balance
    min_contribution: Balance
    end: int
    cap: Balance
    funds_account: str
    raised: Balance
    target_address: Optional[str]
    call: Optional[str]
    finalized: bool
    contributors_count: int

    @classmethod
    def from_dict(cls, idx: int, data: dict) -> "CrowdloanInfo":
        """Returns a CrowdloanInfo object from decoded chain data."""
        return cls(
            id=idx,
            creator=decode_account_id(data["creator"]),
            deposit=Balance.from_rao(data["deposit"]),
            min_contribution=Balance.from_rao(data["min_contribution"]),
            end=data["end"],
            cap=Balance.from_rao(data["cap"]),
            funds_account=decode_account_id(data["funds_account"])
            if data.get("funds_account")
            else None,
            raised=Balance.from_rao(data["raised"]),
            target_address=decode_account_id(data.get("target_address"))
            if data.get("target_address")
            else None,
            call=data.get("call") if data.get("call") else None,
            finalized=data["finalized"],
            contributors_count=data["contributors_count"],
        )


@dataclass
class CrowdloanConstants:
    """
    Represents all runtime constants defined in the `pallet-crowdloan`.

    These attributes correspond directly to on-chain configuration constants exposed by the Crowdloan pallet. They
    define contribution limits, duration bounds, pallet identifiers, and refund behavior that govern how crowdloan
    campaigns operate within the Subtensor network.

    Each attribute is fetched directly from the runtime via `Subtensor.substrate.get_constant("Crowdloan", <name>)` and
    reflects the current chain configuration at the time of retrieval.

    Attributes:
        AbsoluteMinimumContribution: The absolute minimum amount required to contribute to any crowdloan.
        MaxContributors: The maximum number of unique contributors allowed per crowdloan.
        MaximumBlockDuration: The maximum allowed duration (in blocks) for a crowdloan campaign.
        MinimumDeposit: The minimum deposit required from the creator to open a new crowdloan.
        MinimumBlockDuration: The minimum allowed duration (in blocks) for a crowdloan campaign.
        RefundContributorsLimit: The maximum number of contributors that can be refunded in single on-chain refund call.

    Note:
        All Balance amounts are in RAO.
    """

    AbsoluteMinimumContribution: Optional["Balance"]
    MaxContributors: Optional[int]
    MaximumBlockDuration: Optional[int]
    MinimumDeposit: Optional["Balance"]
    MinimumBlockDuration: Optional[int]
    RefundContributorsLimit: Optional[int]

    @classmethod
    def constants_names(cls) -> list[str]:
        """Returns the list of all constant field names defined in this dataclass."""
        from dataclasses import fields

        return [f.name for f in fields(cls)]

    @classmethod
    def from_dict(cls, data: dict) -> "CrowdloanConstants":
        """
        Creates a `CrowdloanConstants` instance from a dictionary of decoded chain constants.

        Parameters:
            data: Dictionary mapping constant names to their decoded values (returned by `Subtensor.query_constant()`).

        Returns:
            CrowdloanConstants: The structured dataclass with constants filled in.
        """
        return cls(**{name: data.get(name) for name in cls.constants_names()})
