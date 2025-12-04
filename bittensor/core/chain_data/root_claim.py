from enum import Enum
from typing import Literal, Callable
from dataclasses import dataclass


@dataclass
class KeepSubnetsDescriptor:
    """Descriptor that allows callable syntax for KeepSubnets variant."""

    subnets: list[int]

    def __post_init__(self):
        """Validate subnets after initialization."""
        if not self.subnets:
            raise ValueError(
                "KeepSubnets must have at least one subnet represented by a netuid in the list."
            )
        if not all(isinstance(netuid, int) for netuid in self.subnets):
            raise ValueError("All subnet IDs must be integers in the list.")

    def to_dict(self) -> dict:
        """Converts the descriptor to the required dictionary format."""
        return {"KeepSubnets": {"subnets": self.subnets}}

    def __call__(self, subnets: list[int]) -> dict:
        """Allows calling the descriptor with subnets to create a new instance and return dict."""
        return KeepSubnetsDescriptor(subnets).to_dict()

    def __get__(self, instance, owner) -> Callable[[list[int]], dict]:
        """Descriptor protocol - returns a callable that creates the dict."""

        def create(subnets: list[int]) -> dict:
            return KeepSubnetsDescriptor(subnets).to_dict()

        return create


class RootClaimType(str, Enum):
    """
    Enumeration of root claim types in the Bittensor network.

    This enum defines how coldkeys manage their root alpha emissions:
    - Swap: Swap any alpha emission for TAO
    - Keep: Keep all alpha emission
    - KeepSubnets: Keep alpha emission for specified subnets, swap everything else

    The values match exactly with the RootClaimTypeEnum defined in the Subtensor runtime.
    """

    Swap = "Swap"
    Keep = "Keep"
    KeepSubnets = KeepSubnetsDescriptor

    @classmethod
    def normalize(
        cls, value: "Literal['Swap', 'Keep'] | RootClaimType | dict"
    ) -> str | dict:
        """
        Normalizes a root claim type to a format suitable for Substrate calls.

        This method handles various input formats:
        - String values ("Swap", "Keep") → returns string
        - Enum values (RootClaimType.Swap) → returns string
        - Dict values ({"KeepSubnets": {"subnets": [1, 2, 3]}}) → returns dict as-is
        - Callable KeepSubnets([1, 2, 3]) → returns dict

        Parameters:
            value: The root claim type in any supported format.

        Returns:
            Normalized value - string for Swap/Keep or dict for KeepSubnets.

        Raises:
            ValueError: If the value is not a valid root claim type or KeepSubnets has no subnets.
            TypeError: If the value type is not supported.
        """
        # Handle KeepSubnetsDescriptor instance
        if isinstance(value, KeepSubnetsDescriptor):
            return value.to_dict()

        # Handle enum instance
        if isinstance(value, RootClaimType):
            # If it's KeepSubnets, it's actually the descriptor, so this shouldn't happen
            # But if someone accesses it directly, we need to handle it
            if value == "KeepSubnets":
                raise ValueError(
                    "KeepSubnets must be called with subnet list: RootClaimType.KeepSubnets([1, 2, 3])"
                )
            return value.value

        # Handle string values
        if isinstance(value, str):
            if value in ("Swap", "Keep"):
                return value
            elif value == "KeepSubnets":
                raise ValueError(
                    "KeepSubnets must be provided as dict or called: RootClaimType.KeepSubnets([1, 2, 3])"
                )
            else:
                raise ValueError(
                    f"Invalid root claim type: {value}. "
                    f"Valid types are: 'Swap', 'Keep', or KeepSubnets dict/callable"
                )

        # Handle dict values (for KeepSubnets)
        if isinstance(value, dict):
            if "KeepSubnets" in value and isinstance(value["KeepSubnets"], dict):
                subnets = value["KeepSubnets"].get("subnets", [])
                if not subnets:
                    raise ValueError("KeepSubnets must have at least one subnet")
                if not all(isinstance(netuid, int) for netuid in subnets):
                    raise ValueError("All subnet IDs must be integers")
                return value
            else:
                raise ValueError(
                    f"Invalid dict format for root claim type. "
                    f"Expected {{'KeepSubnets': {{'subnets': [1, 2, 3]}}}}, got {value}"
                )

        raise TypeError(
            f"root_claim_type must be str, RootClaimType, or dict, got {type(value).__name__}"
        )
