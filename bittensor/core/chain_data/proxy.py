from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

from bittensor.core.chain_data.utils import decode_account_id
from bittensor.utils.balance import Balance


class ProxyType(str, Enum):
    """
    Enumeration of all supported proxy types in the Bittensor network.

    These types define the permissions that a proxy account has when acting on behalf of the real account. Each type
    restricts what operations the proxy can perform.

    Note:
        The values match exactly with the ProxyType enum defined in the Subtensor runtime. Any changes to the runtime
        enum must be reflected here.
    """

    any = "Any"
    Owner = "Owner"
    NonCritical = "NonCritical"
    NonTransfer = "NonTransfer"
    Senate = "Senate"
    NonFungible = "NonFungible"
    Triumvirate = "Triumvirate"
    Governance = "Governance"
    Staking = "Staking"
    Registration = "Registration"
    Transfer = "Transfer"
    SmallTransfer = "SmallTransfer"
    RootWeights = "RootWeights"
    ChildKeys = "ChildKeys"
    SudoUncheckedSetCode = "SudoUncheckedSetCode"
    SwapHotkey = "SwapHotkey"
    SubnetLeaseBeneficiary = "SubnetLeaseBeneficiary"
    RootClaim = "RootClaim"

    @classmethod
    def all_types(cls) -> list[str]:
        """Returns a list of all proxy type values."""
        return [member.value for member in cls]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Checks if a string value is a valid proxy type."""
        return value in cls.all_types()

    @classmethod
    def normalize(cls, proxy_type: Union[str, "ProxyType"]) -> str:
        """
        Normalizes a proxy type to a string value.

        This method handles both string and ProxyType enum values, validates the input, and returns the string
        representation suitable for Substrate calls.

        Parameters:
            proxy_type: Either a string or ProxyType enum value.

        Returns:
            str: The normalized string value of the proxy type.

        Raises:
            ValueError: If the proxy_type is not a valid proxy type.
        """
        if isinstance(proxy_type, ProxyType):
            return proxy_type.value
        elif isinstance(proxy_type, str):
            if not cls.is_valid(proxy_type):
                raise ValueError(
                    f"Invalid proxy type: {proxy_type}. "
                    f"Valid types are: {', '.join(cls.all_types())}"
                )
            return proxy_type
        else:
            raise TypeError(
                f"proxy_type must be str or ProxyType, got {type(proxy_type).__name__}"
            )


@dataclass
class ProxyInfo:
    delegate: str
    proxy_type: str
    delay: int

    @classmethod
    def from_dict(cls, data: dict):
        """Returns a ProxyInfo object from proxy data."""
        return cls(
            delegate=decode_account_id(data["delegate"]),
            proxy_type=data["proxy_type"],
            delay=data["delay"],
        )

    @classmethod
    def from_tuple(cls, data: tuple):
        """Returns a list of ProxyInfo objects from a tuple of proxy data."""
        return [
            cls(
                delegate=decode_account_id(proxy["delegate"]),
                proxy_type=proxy["proxy_type"],
                delay=proxy["delay"],
            )
            for proxy in data
        ]

    @classmethod
    def from_query(cls, query: Any):
        """Returns a ProxyInfo object from a Substrate query."""
        try:
            proxies = query.value[0][0]
            balance = query.value[1]
            return cls.from_tuple(proxies), Balance.from_rao(balance)
        except IndexError:
            return [], Balance.from_rao(0)


@dataclass
class ProxyConstants:
    """
    Represents all runtime constants defined in the `Proxy` pallet.

    Attributes:


    Note:
        All Balance amounts are in RAO.
    """

    AnnouncementDepositBase: Optional[Balance]
    AnnouncementDepositFactor: Optional[Balance]
    MaxProxies: Optional[int]
    MaxPending: Optional[int]
    ProxyDepositBase: Optional[Balance]
    ProxyDepositFactor: Optional[Balance]

    @classmethod
    def constants_names(cls) -> list[str]:
        """Returns the list of all constant field names defined in this dataclass."""
        from dataclasses import fields

        return [f.name for f in fields(cls)]

    @classmethod
    def from_dict(cls, data: dict) -> "ProxyConstants":
        """
        Creates a `ProxyConstants` instance from a dictionary of decoded chain constants.

        Parameters:
            data: Dictionary mapping constant names to their decoded values (returned by `Subtensor.query_constant()`).

        Returns:
            ProxyConstants: The structured dataclass with constants filled in.
        """
        return cls(**{name: data.get(name) for name in cls.constants_names()})

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)
