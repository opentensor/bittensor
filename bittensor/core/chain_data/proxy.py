from bittensor.core.chain_data.utils import decode_account_id
from dataclasses import dataclass
from typing import Any, Optional
from bittensor.utils.balance import Balance


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
