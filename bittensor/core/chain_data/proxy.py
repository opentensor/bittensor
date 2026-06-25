from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence, Union

from bittensor.utils.balance import Balance


class ProxyType(str, Enum):
    """Enumeration of all supported proxy types in the Bittensor network.

    These types define the permissions that a proxy account has when acting on behalf of the real account. Each type
    restricts what operations the proxy can perform.

    Proxy Types:
        Any: Full permissions — allows all calls. Use with extreme caution.
        ChildKeys: Only child key management operations.
        NonCritical: All operations except critical/destructive ones.
        NonTransfer: All operations except token transfers.
        NonFungible: All operations except token/staking/registration/key-swap operations.
        Owner: Subnet identity and settings management.
        Registration: Only neuron registration operations.
        RootClaim: Only root claim operations.
        SmallTransfer: Only token transfers below the on-chain limit.
        Staking: Only staking-related operations.
        SubnetLeaseBeneficiary: Subnet management for lease beneficiaries.
        SudoUncheckedSetCode: Only runtime code updates via sudo.
        SwapHotkey: Only hotkey swap operations.
        Transfer: Only token transfer operations.

    Notes:
        - To retrieve the exact, up-to-date filter rules (which extrinsics each type permits or denies), use
          :meth:`~bittensor.core.async_/subtensor.Async/Subtensor.get_proxy_filter`.
        - Proxy overview: <https://docs.learnbittensor.org/keys/proxies>
        - Creating and managing proxies: <https://docs.learnbittensor.org/keys/proxies/create-proxy>
        - Pure proxies: <https://docs.learnbittensor.org/keys/proxies/pure-proxies>
    """

    Any = "Any"
    Owner = "Owner"
    NonCritical = "NonCritical"
    NonTransfer = "NonTransfer"
    NonFungible = "NonFungible"
    Staking = "Staking"
    Registration = "Registration"
    Transfer = "Transfer"
    SmallTransfer = "SmallTransfer"
    ChildKeys = "ChildKeys"
    SudoUncheckedSetCode = "SudoUncheckedSetCode"
    SwapHotkey = "SwapHotkey"
    SubnetLeaseBeneficiary = "SubnetLeaseBeneficiary"
    RootClaim = "RootClaim"

    # deprecated proxy types
    Triumvirate = "Triumvirate"
    Governance = "Governance"
    Senate = "Senate"
    RootWeights = "RootWeights"

    @classmethod
    def all_types(cls) -> list[str]:
        """Returns a list of all proxy type values.

        Returns:
            List of all valid proxy type string values (e.g., `["Any", "Owner", "Staking", ...]`).
        """
        return [member.value for member in cls]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Checks if a string value is a valid proxy type.

        Parameters:
            value: String value to validate.

        Returns:
            `True` if the value is a valid proxy type, `False` otherwise.
        """
        return value in cls.all_types()

    @classmethod
    def normalize(cls, proxy_type: Union[str, "ProxyType"]) -> str:
        """Normalizes a proxy type to a string value.

        This method handles both string and ProxyType enum values, validates the input, and returns the string
        representation suitable for Substrate calls.

        Parameters:
            proxy_type: Either a string or ProxyType enum value.

        Returns:
            The normalized string value of the proxy type.

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
    """Dataclass representing proxy relationship information.

    This class contains information about a proxy relationship between a real account and a delegate account. A proxy
    relationship allows the delegate to perform certain operations on behalf of the real account, with restrictions
    defined by the proxy type and a delay period.

    Attributes:
        delegate: The SS58 address of the delegate proxy account that can act on behalf of the real account.
        proxy_type: The type of proxy permissions granted to the delegate (e.g., `"Any"`, `"NonTransfer"`,
            `"ChildKeys"`, `"Staking"`). This determines what operations the delegate can perform.
        delay: The number of blocks that must elapse between announcing a call and executing it (time-lock period). A
            delay of `0` allows immediate execution without announcements. Non-zero delays require the delegate to
            announce the call first via `announce_proxy`, wait for the delay period to pass, then execute it via
            `proxy_announced`, giving the real account time to review and potentially reject the call via
            `reject_proxy_announcement` before execution.

    Notes:
        - Bittensor proxies: <https://docs.learnbittensor.org/keys/proxies>
        - Creating proxies: <https://docs.learnbittensor.org/keys/proxies/create-proxy>
    """

    delegate: str
    proxy_type: str
    delay: int

    @classmethod
    def from_tuple(cls, data: Sequence[dict[str, str | int]]) -> list["ProxyInfo"]:
        """Creates a list of ProxyInfo objects from chain proxy data.

        This method decodes the raw proxy data returned from the Proxy.Proxies storage function and creates
        structured ProxyInfo objects.

        Parameters:
            data: Tuple of chain proxy data from the `Proxy.Proxies` storage function.

        Returns:
            List of ProxyInfo objects representing all proxy relationships for a real account.

        Notes:
            See: <https://docs.learnbittensor.org/keys/proxies>
        """
        return [
            cls(
                delegate=proxy["delegate"],
                proxy_type=proxy["proxy_type"],
                delay=proxy["delay"],
            )
            for proxy in data
        ]

    @classmethod
    def from_query(cls, query: Any) -> tuple[list["ProxyInfo"], Balance]:
        """Creates a list of ProxyInfo objects and deposit balance from a Substrate query result.

        This method decodes the query result from the Proxy.Proxies storage function, extracting both the proxy
        relationships and the deposit amount reserved for maintaining these proxies.

        Parameters:
            query: Query result from Substrate `query()` call to `Proxy.Proxies` storage function.

        Returns:
            Tuple containing:

                - List of ProxyInfo objects representing all proxy relationships for the real account.
                - Balance object representing the reserved deposit amount (in RAO).

        Notes:
            The deposit is held as long as the proxy relationships exist and is returned when proxies are removed.

            See: <https://docs.learnbittensor.org/keys/proxies>
        """
        # proxies data is always in that path
        proxies = query.value[0]
        # balance data is always in that path
        balance = query.value[1]
        return cls.from_tuple(proxies), Balance.from_rao(balance)

    @classmethod
    def from_query_map_record(
        cls, record: tuple[str, tuple[list[dict[str, str | int]], int]]
    ) -> tuple[str, list["ProxyInfo"]]:
        """Creates a dictionary mapping delegate addresses to their ProxyInfo lists from a query_map record.

        Processes a single record from a query_map call to the Proxy.Proxies storage function. Each record represents
        one real account and its associated proxy/ies relationships.

        Parameters:
            record: Data item from query_map records call to Proxies storage function. Structure is [key, value] where
                key is the real account and value contains proxies data.

        Returns:
            Tuple containing:
                - SS58 address of the real account (delegator).
                - List of ProxyInfo objects representing all proxy relationships for this real account.
        """
        # record[0] is the real account (key from storage)
        # record[1] is the value containing proxies data
        real_account_ss58 = record[0]
        # list with proxies data is always in that path
        proxy_data = cls.from_tuple(record[1][0])
        return real_account_ss58, proxy_data


@dataclass
class ProxyAnnouncementInfo:
    """Dataclass representing proxy announcement information.

    This class contains information about a pending proxy announcement. Announcements are used when a proxy account
    with a non-zero delay period (time-lock) wants to declare its intention to execute a call on behalf of the real
    account. The announcement must be made before the actual call can be executed, allowing the real account time to
    review and potentially reject the operation via `reject_proxy_announcement` before it takes effect. After the
    delay period passes, the proxy can execute the announced call via `proxy_announced`.

    Attributes:
        real: The SS58 address of the real account on whose behalf the call will be made.
        call_hash: The hash of the call that will be executed in the future (hex string with `0x` prefix). This hash
            must match the actual call when it is executed via `proxy_announced`.
        height: The block height at which the announcement was made. The delay period is calculated from this block.

    Notes:
        - Announcements are required when using delayed proxies (non-zero delay), providing an additional security
          layer for time-locked operations.
        - Bittensor proxies: <https://docs.learnbittensor.org/keys/proxies>
    """

    real: str
    call_hash: str
    height: int

    @classmethod
    def from_dict(
        cls, data: tuple[list[dict[str, str | int]], int]
    ) -> list["ProxyAnnouncementInfo"]:
        """Creates a list of ProxyAnnouncementInfo objects from chain announcement data.

        This method decodes the raw announcement data returned from the Proxy.Announcements storage function.

        Parameters:
            data: Tuple of announcements data from the `Proxy.Announcements` storage function.

        Returns:
            List of ProxyAnnouncementInfo objects representing all pending announcements.

        Notes:
            See: <https://docs.learnbittensor.org/keys/proxies>
        """
        return [
            cls(
                real=annt["real"],
                call_hash=annt["call_hash"],
                height=annt["height"],
            )
            for annt in data[0]
        ]

    @classmethod
    def from_query_map_record(
        cls, record: tuple[str, tuple[list[dict[str, str | int]], int]]
    ) -> tuple[str, list["ProxyAnnouncementInfo"]]:
        """Returns a list of ProxyAnnouncementInfo objects from a tuple of announcements data.

        Parameters:
            record: Data item from query_map records call to Announcements storage function. Structure is [key, value]
                where key is the delegate account and value contains announcements data.

        Returns:
            Tuple containing:
                - SS58 address of the delegate account making the announcement.
                - List of ProxyAnnouncementInfo objects for all pending announcements from this delegate.
        """
        # record[0] is the real account (key from storage)
        # record[1] is the value containing announcements data
        delegate = record[0]
        # list with proxies data is always in that path
        announcements_data = cls.from_dict(record[1])
        return delegate, announcements_data


@dataclass
class ProxyConstants:
    """Fetches all runtime constants defined in the Proxy pallet.

    Displays current values for on-chain configuration constants for the Proxy pallet. They define
    deposit requirements, account limits, and announcement constraints that govern the behavior of proxies.

    Each attribute is fetched directly from the runtime via `Subtensor.query_constant("Proxy", <name>)` and reflects
    the current chain configuration at the time of retrieval.

    Attributes:
        AnnouncementDepositBase: Base deposit amount (in RAO) required to announce a future proxy call. This deposit
            is held until the announced call is executed or cancelled.
        AnnouncementDepositFactor: Additional deposit factor (in RAO) per byte of the call hash being announced. The
            total announcement deposit is calculated as: `AnnouncementDepositBase + (call_hash_size *
            AnnouncementDepositFactor)`.
        MaxProxies: Maximum number of proxy relationships that a single account can have. This limits the total
            number of delegates that can act on behalf of an account.
        MaxPending: Maximum number of pending proxy announcements that can exist for a single account at any given
            time. This prevents spam and limits the storage requirements for pending announcements.
        ProxyDepositBase: Base deposit amount (in RAO) required when adding a proxy relationship. This deposit is
            held as long as the proxy relationship exists and is returned when the proxy is removed.
        ProxyDepositFactor: Additional deposit factor (in RAO) per proxy type added. The total proxy deposit is
            calculated as: `ProxyDepositBase + (number_of_proxy_types * ProxyDepositFactor)`.

    Notes:
        - All Balance amounts are in RAO.
        - See: <https://docs.learnbittensor.org/keys/proxies>
    """

    AnnouncementDepositBase: Optional[Balance]
    AnnouncementDepositFactor: Optional[Balance]
    MaxProxies: Optional[int]
    MaxPending: Optional[int]
    ProxyDepositBase: Optional[Balance]
    ProxyDepositFactor: Optional[Balance]

    @classmethod
    def constants_names(cls) -> list[str]:
        """Returns the all constant field names defined in this dataclass.

        Returns:
            List of constant field names as strings.
        """
        from dataclasses import fields

        return [f.name for f in fields(cls)]

    @classmethod
    def from_dict(cls, data: dict) -> "ProxyConstants":
        """Creates a ProxyConstants instance from a dictionary of decoded chain constants.

        Parameters:
            data: Dictionary mapping constant names to their decoded values (returned by `Subtensor.query_constant()`).

        Returns:
            ProxyConstants object with constants filled in. Fields not found in data will be set to `None`.
        """
        return cls(**{name: data.get(name) for name in cls.constants_names()})

    def to_dict(self) -> dict:
        """Converts the ProxyConstants instance to a dictionary.

        Returns:
            Dictionary mapping constant names to their values. Balance objects remain as Balance instances.
        """
        from dataclasses import asdict

        return asdict(self)


@dataclass
class ProxyTypeInfo:
    """Runtime information about a proxy type variant.

    This data is returned by the ``ProxyFilterRuntimeApi.getProxyTypes`` runtime API and represents the authoritative
    source of truth for which proxy types exist in the current runtime.

    Attributes:
        name: The proxy type name (e.g., ``"Staking"``, ``"NonTransfer"``).
        index: The numeric index of this proxy type in the runtime enum.
        deprecated: Whether this proxy type is deprecated and no longer functional.

    Notes:
        - See: <https://docs.learnbittensor.org/keys/proxies>
    """

    name: str
    index: int
    deprecated: bool

    @classmethod
    def from_list(cls, data: list[dict]) -> list["ProxyTypeInfo"]:
        """Creates a list of ProxyTypeInfo from the ``ProxyFilterRuntimeApi.getProxyTypes`` runtime API response.

        Parameters:
            data: List of dictionaries from the runtime API response.

        Returns:
            List of ProxyTypeInfo objects.
        """
        return [
            cls(name=item["name"], index=item["index"], deprecated=item["deprecated"])
            for item in data
        ]


@dataclass
class ProxyFilterInfo:
    """Describes how a specific proxy type filters incoming runtime calls.

    This data is returned by the ``ProxyFilterRuntimeApi.getProxyFilter`` runtime API and represents the authoritative
    source of truth for proxy permissions. It describes which extrinsics each proxy type is allowed or denied to
    execute on behalf of the real account.

    Attributes:
        proxy_type: The numeric index of the proxy type.
        name: Human-readable name of the proxy type (e.g., ``"Staking"``, ``"NonTransfer"``).
        deprecated: Whether this proxy type is deprecated.
        filter_mode: How filtering works. One of:

            - ``"AllowAll"``: All calls are permitted (e.g., ``ProxyType.Any``).
            - ``"DenyAll"``: No calls are permitted (e.g., deprecated types).
            - ``"Allow"``: Only calls listed in ``calls`` are permitted (minus ``exceptions``).
            - ``"Deny"``: All calls are permitted EXCEPT those listed in ``calls``.
        calls: List of call descriptors that the filter applies to. Each is a dict with keys: ``pallet_name``,
            ``pallet_index``, ``call_name`` (``None`` means all calls in the pallet), ``call_index``, ``condition``
            (``None`` or a dict describing the condition — e.g., ``{"ParamLessThan": {"param_name": ..., "limit": ...}}``
            or ``{"NestedCallMustBe": {"pallet_name": ..., "call_name": ...}}``).
        exceptions: List of call descriptors excluded from the filter rule (same structure as ``calls``).

    Notes:
        - See: <https://docs.learnbittensor.org/keys/proxies>
    """

    proxy_type: int
    name: str
    deprecated: bool
    filter_mode: str
    calls: list[dict]
    exceptions: list[dict]

    @classmethod
    def from_list(cls, data: list[dict]) -> list["ProxyFilterInfo"]:
        """Creates a list of ProxyFilterInfo from the ``ProxyFilterRuntimeApi.getProxyFilter`` runtime API response.

        Parameters:
            data: List of dictionaries from the runtime API response.

        Returns:
            List of ProxyFilterInfo objects.
        """
        return [
            cls(
                proxy_type=item["proxy_type"],
                name=item["name"],
                deprecated=item["deprecated"],
                filter_mode=item["filter_mode"],
                calls=item.get("calls", []),
                exceptions=item.get("exceptions", []),
            )
            for item in data
        ]
