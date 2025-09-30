from dataclasses import dataclass
from typing import Optional

from bittensor.core.chain_data.info_base import InfoBase
from bittensor.core.chain_data.utils import decode_account_id
from bittensor.utils import u16_normalized_float
from bittensor.utils.balance import Balance


@dataclass
class DelegateInfoBase(InfoBase):
    """Base class containing common delegate information fields.

    Attributes:
        hotkey_ss58: Hotkey of delegate.
        owner_ss58: Coldkey of owner.
        take: Take of the delegate as a percentage.
        validator_permits: List of subnets that the delegate is allowed to validate on.
        registrations: List of subnets that the delegate is registered on.
        return_per_1000: Return per 1000 tao of the delegate over a day.
    """

    hotkey_ss58: str  # Hotkey of delegate
    owner_ss58: str  # Coldkey of owner
    take: float  # Take of the delegate as a percentage
    validator_permits: list[
        int
    ]  # List of subnets that the delegate is allowed to validate on
    registrations: list[int]  # list of subnets that the delegate is registered on
    return_per_1000: Balance  # Return per 1000 tao of the delegate over a day


@dataclass
class DelegateInfo(DelegateInfoBase):
    """
    Dataclass for delegate information.

    Additional Attributes:
        total_stake: Total stake of the delegate mapped by netuid.
        nominators: Mapping of nominator SS58 addresses to their stakes per subnet.
    """

    total_stake: dict[int, Balance]  # Total stake of the delegate by netuid and stake
    nominators: dict[
        str, dict[int, Balance]
    ]  # Mapping of nominator addresses to their stakes per subnet

    @classmethod
    def _from_dict(cls, decoded: dict) -> Optional["DelegateInfo"]:
        hotkey = decode_account_id(decoded.get("delegate_ss58"))
        owner = decode_account_id(decoded.get("owner_ss58"))

        nominators = {}
        total_stake_by_netuid = {}

        for raw_nominator, raw_stakes in decoded.get("nominators", []):
            nominator_ss58 = decode_account_id(raw_nominator)
            stakes = {
                int(netuid): Balance.from_rao(stake_amt).set_unit(int(netuid))
                for (netuid, stake_amt) in raw_stakes
            }
            nominators[nominator_ss58] = stakes

            for netuid, stake in stakes.items():
                if netuid not in total_stake_by_netuid:
                    total_stake_by_netuid[netuid] = Balance(0).set_unit(netuid)
                total_stake_by_netuid[netuid] += stake

        return cls(
            hotkey_ss58=hotkey,
            total_stake=total_stake_by_netuid,
            nominators=nominators,
            owner_ss58=owner,
            take=u16_normalized_float(decoded.get("take")),
            validator_permits=list(decoded.get("validator_permits", [])),
            registrations=list(decoded.get("registrations", [])),
            return_per_1000=Balance.from_rao(decoded.get("return_per_1000")),
        )


@dataclass
class DelegatedInfo(DelegateInfoBase):
    """
    Dataclass for delegated information. This class represents a delegate's information specific to a particular subnet.

    Additional Attributes:
        netuid: Network ID of the subnet.
        stake: Stake amount for this specific delegation.
    """

    netuid: int
    stake: Balance

    @classmethod
    def _from_dict(
        cls, decoded: tuple[dict, tuple[int, int]]
    ) -> Optional["DelegatedInfo"]:
        delegate_info, (netuid, stake) = decoded
        hotkey = decode_account_id(delegate_info.get("delegate_ss58"))
        owner = decode_account_id(delegate_info.get("owner_ss58"))
        return cls(
            hotkey_ss58=hotkey,
            owner_ss58=owner,
            take=u16_normalized_float(delegate_info.get("take")),
            validator_permits=list(delegate_info.get("validator_permits", [])),
            registrations=list(delegate_info.get("registrations", [])),
            return_per_1000=Balance.from_rao(delegate_info.get("return_per_1000")),
            netuid=int(netuid),
            stake=Balance.from_rao(int(stake)).set_unit(int(netuid)),
        )
