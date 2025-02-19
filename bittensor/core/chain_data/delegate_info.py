from dataclasses import dataclass
from typing import Any, Optional, Union

from bittensor.core.chain_data.info_base import InfoBase
from bittensor.core.chain_data.utils import decode_account_id
from bittensor.utils import u16_normalized_float
from bittensor.utils.balance import Balance


@dataclass
class DelegateInfo(InfoBase):
    """
    Dataclass for delegate information. For a lighter version of this class, see ``DelegateInfoLite``.

    Args:
        hotkey_ss58 (str): Hotkey of the delegate for which the information is being fetched.
        total_stake (int): Total stake of the delegate.
        nominators (dict[str, dict[int, Balance]]): List of nominators of the delegate and their stake.
        take (float): Take of the delegate as a percentage.
        owner_ss58 (str): Coldkey of the owner.
        registrations (list[int]): List of subnets that the delegate is registered on.
        validator_permits (list[int]): List of subnets that the delegate is allowed to validate on.
        return_per_1000 (int): Return per 1000 TAO, for the delegate over a day.
        total_daily_return (int): Total daily return of the delegate.
        netuid (int): Netuid of the subnet.
    """

    hotkey_ss58: str  # Hotkey of delegate
    total_stake: Balance  # Total stake of the delegate
    nominators: dict[
        str, dict[int, Balance]
    ]  # list of nominators of the delegate and their stake
    owner_ss58: str  # Coldkey of owner
    take: float  # Take of the delegate as a percentage
    validator_permits: list[
        int
    ]  # List of subnets that the delegate is allowed to validate on
    registrations: list[int]  # list of subnets that the delegate is registered on
    return_per_1000: Balance  # Return per 1000 tao of the delegate over a day
    total_daily_return: Balance  # Total daily return of the delegate
    netuid: Optional[int] = None

    @classmethod
    def _from_dict(cls, decoded: Union[dict, tuple]) -> "DelegateInfo":
        hotkey = decode_account_id(decoded.get("delegate_ss58"))
        owner = decode_account_id(decoded.get("owner_ss58"))

        nominators = {}
        total_stake_by_netuid = {}
        for nominator in decoded.get("nominators", []):
            nominator_ss58 = decode_account_id(nominator[0])
            stakes = {
                int(netuid): Balance.from_rao(stake).set_unit(netuid)
                for netuid, stake in nominator[1]
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
            total_daily_return=Balance.from_rao(decoded.get("total_daily_return")),
        )

    @classmethod
    def delegated_list_from_dicts(
        cls, delegates_decoded: list[Any]
    ) -> list["DelegateInfo"]:
        all_delegates = []
        for delegate, (netuid, stake) in delegates_decoded:
            instance = DelegateInfo.from_dict(delegate)
            instance.netuid = int(netuid)
            instance.total_stake = Balance.from_rao(int(stake)).set_unit(int(netuid))
            all_delegates.append(instance)
        return all_delegates
