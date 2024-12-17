from dataclasses import dataclass
from typing import Any, Optional

import bt_decode
import munch

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
        nominators (list[tuple[str, int]]): List of nominators of the delegate and their stake.
        take (float): Take of the delegate as a percentage.
        owner_ss58 (str): Coldkey of the owner.
        registrations (list[int]): List of subnets that the delegate is registered on.
        validator_permits (list[int]): List of subnets that the delegate is allowed to validate on.
        return_per_1000 (int): Return per 1000 TAO, for the delegate over a day.
        total_daily_return (int): Total daily return of the delegate.
    """

    hotkey_ss58: str  # Hotkey of delegate
    total_stake: Balance  # Total stake of the delegate
    nominators: list[
        tuple[str, Balance]
    ]  # List of nominators of the delegate and their stake
    owner_ss58: str  # Coldkey of owner
    take: float  # Take of the delegate as a percentage
    validator_permits: list[
        int
    ]  # List of subnets that the delegate is allowed to validate on
    registrations: list[int]  # list of subnets that the delegate is registered on
    return_per_1000: Balance  # Return per 1000 tao of the delegate over a day
    total_daily_return: Balance  # Total daily return of the delegate

    @classmethod
    def _fix_decoded(cls, decoded: "DelegateInfo") -> Optional["DelegateInfo"]:
        hotkey = decode_account_id(decoded.delegate_ss58)
        owner = decode_account_id(decoded.owner_ss58)
        nominators = [
            (decode_account_id(x), Balance.from_rao(y)) for x, y in decoded.nominators
        ]
        total_stake = sum((x[1] for x in nominators)) if nominators else Balance(0)
        return DelegateInfo(
            hotkey_ss58=hotkey,
            total_stake=total_stake,
            nominators=nominators,
            owner_ss58=owner,
            take=u16_normalized_float(decoded.take),
            validator_permits=decoded.validator_permits,
            registrations=decoded.registrations,
            return_per_1000=Balance.from_rao(decoded.return_per_1000),
            total_daily_return=Balance.from_rao(decoded.total_daily_return),
        )

    @classmethod
    def list_from_vec_u8(cls, vec_u8: bytes) -> list["DelegateInfo"]:
        decoded = bt_decode.DelegateInfo.decode_vec(vec_u8)
        return [cls._fix_decoded(d) for d in decoded]

    @classmethod
    def fix_delegated_list(
        cls, delegated_list: list[tuple["DelegateInfo", Balance]]
    ) -> list[tuple["DelegateInfo", Balance]]:
        results = []
        for d, b in delegated_list:
            nominators = [
                (decode_account_id(x), Balance.from_rao(y)) for x, y in d.nominators
            ]
            total_stake = sum((x[1] for x in nominators)) if nominators else Balance(0)
            delegate = DelegateInfo(
                hotkey_ss58=decode_account_id(d.delegate_ss58),
                total_stake=total_stake,
                nominators=nominators,
                owner_ss58=decode_account_id(d.owner_ss58),
                take=u16_normalized_float(d.take),
                validator_permits=d.validator_permits,
                registrations=d.registrations,
                return_per_1000=Balance.from_rao(d.return_per_1000),
                total_daily_return=Balance.from_rao(d.total_daily_return),
            )
            results.append((delegate, Balance.from_rao(b)))
        return results

    @classmethod
    def delegated_list_from_vec_u8(
        cls, vec_u8: bytes
    ) -> list[tuple["DelegateInfo", Balance]]:
        decoded = bt_decode.DelegateInfo.decode_delegated(vec_u8)
        return cls.fix_delegated_list(decoded)

    @classmethod
    def delegated_list_from_any(
        cls, any_list: list[Any]
    ) -> list[tuple["DelegateInfo", Balance]]:
        any_list = [munch.munchify(any_) for any_ in any_list]
        return cls.fix_delegated_list(any_list)
