import bt_decode
from dataclasses import dataclass
from typing import Optional
from bittensor.core.chain_data.utils import decode_account_id
from bittensor.utils import u16_normalized_float
from bittensor.utils.balance import Balance

@dataclass
class DelegateInfo:
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
    nominators: list[tuple[str, Balance]]  # List of nominators of the delegate and their stake
    owner_ss58: str  # Coldkey of owner
    take: float  # Take of the delegate as a percentage
    validator_permits: list[int]  # List of subnets that the delegate is allowed to validate on
    registrations: list[int]  # list of subnets that the delegate is registered on
    return_per_1000: Balance  # Return per 1000 tao of the delegate over a day
    total_daily_return: Balance  # Total daily return of the delegate

    @classmethod
    def from_vec_u8(cls, vec_u8: bytes) -> Optional["DelegateInfo"]:
        decoded = bt_decode.DelegateInfo.decode(vec_u8)

        hotkey = decode_account_id(decoded.delegate_ss58)
        owner = decode_account_id(decoded.owner_ss58)

        # Convert nominators to a list of (str, Balance)
        nominators = [
            (decode_account_id(x), Balance.from_rao(y)) 
            for x, y in decoded.nominators
        ]

        total_stake = sum((x[1] for x in nominators)) if nominators else Balance(0)

        # IMPORTANT: Decide if these are RAO-based or already in final units.
        # If your chain data is NOT in raw RAO, remove `from_rao`:
        return_per_1000_val = Balance.from_rao(decoded.return_per_1000)  
        total_daily_return_val = Balance.from_rao(decoded.total_daily_return)

        # Convert registrations to int if needed
        registrations_val = [int(r) for r in decoded.registrations]

        return DelegateInfo(
            hotkey_ss58=hotkey,
            total_stake=total_stake,
            nominators=nominators,
            owner_ss58=owner,
            take=u16_normalized_float(decoded.take),
            validator_permits=decoded.validator_permits,
            registrations=registrations_val,
            return_per_1000=return_per_1000_val,
            total_daily_return=total_daily_return_val,
        )

    @classmethod
    def list_from_vec_u8(cls, vec_u8: bytes) -> list["DelegateInfo"]:
        decoded_list = bt_decode.DelegateInfo.decode_vec(vec_u8)
        results = []

        for d in decoded_list:
            hotkey = decode_account_id(d.delegate_ss58)
            owner = decode_account_id(d.owner_ss58)

            nominators = [
                (decode_account_id(x), Balance.from_rao(y)) 
                for x, y in d.nominators
            ]
            total_stake = sum((x[1] for x in nominators)) if nominators else Balance(0)

            registrations_val = [int(r) for r in d.registrations]

            return_per_1000_val = Balance.from_rao(d.return_per_1000)  
            total_daily_return_val = Balance.from_rao(d.total_daily_return)

            results.append(
                DelegateInfo(
                    hotkey_ss58=hotkey,
                    total_stake=total_stake,
                    nominators=nominators,
                    owner_ss58=owner,
                    take=u16_normalized_float(d.take),
                    validator_permits=d.validator_permits,
                    registrations=registrations_val,
                    return_per_1000=return_per_1000_val,
                    total_daily_return=total_daily_return_val,
                )
            )

        return results

    @classmethod
    def delegated_list_from_vec_u8(
        cls, vec_u8: bytes
    ) -> list[tuple["DelegateInfo", Balance]]:
        decoded = bt_decode.DelegateInfo.decode_delegated(vec_u8)
        results = []

        for d, b in decoded:
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
