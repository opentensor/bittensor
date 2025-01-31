from dataclasses import dataclass
from typing import Optional

from bittensor.core.chain_data.axon_info import AxonInfo
from bittensor.core.chain_data.chain_identity import ChainIdentity
from bittensor.core.chain_data.subnet_identity import SubnetIdentity
from bittensor.core.chain_data.utils import (
    ChainDataType,
    from_scale_encoding,
)
from bittensor.utils import u64_normalized_float as u64tf, u16_normalized_float as u16tf
from bittensor.utils.balance import Balance
from scalecodec.utils.ss58 import ss58_encode


# to balance with unit (just shortcut)
def _tbwu(val: int, netuid: Optional[int] = 0) -> Balance:
    """Returns a Balance object from a value and unit."""
    return Balance.from_rao(val, netuid)


@dataclass
class MetagraphInfo:
    # Subnet index
    netuid: int

    # Name and symbol
    name: str
    symbol: str
    identity: Optional[SubnetIdentity]
    network_registered_at: int

    # Keys for owner.
    owner_hotkey: str  # hotkey
    owner_coldkey: str  # coldkey

    # Tempo terms.
    block: int  # block at call.
    tempo: int  # epoch tempo
    last_step: int
    blocks_since_last_step: int

    # Subnet emission terms
    subnet_emission: Balance  # subnet emission via tao
    alpha_in: Balance  # amount of alpha in reserve
    alpha_out: Balance  # amount of alpha outstanding
    tao_in: Balance  # amount of tao injected per block
    alpha_out_emission: Balance  # amount injected in alpha reserves per block
    alpha_in_emission: Balance  # amount injected outstanding per block
    tao_in_emission: Balance  # amount of tao injected per block
    pending_alpha_emission: Balance  # pending alpha to be distributed
    pending_root_emission: Balance  # pending tao for root divs to be distributed

    # Hparams for epoch
    rho: int  # subnet rho param
    kappa: float  # subnet kappa param

    # Validator params
    min_allowed_weights: float  # min allowed weights per val
    max_weights_limit: float  # max allowed weights per val
    weights_version: int  # allowed weights version
    weights_rate_limit: int  # rate limit on weights.
    activity_cutoff: int  # validator weights cut off period in blocks
    max_validators: int  # max allowed validators.

    # Registration
    num_uids: int
    max_uids: int
    burn: Balance  # current burn cost.
    difficulty: float  # current difficulty.
    registration_allowed: bool  # allows registrations.
    pow_registration_allowed: bool  # pow registration enabled.
    immunity_period: int  # subnet miner immunity period
    min_difficulty: float  # min pow difficulty
    max_difficulty: float  # max pow difficulty
    min_burn: Balance  # min tao burn
    max_burn: Balance  # max tao burn
    adjustment_alpha: float  # adjustment speed for registration params.
    adjustment_interval: int  # pow and burn adjustment interval
    target_regs_per_interval: int  # target registrations per interval
    max_regs_per_block: int  # max registrations per block.
    serving_rate_limit: int  # axon serving rate limit

    # CR
    commit_reveal_weights_enabled: bool  # Is CR enabled.
    commit_reveal_period: int  # Commit reveal interval

    # Bonds
    liquid_alpha_enabled: bool  # Bonds liquid enabled.
    alpha_high: float  # Alpha param high
    alpha_low: float  # Alpha param low
    bonds_moving_avg: float  # Bonds moving avg

    # Metagraph info.
    hotkeys: list[str]  # hotkey per UID
    coldkeys: list[str]  # coldkey per UID
    identities: list[Optional[ChainIdentity]]  # coldkeys identities
    axons: list[AxonInfo]  # UID axons.
    active: list[bool]  # Active per UID
    validator_permit: list[bool]  # Val permit per UID
    pruning_score: list[float]  # Pruning per UID
    last_update: list[int]  # Last update per UID
    emission: list[Balance]  # Emission per UID
    dividends: list[float]  # Dividends per UID
    incentives: list[float]  # Mining incentives per UID
    consensus: list[float]  # Consensus per UID
    trust: list[float]  # Trust per UID
    rank: list[float]  # Rank per UID
    block_at_registration: list[int]  # Reg block per UID
    alpha_stake: list[Balance]  # Alpha staked per UID
    tao_stake: list[Balance]  # TAO staked per UID
    total_stake: list[Balance]  # Total stake per UID

    # Dividend break down.
    tao_dividends_per_hotkey: list[
        tuple[str, Balance]
    ]  # List of dividend payouts in tao via root.
    alpha_dividends_per_hotkey: list[
        tuple[str, Balance]
    ]  # List of dividend payout in alpha via subnet.

    @classmethod
    def from_vec_u8(cls, vec_u8: bytes) -> Optional["MetagraphInfo"]:
        """Returns a Metagraph object from encoded MetagraphInfo vector."""
        if len(vec_u8) == 0:
            return None
        decoded = from_scale_encoding(vec_u8, ChainDataType.MetagraphInfo)
        if decoded is None:
            return None

        return MetagraphInfo.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: bytes) -> list["MetagraphInfo"]:
        """Returns a list of Metagraph objects from a list of encoded MetagraphInfo vectors."""
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.MetagraphInfo, is_vec=True, is_option=True
        )
        if decoded is None:
            return []

        decoded = [
            MetagraphInfo.fix_decoded_values(meta)
            for meta in decoded
            if meta is not None
        ]
        return decoded

    @classmethod
    def fix_decoded_values(cls, decoded: dict) -> "MetagraphInfo":
        """Returns a Metagraph object from a decoded MetagraphInfo dictionary."""
        # Subnet index
        _netuid = decoded["netuid"]

        # Name and symbol
        decoded.update({"name": bytes(decoded.get("name")).decode()})
        decoded.update({"symbol": bytes(decoded.get("symbol")).decode()})
        decoded.update({"identity": decoded.get("identity", {})})

        # Keys for owner.
        decoded["owner_hotkey"] = ss58_encode(decoded["owner_hotkey"])
        decoded["owner_coldkey"] = ss58_encode(decoded["owner_coldkey"])

        # Subnet emission terms
        decoded["subnet_emission"] = _tbwu(decoded["subnet_emission"])
        decoded["alpha_in"] = _tbwu(decoded["alpha_in"], _netuid)
        decoded["alpha_out"] = _tbwu(decoded["alpha_out"], _netuid)
        decoded["tao_in"] = _tbwu(decoded["tao_in"])
        decoded["alpha_out_emission"] = _tbwu(decoded["alpha_out_emission"], _netuid)
        decoded["alpha_in_emission"] = _tbwu(decoded["alpha_in_emission"], _netuid)
        decoded["tao_in_emission"] = _tbwu(decoded["tao_in_emission"])
        decoded["pending_alpha_emission"] = _tbwu(
            decoded["pending_alpha_emission"], _netuid
        )
        decoded["pending_root_emission"] = _tbwu(decoded["pending_root_emission"])

        # Hparams for epoch
        decoded["kappa"] = u16tf(decoded["kappa"])

        # Validator params
        decoded["min_allowed_weights"] = u16tf(decoded["min_allowed_weights"])
        decoded["max_weights_limit"] = u16tf(decoded["max_weights_limit"])

        # Registration
        decoded["burn"] = _tbwu(decoded["burn"])
        decoded["difficulty"] = u64tf(decoded["difficulty"])
        decoded["min_difficulty"] = u64tf(decoded["min_difficulty"])
        decoded["max_difficulty"] = u64tf(decoded["max_difficulty"])
        decoded["min_burn"] = _tbwu(decoded["min_burn"])
        decoded["max_burn"] = _tbwu(decoded["max_burn"])
        decoded["adjustment_alpha"] = u64tf(decoded["adjustment_alpha"])

        # Bonds
        decoded["alpha_high"] = u16tf(decoded["alpha_high"])
        decoded["alpha_low"] = u16tf(decoded["alpha_low"])
        decoded["bonds_moving_avg"] = u64tf(decoded["bonds_moving_avg"])

        # Metagraph info.
        decoded["hotkeys"] = [ss58_encode(ck) for ck in decoded.get("hotkeys", [])]
        decoded["coldkeys"] = [ss58_encode(hk) for hk in decoded.get("coldkeys", [])]
        decoded["axons"] = decoded.get("axons", [])
        decoded["pruning_score"] = [
            u16tf(ps) for ps in decoded.get("pruning_score", [])
        ]
        decoded["emission"] = [_tbwu(em, _netuid) for em in decoded.get("emission", [])]
        decoded["dividends"] = [u16tf(dv) for dv in decoded.get("dividends", [])]
        decoded["incentives"] = [u16tf(ic) for ic in decoded.get("incentives", [])]
        decoded["consensus"] = [u16tf(cs) for cs in decoded.get("consensus", [])]
        decoded["trust"] = [u16tf(tr) for tr in decoded.get("trust", [])]
        decoded["rank"] = [u16tf(rk) for rk in decoded.get("trust", [])]
        decoded["alpha_stake"] = [_tbwu(ast, _netuid) for ast in decoded["alpha_stake"]]
        decoded["tao_stake"] = [_tbwu(ts) for ts in decoded["tao_stake"]]
        decoded["total_stake"] = [_tbwu(ts, _netuid) for ts in decoded["total_stake"]]

        # Dividend break down
        decoded["tao_dividends_per_hotkey"] = [
            (ss58_encode(alpha[0]), _tbwu(alpha[1]))
            for alpha in decoded["tao_dividends_per_hotkey"]
        ]
        decoded["alpha_dividends_per_hotkey"] = [
            (ss58_encode(adphk[0]), _tbwu(adphk[1], _netuid))
            for adphk in decoded["alpha_dividends_per_hotkey"]
        ]

        return MetagraphInfo(**decoded)


@dataclass
class MetagraphInfoEmissions:
    subnet_emission: Balance
    alpha_in_emission: Balance
    alpha_out_emission: Balance
    tao_in_emission: Balance
    pending_alpha_emission: Balance
    pending_root_emission: Balance


@dataclass
class MetagraphInfoPool:
    alpha_out: Balance
    alpha_in: Balance
    tao_in: Balance


@dataclass
class MetagraphInfoParams:
    activity_cutoff: int
    adjustment_alpha: float
    adjustment_interval: int
    alpha_high: float
    alpha_low: float
    bonds_moving_avg: float
    burn: Balance
    commit_reveal_period: int
    commit_reveal_weights_enabled: bool
    difficulty: float
    immunity_period: int
    kappa: float
    liquid_alpha_enabled: bool
    max_burn: Balance
    max_difficulty: float
    max_regs_per_block: int
    max_validators: int
    max_weights_limit: float
    min_allowed_weights: float
    min_burn: Balance
    min_difficulty: float
    pow_registration_allowed: bool
    registration_allowed: bool
    rho: int
    serving_rate_limit: int
    target_regs_per_interval: int
    tempo: int
    weights_rate_limit: int
    weights_version: int
