from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
from bittensor.core import settings
from bittensor.core.chain_data.axon_info import AxonInfo
from bittensor.core.chain_data.chain_identity import ChainIdentity
from bittensor.core.chain_data.info_base import InfoBase
from bittensor.core.chain_data.subnet_identity import SubnetIdentity
from bittensor.core.chain_data.utils import decode_account_id
from bittensor.utils import (
    get_netuid_and_mechid_by_storage_index,
    u64_normalized_float as u64tf,
    u16_normalized_float as u16tf,
)
from bittensor.utils.balance import Balance, fixed_to_float


SELECTIVE_METAGRAPH_COMMITMENTS_OFFSET = 14


def get_selective_metagraph_commitments(
    decoded: dict,
) -> Optional[tuple[tuple[str, str]]]:
    """Returns a tuple of hotkeys and commitments from decoded chain data if provided, else None."""
    if commitments := decoded.get("commitments"):
        result = []
        for commitment in commitments:
            account_id_bytes, commitment_bytes = commitment
            hotkey = decode_account_id(account_id_bytes)
            commitment = bytes(
                commitment_bytes[SELECTIVE_METAGRAPH_COMMITMENTS_OFFSET:]
            ).decode("utf-8", errors="ignore")
            result.append((hotkey, commitment))
        return tuple(result)
    return None


# to balance with unit (shortcut)
def _tbwu(val: Optional[int], netuid: Optional[int] = 0) -> Optional[Balance]:
    """Returns a Balance object from a value and unit."""
    if val is None:
        return None
    return Balance.from_rao(val, netuid)


def _chr_str(codes: tuple[int]) -> str:
    """Converts a tuple of integer Unicode code points into a string."""
    return "".join(map(chr, codes))


def process_nested(
    data: Union[tuple, dict], chr_transform
) -> Optional[Union[list, dict]]:
    """Processes nested data structures by applying a transformation function to their elements."""
    if isinstance(data, (list, tuple)):
        if len(data) > 0:
            return [
                {k: chr_transform(v) for k, v in item.items()}
                if item is not None
                else None
                for item in data
            ]
        return {}
    elif isinstance(data, dict):
        return {k: chr_transform(v) for k, v in data.items()}
    return None


@dataclass
class MetagraphInfo(InfoBase):
    # Subnet index
    netuid: int
    mechid: int

    # Name and symbol
    name: str
    symbol: str
    identity: Optional[SubnetIdentity]
    network_registered_at: int

    # Keys for owner.
    owner_hotkey: Optional[str]  # hotkey
    owner_coldkey: Optional[str]  # coldkey

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
    subnet_volume: Balance  # volume of the subnet in TAO
    moving_price: Balance  # subnet moving price.

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

    # List of validators
    validators: Optional[list[str]]

    commitments: Optional[tuple[tuple[str, str]]]

    @classmethod
    def _from_dict(cls, decoded: dict) -> "MetagraphInfo":
        """Returns a MetagraphInfo object from decoded chain data."""
        # Subnet index
        _netuid, _mechid = get_netuid_and_mechid_by_storage_index(decoded["netuid"])

        # Name and symbol
        if name := decoded.get("name"):
            decoded.update({"name": bytes(name).decode()})

        if symbol := decoded.get("symbol"):
            decoded.update({"symbol": bytes(symbol).decode()})

        ii_list = []
        if decoded.get("identity") is not None:
            ii_list.append("identity")

        if decoded.get("identities") is not None:
            ii_list.append("identities")

        for key in ii_list:
            raw_data = decoded.get(key)
            processed = process_nested(raw_data, _chr_str)
            decoded.update({key: processed})

        return cls(
            # Subnet index
            netuid=_netuid,
            mechid=_mechid,
            # Name and symbol
            name=decoded["name"],
            symbol=decoded["symbol"],
            identity=decoded["identity"],
            network_registered_at=decoded["network_registered_at"],
            # Keys for owner.
            owner_hotkey=(
                decode_account_id(decoded["owner_hotkey"][0])
                if decoded.get("owner_hotkey") is not None
                else None
            ),
            owner_coldkey=(
                decode_account_id(decoded["owner_coldkey"][0])
                if decoded.get("owner_coldkey") is not None
                else None
            ),
            # Tempo terms.
            block=decoded["block"],
            tempo=decoded["tempo"],
            last_step=decoded["last_step"],
            blocks_since_last_step=decoded["blocks_since_last_step"],
            # Subnet emission terms
            subnet_emission=_tbwu(decoded["subnet_emission"]),
            alpha_in=_tbwu(decoded["alpha_in"], _netuid),
            alpha_out=_tbwu(decoded["alpha_out"], _netuid),
            tao_in=_tbwu(decoded["tao_in"]),
            alpha_out_emission=_tbwu(decoded["alpha_out_emission"], _netuid),
            alpha_in_emission=_tbwu(decoded["alpha_in_emission"], _netuid),
            tao_in_emission=_tbwu(decoded["tao_in_emission"]),
            pending_alpha_emission=_tbwu(decoded["pending_alpha_emission"], _netuid),
            pending_root_emission=_tbwu(decoded["pending_root_emission"]),
            subnet_volume=_tbwu(decoded["subnet_volume"], _netuid),
            moving_price=(
                Balance.from_tao(fixed_to_float(decoded.get("moving_price"), 32))
                if decoded.get("moving_price") is not None
                else None
            ),
            # Hparams for epoch
            rho=decoded["rho"],
            kappa=decoded["kappa"],
            # Validator params
            min_allowed_weights=(
                u16tf(decoded["min_allowed_weights"])
                if decoded.get("min_allowed_weights") is not None
                else None
            ),
            max_weights_limit=(
                u16tf(decoded["max_weights_limit"])
                if decoded["max_weights_limit"] is not None
                else None
            ),
            weights_version=decoded["weights_version"],
            weights_rate_limit=decoded["weights_rate_limit"],
            activity_cutoff=decoded["activity_cutoff"],
            max_validators=decoded["max_validators"],
            # Registration
            num_uids=decoded["num_uids"],
            max_uids=decoded["max_uids"],
            burn=_tbwu(decoded["burn"]),
            difficulty=(
                u64tf(decoded["difficulty"])
                if decoded["difficulty"] is not None
                else None
            ),
            registration_allowed=decoded["registration_allowed"],
            pow_registration_allowed=decoded["pow_registration_allowed"],
            immunity_period=decoded["immunity_period"],
            min_difficulty=(
                u64tf(decoded["min_difficulty"])
                if decoded["min_difficulty"] is not None
                else None
            ),
            max_difficulty=(
                u64tf(decoded["max_difficulty"])
                if decoded["max_difficulty"] is not None
                else None
            ),
            min_burn=_tbwu(decoded["min_burn"]),
            max_burn=_tbwu(decoded["max_burn"]),
            adjustment_alpha=(
                u64tf(decoded["adjustment_alpha"])
                if decoded["adjustment_alpha"] is not None
                else None
            ),
            adjustment_interval=decoded["adjustment_interval"],
            target_regs_per_interval=decoded["target_regs_per_interval"],
            max_regs_per_block=decoded["max_regs_per_block"],
            serving_rate_limit=decoded["serving_rate_limit"],
            # CR
            commit_reveal_weights_enabled=decoded["commit_reveal_weights_enabled"],
            commit_reveal_period=decoded["commit_reveal_period"],
            # Bonds
            liquid_alpha_enabled=decoded["liquid_alpha_enabled"],
            alpha_high=(
                u16tf(decoded["alpha_high"])
                if decoded["alpha_high"] is not None
                else None
            ),
            alpha_low=(
                u16tf(decoded["alpha_low"])
                if decoded["alpha_low"] is not None
                else None
            ),
            bonds_moving_avg=(
                u64tf(decoded["bonds_moving_avg"])
                if decoded["bonds_moving_avg"] is not None
                else None
            ),
            # Metagraph info.
            hotkeys=(
                [decode_account_id(ck) for ck in decoded.get("hotkeys", [])]
                if decoded.get("hotkeys") is not None
                else None
            ),
            coldkeys=(
                [decode_account_id(hk) for hk in decoded.get("coldkeys", [])]
                if decoded.get("coldkeys") is not None
                else None
            ),
            identities=decoded["identities"],
            axons=decoded.get("axons", []),
            active=decoded["active"],
            validator_permit=decoded["validator_permit"],
            pruning_score=(
                [u16tf(ps) for ps in decoded.get("pruning_score", [])]
                if decoded.get("pruning_score") is not None
                else None
            ),
            last_update=decoded["last_update"],
            emission=(
                [_tbwu(em, _netuid) for em in decoded.get("emission", [])]
                if decoded.get("emission") is not None
                else None
            ),
            dividends=(
                [u16tf(dv) for dv in decoded.get("dividends", [])]
                if decoded.get("dividends") is not None
                else None
            ),
            incentives=(
                [u16tf(ic) for ic in decoded.get("incentives", [])]
                if decoded.get("incentives") is not None
                else None
            ),
            consensus=(
                [u16tf(cs) for cs in decoded.get("consensus", [])]
                if decoded.get("consensus") is not None
                else None
            ),
            trust=(
                [u16tf(tr) for tr in decoded.get("trust", [])]
                if decoded.get("trust") is not None
                else None
            ),
            rank=(
                [u16tf(rk) for rk in decoded.get("rank", [])]
                if decoded.get("rank") is not None
                else None
            ),
            block_at_registration=decoded["block_at_registration"],
            alpha_stake=(
                [_tbwu(ast, _netuid) for ast in decoded["alpha_stake"]]
                if decoded.get("alpha_stake") is not None
                else None
            ),
            tao_stake=(
                [
                    _tbwu(ts) * settings.ROOT_TAO_STAKE_WEIGHT
                    for ts in decoded["tao_stake"]
                ]
                if decoded.get("tao_stake") is not None
                else None
            ),
            total_stake=(
                [_tbwu(ts, _netuid) for ts in decoded["total_stake"]]
                if decoded.get("total_stake") is not None
                else None
            ),
            # Dividend break down
            tao_dividends_per_hotkey=(
                [
                    (decode_account_id(alpha[0]), _tbwu(alpha[1]))
                    for alpha in decoded["tao_dividends_per_hotkey"]
                ]
                if decoded.get("tao_dividends_per_hotkey") is not None
                else None
            ),
            alpha_dividends_per_hotkey=(
                [
                    (decode_account_id(adphk[0]), _tbwu(adphk[1], _netuid))
                    for adphk in decoded["alpha_dividends_per_hotkey"]
                ]
                if decoded.get("alpha_dividends_per_hotkey") is not None
                else None
            ),
            validators=[v for v in decoded["validators"]]
            if decoded.get("validators")
            else None,
            commitments=get_selective_metagraph_commitments(decoded),
        )


@dataclass
class MetagraphInfoEmissions:
    """Emissions presented in tao values."""

    subnet_emission: float
    alpha_in_emission: float
    alpha_out_emission: float
    tao_in_emission: float
    pending_alpha_emission: float
    pending_root_emission: float


@dataclass
class MetagraphInfoPool:
    """Pool presented in tao values."""

    alpha_out: float
    alpha_in: float
    tao_in: float
    subnet_volume: float
    moving_price: float


@dataclass
class MetagraphInfoParams:
    activity_cutoff: int
    adjustment_alpha: float
    adjustment_interval: int
    alpha_high: float
    alpha_low: float
    bonds_moving_avg: float
    burn: float
    commit_reveal_period: int
    commit_reveal_weights_enabled: bool
    difficulty: float
    immunity_period: int
    kappa: float
    liquid_alpha_enabled: bool
    max_burn: float
    max_difficulty: float
    max_regs_per_block: int
    max_validators: int
    max_weights_limit: float
    min_allowed_weights: float
    min_burn: float
    min_difficulty: float
    pow_registration_allowed: bool
    registration_allowed: bool
    rho: int
    serving_rate_limit: int
    target_regs_per_interval: int
    tempo: int
    weights_rate_limit: int
    weights_version: int


class SelectiveMetagraphIndex(Enum):
    Netuid = 0
    Name = 1
    Symbol = 2
    Identity = 3
    NetworkRegisteredAt = 4
    OwnerHotkey = 5
    OwnerColdkey = 6
    Block = 7
    Tempo = 8
    LastStep = 9
    BlocksSinceLastStep = 10
    SubnetEmission = 11
    AlphaIn = 12
    AlphaOut = 13
    TaoIn = 14
    AlphaOutEmission = 15
    AlphaInEmission = 16
    TaoInEmission = 17
    PendingAlphaEmission = 18
    PendingRootEmission = 19
    SubnetVolume = 20
    MovingPrice = 21
    Rho = 22
    Kappa = 23
    MinAllowedWeights = 24
    MaxWeightsLimit = 25
    WeightsVersion = 26
    WeightsRateLimit = 27
    ActivityCutoff = 28
    MaxValidators = 29
    NumUids = 30
    MaxUids = 31
    Burn = 32
    Difficulty = 33
    RegistrationAllowed = 34
    PowRegistrationAllowed = 35
    ImmunityPeriod = 36
    MinDifficulty = 37
    MaxDifficulty = 38
    MinBurn = 39
    MaxBurn = 40
    AdjustmentAlpha = 41
    AdjustmentInterval = 42
    TargetRegsPerInterval = 43
    MaxRegsPerBlock = 44
    ServingRateLimit = 45
    CommitRevealWeightsEnabled = 46
    CommitRevealPeriod = 47
    LiquidAlphaEnabled = 48
    AlphaHigh = 49
    AlphaLow = 50
    BondsMovingAvg = 51
    Hotkeys = 52
    Coldkeys = 53
    Identities = 54
    Axons = 55
    Active = 56
    ValidatorPermit = 57
    PruningScore = 58
    LastUpdate = 59
    Emission = 60
    Dividends = 61
    Incentives = 62
    Consensus = 63
    Trust = 64
    Rank = 65
    BlockAtRegistration = 66
    AlphaStake = 67
    TaoStake = 68
    TotalStake = 69
    TaoDividendsPerHotkey = 70
    AlphaDividendsPerHotkey = 71
    Validators = 72
    Commitments = 73

    @staticmethod
    def all_indices() -> list[int]:
        return [member.value for member in SelectiveMetagraphIndex]
