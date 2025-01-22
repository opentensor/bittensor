from dataclasses import dataclass
from typing import Optional

from bittensor.core.chain_data.axon_info import AxonInfo
from bittensor.core.chain_data.chain_identity import ChainIdentity
from bittensor.core.chain_data.subnet_identity import SubnetIdentity
from bittensor.core.chain_data.utils import (
    ChainDataType,
    from_scale_encoding,
)


@dataclass
class MetagraphInfo:
    # Subnet index
    netuid: int

    # Name and symbol
    name: str
    symbol: str
    identity: Optional["SubnetIdentity"]
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
    subnet_emission: int
    alpha_in: int
    alpha_out: int
    tao_in: int  # amount of tao injected per block
    alpha_out_emission: int  # amount injected in alpha reserves per block
    alpha_in_emission: int  # amount injected outstanding per block
    tao_in_emission: int  # amount of tao injected per block
    pending_alpha_emission: int  # pending alpha to be distributed
    pending_root_emission: int  # pending tao for root divs to be distributed

    # Hparams for epoch
    rho: int  # subnet rho param
    kappa: int  # subnet kappa param

    # Validator params
    min_allowed_weights: int  # min allowed weights per val
    max_weights_limit: int  # max allowed weights per val
    weights_version: int  # allowed weights version
    weights_rate_limit: int  # rate limit on weights.
    activity_cutoff: int  # validator weights cut off period in blocks
    max_validators: int  # max allowed validators.

    # Registration
    num_uids: int
    max_uids: int
    burn: int  # current burn cost.
    difficulty: int  # current difficulty.
    registration_allowed: bool  # allows registrations.
    immunity_period: int  # subnet miner immunity period
    min_difficulty: int  # min pow difficulty
    max_difficulty: int  # max pow difficulty
    min_burn: int  # min tao burn
    max_burn: int  # max tao burn
    adjustment_alpha: int  # adjustment speed for registration params.
    adjustment_interval: int  # pow and burn adjustment interval
    target_regs_per_interval: int  # target registrations per interval
    max_regs_per_block: int  # max registrations per block.
    serving_rate_limit: int  # axon serving rate limit

    # CR
    commit_reveal_weights_enabled: bool  # Is CR enabled.
    commit_reveal_period: int  # Commit reveal interval

    # Bonds
    liquid_alpha_enabled: bool  # Bonds liquid enabled.
    alpha_high: int  # Alpha param high
    alpha_low: int  # Alpha param low
    bonds_moving_avg: int  # Bonds moving avg

    # Metagraph info.
    hotkeys: list[str]  # hotkey per UID
    coldkeys: list[str]  # coldkey per UID
    identities: list["ChainIdentity"]  # coldkeys identities
    axons: list["AxonInfo"]  # UID axons.
    active: list[bool]  # Active per UID
    validator_permit: list[bool]  # Val permit per UID
    pruning_score: list[int]  # Pruning per UID
    last_update: list[int]  # Last update per UID
    emission: list[int]  # Emission per UID
    dividends: list[int]  # Dividends per UID
    incentives: list[int]  # Mining incentives per UID
    consensus: list[int]  # Consensus per UID
    trust: list[int]  # Trust per UID
    rank: list[int]  # Rank per UID
    block_at_registration: list[int]  # Reg block per UID
    alpha_stake: list[int]  # Alpha staked per UID
    tao_stake: list[int]  # TAO staked per UID
    total_stake: list[int]  # Total stake per UID

    # Dividend break down.
    tao_dividends_per_hotkey: list[
        tuple[str, int]
    ]  # List of dividend payouts in tao via root.
    alpha_dividends_per_hotkey: list[
        tuple[str, int]
    ]  # List of dividend payout in alpha via subnet.

    @classmethod
    def from_vec_u8(cls, vec_u8: bytes) -> Optional["MetagraphInfo"]:
        """Returns a Metagraph object from a encoded MetagraphInfo vector."""
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
        decoded = [MetagraphInfo.fix_decoded_values(d) for d in decoded]
        return decoded

    @classmethod
    def fix_decoded_values(cls, decoded: dict) -> "MetagraphInfo":
        """Returns a Metagraph object from a decoded MetagraphInfo dictionary."""
        decoded.update({"name": bytes(decoded.get("name")).decode()})
        decoded.update({"symbol": bytes(decoded.get("symbol")).decode()})
        decoded.update({"identity": decoded.get("identity", {})})
        decoded.update({"identities": decoded.get("identities", {})})
        decoded.update({"axons": decoded.get("axons", {})})

        return MetagraphInfo(**decoded)
