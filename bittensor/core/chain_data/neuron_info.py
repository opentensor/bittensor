from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import bt_decode
import netaddr

from bittensor.core.chain_data.axon_info import AxonInfo
from bittensor.core.chain_data.prometheus_info import PrometheusInfo
from bittensor.core.chain_data.utils import decode_account_id, process_stake_data
from bittensor.utils import u16_normalized_float
from bittensor.utils.balance import Balance

if TYPE_CHECKING:
    from bittensor.core.chain_data.neuron_info_lite import NeuronInfoLite


@dataclass
class NeuronInfo:
    """Dataclass for neuron metadata."""

    hotkey: str
    coldkey: str
    uid: int
    netuid: int
    active: int
    stake: Balance
    # mapping of coldkey to amount staked to this Neuron
    stake_dict: dict[str, Balance]
    total_stake: Balance
    rank: float
    emission: float
    incentive: float
    consensus: float
    trust: float
    validator_trust: float
    dividends: float
    last_update: int
    validator_permit: bool
    weights: list[list[int]]
    bonds: list[list[int]]
    pruning_score: int
    prometheus_info: Optional["PrometheusInfo"] = None
    axon_info: Optional[AxonInfo] = None
    is_null: bool = False

    @classmethod
    def from_weights_bonds_and_neuron_lite(
        cls,
        neuron_lite: "NeuronInfoLite",
        weights_as_dict: dict[int, list[tuple[int, int]]],
        bonds_as_dict: dict[int, list[tuple[int, int]]],
    ) -> "NeuronInfo":
        n_dict = neuron_lite.__dict__
        n_dict["weights"] = weights_as_dict.get(neuron_lite.uid, [])
        n_dict["bonds"] = bonds_as_dict.get(neuron_lite.uid, [])

        return cls(**n_dict)

    @staticmethod
    def get_null_neuron() -> "NeuronInfo":
        neuron = NeuronInfo(
            uid=0,
            netuid=0,
            active=0,
            stake=Balance.from_rao(0),
            stake_dict={},
            total_stake=Balance.from_rao(0),
            rank=0,
            emission=0,
            incentive=0,
            consensus=0,
            trust=0,
            validator_trust=0,
            dividends=0,
            last_update=0,
            validator_permit=False,
            weights=[],
            bonds=[],
            prometheus_info=None,
            axon_info=None,
            is_null=True,
            coldkey="000000000000000000000000000000000000000000000000",
            hotkey="000000000000000000000000000000000000000000000000",
            pruning_score=0,
        )
        return neuron

    @classmethod
    def from_vec_u8(cls, vec_u8: bytes) -> "NeuronInfo":
        n = bt_decode.NeuronInfo.decode(bytes(vec_u8))
        stake_dict = process_stake_data(n.stake)
        total_stake = sum(stake_dict.values()) if stake_dict else Balance(0)
        axon_info = n.axon_info
        coldkey = decode_account_id(n.coldkey)
        hotkey = decode_account_id(n.hotkey)
        return NeuronInfo(
            hotkey=hotkey,
            coldkey=coldkey,
            uid=n.uid,
            netuid=n.netuid,
            active=n.active,
            stake=total_stake,
            stake_dict=stake_dict,
            total_stake=total_stake,
            rank=u16_normalized_float(n.rank),
            emission=n.emission / 1e9,
            incentive=u16_normalized_float(n.incentive),
            consensus=u16_normalized_float(n.consensus),
            trust=u16_normalized_float(n.trust),
            validator_trust=u16_normalized_float(n.validator_trust),
            dividends=u16_normalized_float(n.dividends),
            last_update=n.last_update,
            validator_permit=n.validator_permit,
            weights=[[e[0], e[1]] for e in n.weights],
            bonds=[[e[0], e[1]] for e in n.bonds],
            pruning_score=n.pruning_score,
            prometheus_info=PrometheusInfo(
                block=n.prometheus_info.block,
                version=n.prometheus_info.version,
                ip=str(netaddr.IPAddress(n.prometheus_info.ip)),
                port=n.prometheus_info.port,
                ip_type=n.prometheus_info.ip_type,
            ),
            axon_info=AxonInfo(
                version=axon_info.version,
                ip=str(netaddr.IPAddress(axon_info.ip)),
                port=axon_info.port,
                ip_type=axon_info.ip_type,
                placeholder1=axon_info.placeholder1,
                placeholder2=axon_info.placeholder2,
                protocol=axon_info.protocol,
                hotkey=hotkey,
                coldkey=coldkey,
            ),
            is_null=False,
        )
