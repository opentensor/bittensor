from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from scalecodec.utils.ss58 import ss58_encode

from bittensor.core.chain_data.axon_info import AxonInfo
from bittensor.core.chain_data.prometheus_info import PrometheusInfo
from bittensor.core.chain_data.utils import from_scale_encoding, ChainDataType
from bittensor.core.settings import SS58_FORMAT
from bittensor.utils import RAOPERTAO, u16_normalized_float
from bittensor.utils.balance import Balance


@dataclass
class NeuronInfoLite:
    """Dataclass for neuron metadata, but without the weights and bonds."""

    hotkey: str
    coldkey: str
    uid: int
    netuid: int
    active: int
    stake: "Balance"
    # mapping of coldkey to amount staked to this Neuron
    stake_dict: Dict[str, "Balance"]
    total_stake: "Balance"
    rank: float
    emission: float
    incentive: float
    consensus: float
    trust: float
    validator_trust: float
    dividends: float
    last_update: int
    validator_permit: bool
    prometheus_info: Optional["PrometheusInfo"]
    axon_info: Optional["AxonInfo"]
    pruning_score: int
    is_null: bool = False

    @classmethod
    def fix_decoded_values(cls, neuron_info_decoded: Any) -> "NeuronInfoLite":
        """Fixes the values of the NeuronInfoLite object."""
        neuron_info_decoded["hotkey"] = ss58_encode(
            neuron_info_decoded["hotkey"], SS58_FORMAT
        )
        neuron_info_decoded["coldkey"] = ss58_encode(
            neuron_info_decoded["coldkey"], SS58_FORMAT
        )
        stake_dict = {
            ss58_encode(coldkey, SS58_FORMAT): Balance.from_rao(int(stake))
            for coldkey, stake in neuron_info_decoded["stake"]
        }
        neuron_info_decoded["stake_dict"] = stake_dict
        neuron_info_decoded["stake"] = sum(stake_dict.values())
        neuron_info_decoded["total_stake"] = neuron_info_decoded["stake"]
        neuron_info_decoded["rank"] = u16_normalized_float(neuron_info_decoded["rank"])
        neuron_info_decoded["emission"] = neuron_info_decoded["emission"] / RAOPERTAO
        neuron_info_decoded["incentive"] = u16_normalized_float(
            neuron_info_decoded["incentive"]
        )
        neuron_info_decoded["consensus"] = u16_normalized_float(
            neuron_info_decoded["consensus"]
        )
        neuron_info_decoded["trust"] = u16_normalized_float(
            neuron_info_decoded["trust"]
        )
        neuron_info_decoded["validator_trust"] = u16_normalized_float(
            neuron_info_decoded["validator_trust"]
        )
        neuron_info_decoded["dividends"] = u16_normalized_float(
            neuron_info_decoded["dividends"]
        )
        neuron_info_decoded["prometheus_info"] = PrometheusInfo.fix_decoded_values(
            neuron_info_decoded["prometheus_info"]
        )
        neuron_info_decoded["axon_info"] = AxonInfo.from_neuron_info(
            neuron_info_decoded
        )
        return cls(**neuron_info_decoded)

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> "NeuronInfoLite":
        """Returns a NeuronInfoLite object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return NeuronInfoLite.get_null_neuron()

        decoded = from_scale_encoding(vec_u8, ChainDataType.NeuronInfoLite)
        if decoded is None:
            return NeuronInfoLite.get_null_neuron()

        return NeuronInfoLite.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["NeuronInfoLite"]:
        """Returns a list of NeuronInfoLite objects from a ``vec_u8``."""

        decoded_list = from_scale_encoding(
            vec_u8, ChainDataType.NeuronInfoLite, is_vec=True
        )
        if decoded_list is None:
            return []

        decoded_list = [
            NeuronInfoLite.fix_decoded_values(decoded) for decoded in decoded_list
        ]
        return decoded_list

    @staticmethod
    def get_null_neuron() -> "NeuronInfoLite":
        neuron = NeuronInfoLite(
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
            prometheus_info=None,
            axon_info=None,
            is_null=True,
            coldkey="000000000000000000000000000000000000000000000000",
            hotkey="000000000000000000000000000000000000000000000000",
            pruning_score=0,
        )
        return neuron
