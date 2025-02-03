from dataclasses import dataclass
from typing import Any, Optional

from bittensor.core.chain_data.axon_info import AxonInfo
from bittensor.core.chain_data.info_base import InfoBase
from bittensor.core.chain_data.prometheus_info import PrometheusInfo
from bittensor.core.chain_data.utils import decode_account_id, process_stake_data
from bittensor.utils import u16_normalized_float
from bittensor.utils.balance import Balance


@dataclass
class NeuronInfoLite(InfoBase):
    """
    NeuronInfoLite is a dataclass representing neuron metadata without weights and bonds.

    Attributes:
        hotkey (str): The hotkey string for the neuron.
        coldkey (str): The coldkey string for the neuron.
        uid (int): A unique identifier for the neuron.
        netuid (int): Network unique identifier for the neuron.
        active (int): Indicates whether the neuron is active.
        stake (Balance): The stake amount associated with the neuron.
        stake_dict (dict): Mapping of coldkey to the amount staked to this Neuron.
        total_stake (Balance): Total amount of the stake.
        rank (float): The rank of the neuron.
        emission (float): The emission value of the neuron.
        incentive (float): The incentive value of the neuron.
        consensus (float): The consensus value of the neuron.
        trust (float): Trust value of the neuron.
        validator_trust (float): Validator trust value of the neuron.
        dividends (float): Dividends associated with the neuron.
        last_update (int): Timestamp of the last update.
        validator_permit (bool): Indicates if the neuron has a validator permit.
        prometheus_info (Optional[PrometheusInfo]): Prometheus information associated with the neuron.
        axon_info (Optional[AxonInfo]): Axon information associated with the neuron.
        pruning_score (int): The pruning score of the neuron.
        is_null (bool): Indicates whether the neuron is null.

    Methods:
        get_null_neuron: Returns a NeuronInfoLite object representing a null neuron.
        list_from_vec_u8: Decodes a bytes object into a list of NeuronInfoLite instances.
    """

    hotkey: str
    coldkey: str
    uid: int
    netuid: int
    active: int
    stake: "Balance"
    # mapping of coldkey to amount staked to this Neuron
    stake_dict: dict[str, "Balance"]
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

    @staticmethod
    def get_null_neuron() -> "NeuronInfoLite":
        """Returns a null NeuronInfoLite instance."""
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

    @classmethod
    def _from_dict(cls, decoded: Any) -> "NeuronInfoLite":
        coldkey = decode_account_id(decoded["coldkey"])
        hotkey = decode_account_id(decoded["hotkey"])
        stake_dict = process_stake_data(decoded["stake"])
        stake = sum(stake_dict.values()) if stake_dict else Balance(0)

        return NeuronInfoLite(
            active=decoded["active"],
            axon_info=AxonInfo.from_dict(
                decoded["axon_info"]
                | {
                    "hotkey": hotkey,
                    "coldkey": coldkey,
                },
            ),
            coldkey=coldkey,
            consensus=u16_normalized_float(decoded["consensus"]),
            dividends=u16_normalized_float(decoded["dividends"]),
            emission=decoded["emission"] / 1e9,
            hotkey=hotkey,
            incentive=u16_normalized_float(decoded["incentive"]),
            last_update=decoded["last_update"],
            netuid=decoded["netuid"],
            prometheus_info=PrometheusInfo.from_dict(decoded["prometheus_info"]),
            pruning_score=decoded["pruning_score"],
            rank=u16_normalized_float(decoded["rank"]),
            stake_dict=stake_dict,
            stake=stake,
            total_stake=stake,
            trust=u16_normalized_float(decoded["trust"]),
            uid=decoded["uid"],
            validator_permit=decoded["validator_permit"],
            validator_trust=u16_normalized_float(decoded["validator_trust"]),
        )
