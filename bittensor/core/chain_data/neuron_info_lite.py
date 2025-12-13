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
        hotkey: The hotkey string for the neuron.
        coldkey: The coldkey string for the neuron.
        uid: A unique identifier for the neuron.
        netuid: Network unique identifier for the neuron.
        active: Indicates whether the neuron is active.
        stake: The stake amount associated with the neuron.
        stake_dict: Mapping of coldkey to the amount staked to this Neuron.
        total_stake: Total amount of the stake.
        emission: The emission value of the neuron.
        incentive: The incentive value of the neuron.
        consensus: The consensus value of the neuron.
        validator_trust: Validator trust value of the neuron.
        dividends: Dividends associated with the neuron.
        last_update: Timestamp of the last update.
        validator_permit: Indicates if the neuron has a validator permit.
        prometheus_info: Prometheus information associated with the neuron.
        axon_info: Axon information associated with the neuron.
        is_null: Indicates whether the neuron is null.

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
    emission: float
    incentive: float
    consensus: float
    validator_trust: float
    dividends: float
    last_update: int
    validator_permit: bool
    prometheus_info: Optional["PrometheusInfo"]
    axon_info: Optional["AxonInfo"]
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
            emission=0,
            incentive=0,
            consensus=0,
            validator_trust=0,
            dividends=0,
            last_update=0,
            validator_permit=False,
            prometheus_info=None,
            axon_info=None,
            is_null=True,
            coldkey="000000000000000000000000000000000000000000000000",
            hotkey="000000000000000000000000000000000000000000000000",
        )
        return neuron

    @classmethod
    def _from_dict(cls, decoded: Any) -> "NeuronInfoLite":
        """Returns a NeuronInfoLite object from decoded chain data."""
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
            stake_dict=stake_dict,
            stake=stake,
            total_stake=stake,
            uid=decoded["uid"],
            validator_permit=decoded["validator_permit"],
            validator_trust=u16_normalized_float(decoded["validator_trust"]),
        )
