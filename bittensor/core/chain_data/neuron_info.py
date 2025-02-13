from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from bittensor.core.chain_data.axon_info import AxonInfo
from bittensor.core.chain_data.info_base import InfoBase
from bittensor.core.chain_data.prometheus_info import PrometheusInfo
from bittensor.core.chain_data.utils import decode_account_id, process_stake_data
from bittensor.utils import u16_normalized_float
from bittensor.utils.balance import Balance

# for annotation purposes
if TYPE_CHECKING:
    from bittensor.core.chain_data.neuron_info_lite import NeuronInfoLite


@dataclass
class NeuronInfo(InfoBase):
    """Represents the metadata of a neuron including keys, UID, stake, rankings, and other attributes.

    Attributes:
        hotkey (str): The hotkey associated with the neuron.
        coldkey (str): The coldkey associated with the neuron.
        uid (int): The unique identifier for the neuron.
        netuid (int): The network unique identifier for the neuron.
        active (int): The active status of the neuron.
        stake (Balance): The balance staked to this neuron.
        stake_dict (dict[str, Balance]): A dictionary mapping coldkey to the amount staked.
        total_stake (Balance): The total amount of stake.
        rank (float): The rank score of the neuron.
        emission (float): The emission rate.
        incentive (float): The incentive value.
        consensus (float): The consensus score.
        trust (float): The trust score.
        validator_trust (float): The validation trust score.
        dividends (float): The dividends value.
        last_update (int): The timestamp of the last update.
        validator_permit (bool): Validator permit status.
        weights (list[tuple[int]]): List of weights associated with the neuron.
        bonds (list[list[int]]): List of bonds associated with the neuron.
        pruning_score (int): The pruning score of the neuron.
        prometheus_info (Optional[PrometheusInfo]): Information related to Prometheus.
        axon_info (Optional[AxonInfo]): Information related to Axon.
        is_null (bool): Indicator if this is a null neuron.
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
    weights: list[tuple[int, int]]
    bonds: list[list[int]]
    pruning_score: int
    prometheus_info: Optional["PrometheusInfo"] = None
    axon_info: Optional["AxonInfo"] = None
    is_null: bool = False

    @classmethod
    def from_weights_bonds_and_neuron_lite(
        cls,
        neuron_lite: "NeuronInfoLite",
        weights_as_dict: dict[int, list[tuple[int, int]]],
        bonds_as_dict: dict[int, list[tuple[int, int]]],
    ) -> "NeuronInfo":
        """
        Creates an instance of NeuronInfo from NeuronInfoLite and dictionaries of weights and bonds.

        Args:
            neuron_lite (NeuronInfoLite): A lite version of the neuron containing basic attributes.
            weights_as_dict (dict[int, list[tuple[int, int]]]): A dictionary where the key is the UID and the value is
                a list of weight tuples associated with the neuron.
            bonds_as_dict (dict[int, list[tuple[int, int]]]): A dictionary where the key is the UID and the value is a
                list of bond tuples associated with the neuron.

        Returns:
            NeuronInfo: An instance of NeuronInfo populated with the provided weights and bonds.
        """
        n_dict = neuron_lite.__dict__
        n_dict["weights"] = weights_as_dict.get(neuron_lite.uid, [])
        n_dict["bonds"] = bonds_as_dict.get(neuron_lite.uid, [])

        return cls(**n_dict)

    @staticmethod
    def get_null_neuron() -> "NeuronInfo":
        """Returns a null NeuronInfo instance."""
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
    def _from_dict(cls, decoded: Any) -> "NeuronInfo":
        """Returns a NeuronInfo object from decoded chain data."""
        stake_dict = process_stake_data(decoded["stake"])
        total_stake = sum(stake_dict.values()) if stake_dict else Balance(0)
        coldkey = decode_account_id(decoded["coldkey"])
        hotkey = decode_account_id(decoded["hotkey"])
        return NeuronInfo(
            active=decoded["active"],
            axon_info=AxonInfo.from_dict(
                decoded["axon_info"]
                | {
                    "hotkey": hotkey,
                    "coldkey": coldkey,
                },
            ),
            bonds=[[e[0], e[1]] for e in decoded["bonds"]],
            coldkey=coldkey,
            consensus=u16_normalized_float(decoded["consensus"]),
            dividends=u16_normalized_float(decoded["dividends"]),
            emission=decoded["emission"] / 1e9,
            hotkey=hotkey,
            incentive=u16_normalized_float(decoded["incentive"]),
            is_null=False,
            last_update=decoded["last_update"],
            netuid=decoded["netuid"],
            prometheus_info=PrometheusInfo.from_dict(decoded["prometheus_info"]),
            pruning_score=decoded["pruning_score"],
            rank=u16_normalized_float(decoded["rank"]),
            stake_dict=stake_dict,
            stake=total_stake,
            total_stake=total_stake,
            trust=u16_normalized_float(decoded["trust"]),
            uid=decoded["uid"],
            validator_permit=decoded["validator_permit"],
            validator_trust=u16_normalized_float(decoded["validator_trust"]),
            weights=[(e[0], e[1]) for e in decoded["weights"]],
        )
