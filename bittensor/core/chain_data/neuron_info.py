from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import bt_decode
import netaddr

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
    def _fix_decoded(cls, decoded: Any) -> "NeuronInfo":
        """Instantiates NeuronInfo from a byte vector."""
        stake_dict = process_stake_data(decoded.stake)
        total_stake = sum(stake_dict.values()) if stake_dict else Balance(0)
        axon_info = decoded.axon_info
        coldkey = decode_account_id(decoded.coldkey)
        hotkey = decode_account_id(decoded.hotkey)
        return NeuronInfo(
            hotkey=hotkey,
            coldkey=coldkey,
            uid=decoded.uid,
            netuid=decoded.netuid,
            active=decoded.active,
            stake=total_stake,
            stake_dict=stake_dict,
            total_stake=total_stake,
            rank=u16_normalized_float(decoded.rank),
            emission=decoded.emission / 1e9,
            incentive=u16_normalized_float(decoded.incentive),
            consensus=u16_normalized_float(decoded.consensus),
            trust=u16_normalized_float(decoded.trust),
            validator_trust=u16_normalized_float(decoded.validator_trust),
            dividends=u16_normalized_float(decoded.dividends),
            last_update=decoded.last_update,
            validator_permit=decoded.validator_permit,
            weights=[(e[0], e[1]) for e in decoded.weights],
            bonds=[[e[0], e[1]] for e in decoded.bonds],
            pruning_score=decoded.pruning_score,
            prometheus_info=PrometheusInfo(
                block=decoded.prometheus_info.block,
                version=decoded.prometheus_info.version,
                ip=str(netaddr.IPAddress(decoded.prometheus_info.ip)),
                port=decoded.prometheus_info.port,
                ip_type=decoded.prometheus_info.ip_type,
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

    @classmethod
    def list_from_vec_u8(cls, vec_u8: bytes) -> list["NeuronInfo"]:
        nn = bt_decode.NeuronInfo.decode_vec(bytes(vec_u8))

        return [cls._fix_decoded(n) for n in nn]

    @classmethod
    def from_vec_u8(cls, vec_u8: bytes) -> "NeuronInfo":
        n = bt_decode.NeuronInfo.decode(vec_u8)
        return cls._fix_decoded(n)
