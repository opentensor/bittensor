
from dataclasses import dataclass
from typing import List, Tuple, Dict
import bittensor
from bittensor import Balance
import scalecodec


# Constants
RAOPERTAO = 1e9
U16_MAX = 65535
U64_MAX = 18446744073709551615

# Dataclasses for chain data.
@dataclass
class NeuronInfo:
    r"""
    Dataclass for neuron metadata.
    """
    hotkey: str
    coldkey: str
    uid: int
    netuid: int
    active: int
    stake: List[Tuple[str, Balance]]
    total_stake: Balance
    rank: float
    emission: float
    incentive: float
    consensus: float
    trust: float
    dividends: float
    last_update: int
    weights: List[List[int]]
    bonds: List[List[int]]
    prometheus_info: 'PrometheusInfo'
    axon_info: 'AxonInfo'
    is_null: bool = False

    @staticmethod
    def __u8_key_to_ss58(u8_key: Dict) -> str:
        r""" Converts a u8 key to ss58.
        """
        # First byte is length, then 32 bytes of key.
        return scalecodec.ss58_encode( bytes(u8_key['id']).hex(), bittensor.__ss58_format__)
        
    @classmethod
    def from_json(cls, json: Dict) -> 'NeuronInfo':
        r""" Returns a NeuronInfo object from a json dictionary.
        """
        return NeuronInfo(
            hotkey = cls.__u8_key_to_ss58(json['hotkey']),
            coldkey = cls.__u8_key_to_ss58(json['coldkey']),
            uid = json['uid'],
            netuid = json['netuid'],
            active = int(json['active']), # 0 or 1
            stake = [(cls.__u8_key_to_ss58(coldkey), Balance.from_rao(stake) ) for coldkey, stake in json['stake']],
            total_stake = Balance.from_rao(sum([stake for _, stake in json['stake']])),
            rank = json['rank'] / U64_MAX,
            emission = json['emission'] / RAOPERTAO,
            incentive = json['incentive'] / U64_MAX,
            consensus = json['consensus'] / U64_MAX,
            trust = json['trust'] / U64_MAX,
            dividends = json['dividends'] / U64_MAX,
            last_update = json['last_update'],
            weights = json['weights'],
            bonds = json['bonds'],
            prometheus_info = PrometheusInfo.from_json(json['prometheus_info']),
            axon_info = AxonInfo.from_json(json['axon_info']),
        )

    @staticmethod
    def _null_neuron() -> 'NeuronInfo':
        neuron = NeuronInfo(
            uid = 0,
            netuid = 0,
            active =  0,
            stake = 0,
            total_stake = 0,
            rank = 0,
            emission = 0,
            incentive = 0,
            consensus = 0,
            trust = 0,
            dividends = 0,
            last_update = 0,
            weights = [],
            bonds = [],
            prometheus_info = None,
            axon_info = None,
            is_null = True,
            coldkey = "000000000000000000000000000000000000000000000000",
            hotkey = "000000000000000000000000000000000000000000000000"
        )
        return neuron

    @staticmethod
    def _neuron_dict_to_namespace(neuron_dict) -> 'NeuronInfo':
        if neuron_dict['hotkey'] == '5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM':
            return NeuronInfo._null_neuron()
        else:
            neuron = NeuronInfo( **neuron_dict )
            neuron.stake = neuron.stake / RAOPERTAO
            neuron.rank = neuron.rank / U64_MAX
            neuron.trust = neuron.trust / U64_MAX
            neuron.consensus = neuron.consensus / U64_MAX
            neuron.incentive = neuron.incentive / U64_MAX
            neuron.dividends = neuron.dividends / U64_MAX
            neuron.emission = neuron.emission / RAOPERTAO
                
            return neuron

@dataclass
class AxonInfo:
    r"""
    Dataclass for axon info.
    """
    block: int
    version: int
    ip: str
    port: int
    ip_type: int
    protocol: int
    placeholder1: int # placeholder for future use
    placeholder2: int

    @classmethod
    def from_json(cls, json: Dict) -> 'AxonInfo':
        r""" Returns a AxonInfo object from a json dictionary.
        """
        return AxonInfo(
            block = json['block'],
            version = json['version'],
            ip = bittensor.utils.networking.int_to_ip(int(json['ip'])),
            port = json['port'],
            ip_type = json['ip_type'],
            protocol = json['protocol'],
            placeholder1 = json['placeholder1'],
            placeholder2 = json['placeholder2'],
        )

@dataclass
class PrometheusInfo:
    r"""
    Dataclass for prometheus info.
    """
    block: int
    version: int
    ip: str
    port: int
    ip_type: int

    @classmethod
    def from_json(cls, json: Dict) -> 'PrometheusInfo':
        r""" Returns a PrometheusInfo object from a json dictionary.
        """
        return PrometheusInfo(
            block = json['block'],
            version = json['version'],
            ip = bittensor.utils.networking.int_to_ip(int(json['ip'])),
            port = json['port'],
            ip_type = json['ip_type'],
        )


@dataclass
class DelegateInfo:
    r"""
    Dataclass for delegate info.
    """
    hotkey_ss58: str # Hotkey of delegate
    total_stake: Balance # Total stake of the delegate
    nominators: List[Tuple[str, Balance]] # List of nominators of the delegate and their stake
    owner_ss58: str # Coldkey of owner
    take: float # Take of the delegate as a percentage


@dataclass
class SubnetInfo:
    r"""
    Dataclass for subnet info.
    """
    netuid: int
    rho: int
    kappa: int
    difficulty: int
    immunity_period: int
    validator_batch_size: int
    validator_sequence_length: int
    validator_epochs_per_reset: int
    validator_epoch_length: int
    max_allowed_validators: int
    min_allowed_weights: int
    max_weight_limit: float
    scaling_law_power: float
    synergy_scaling_law_power: float
    subnetwork_n: int
    max_n: int
    blocks_since_epoch: int
    tempo: int
    blocks_per_epoch: int
    modality: int
    connection_requirements: Dict[str, int] # netuid -> connection requirements
    emission_value: float
    