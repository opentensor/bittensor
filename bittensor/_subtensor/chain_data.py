# The MIT License (MIT)
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

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
    # mapping of coldkey to amount staked to this Neuron
    stake: Dict[str, Balance]
    total_stake: Balance
    rank: float
    emission: float
    incentive: float
    consensus: float
    trust: float
    dividends: float
    last_update: int
    validator_permit: bool
    weights: List[List[int]]
    bonds: List[List[int]]
    prometheus_info: 'PrometheusInfo'
    axon_info: 'AxonInfo'
    is_null: bool = False

    @classmethod
    def from_json(cls, json: Dict) -> 'NeuronInfo':
        r""" Returns a NeuronInfo object from a json dictionary.
        """
        return NeuronInfo(
            hotkey = bittensor.utils.u8_key_to_ss58(json['hotkey']['id']),
            coldkey = bittensor.utils.u8_key_to_ss58(json['coldkey']['id']),
            uid = json['uid'],
            netuid = json['netuid'],
            active = int(json['active']), # 0 or 1
            stake = { cls.__u8_key_to_ss58(stake[0]): Balance.from_rao(stake[1]) for stake in json['stake']},
            total_stake = Balance.from_rao(sum([stake for _, stake in json['stake']])),
            rank = json['rank'] / U16_MAX,
            emission = json['emission'] / RAOPERTAO,
            incentive = json['incentive'] / U16_MAX,
            consensus = json['consensus'] / U16_MAX,
            trust = json['trust'] / U16_MAX,
            dividends = json['dividends'] / U16_MAX,
            last_update = json['last_update'],
            validator_permit = json['validator_permit'],
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
            stake = {},
            total_stake = Balance.from_rao(0),
            rank = 0,
            emission = 0,
            incentive = 0,
            consensus = 0,
            trust = 0,
            dividends = 0,
            last_update = 0,
            validator_permit = False,
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
        # TODO: Legacy: remove?
        if neuron_dict['hotkey'] == '5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM':
            return NeuronInfo._null_neuron()
        else:
            neuron = NeuronInfo( **neuron_dict )
            # Fix?
            neuron.stake = { hk: Balance.from_rao(stake) for hk, stake in neuron.stake.items() }
            neuron.total_stake = Balance.from_rao(neuron.total_stake)
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

    @classmethod
    def from_json(cls, json: Dict) -> 'DelegateInfo':
        r""" Returns a DelegateInfo object from a json dictionary.
        """
        delegate_ss58 = bittensor.utils.u8_key_to_ss58(json['delegate_ss58']['id'])
        owner = bittensor.utils.u8_key_to_ss58(json['owner_ss58']['id'])
        take = bittensor.utils.U16_NORMALIZED_FLOAT(json['take'])
        nominators = [
            (bittensor.utils.u8_key_to_ss58(nom[0]['id']), Balance.from_rao(nom[1]))
            for nom in json['nominators']
        ]
        total_stake = sum([nom[1] for nom in nominators])

        return DelegateInfo(
            hotkey_ss58=delegate_ss58,
            take = take,
            total_stake=total_stake,
            nominators=nominators,
            owner_ss58=owner
        )


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
    modality: int
    connection_requirements: Dict[str, int] # netuid -> connection requirements
    emission_value: float

    @classmethod
    def from_json(cls, json: Dict) -> 'SubnetInfo':
        r""" Returns a SubnetInfo object from a json dictionary.
        """
        return SubnetInfo(
            netuid = json['netuid'],
            rho = json['rho'],
            kappa = json['kappa'],
            difficulty = json['difficulty'],
            immunity_period = json['immunity_period'],
            validator_batch_size = json['validator_batch_size'],
            validator_sequence_length = json['validator_sequence_length'],
            validator_epochs_per_reset = json['validator_epochs_per_reset'],
            validator_epoch_length = json['validator_epoch_length'],
            max_allowed_validators = json['max_allowed_validators'],
            min_allowed_weights = json['min_allowed_weights'],
            max_weight_limit = json['max_weights_limit'],
            scaling_law_power = json['scaling_law_power'],
            synergy_scaling_law_power= json['synergy_scaling_law_power'],
            subnetwork_n = json['subnetwork_n'],
            max_n = json['max_allowed_uids'],
            blocks_since_epoch = json['blocks_since_last_step'],
            tempo = json['tempo'],
            modality = json['network_modality'],
            connection_requirements= json['network_connect'],
            emission_value= json['emission_values'],
        )
    