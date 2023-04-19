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
from typing import List, Tuple, Dict, Optional, Any
import bittensor
from bittensor import Balance
import torch
from scalecodec.base import RuntimeConfiguration, ScaleBytes
from scalecodec.type_registry import load_type_registry_preset
from scalecodec.utils.ss58 import ss58_encode
from enum import Enum


custom_rpc_type_registry = {
    "types": {
        "SubnetInfo": {
            "type": "struct",
            "type_mapping": [
                ["netuid", "Compact<u16>"],
                ["rho", "Compact<u16>"],
                ["kappa", "Compact<u16>"],
                ["difficulty", "Compact<u64>"],
                ["immunity_period", "Compact<u16>"],
                ["validator_batch_size", "Compact<u16>"],
                ["validator_sequence_length", "Compact<u16>"],
                ["validator_epochs_per_reset", "Compact<u16>"],
                ["validator_epoch_length", "Compact<u16>"],
                ["max_allowed_validators", "Compact<u16>"],
                ["min_allowed_weights", "Compact<u16>"],
                ["max_weights_limit", "Compact<u16>"],
                ["scaling_law_power", "Compact<u16>"],
                ["synergy_scaling_law_power", "Compact<u16>"],
                ["subnetwork_n", "Compact<u16>"],
                ["max_allowed_uids", "Compact<u16>"],
                ["blocks_since_last_step", "Compact<u64>"],
                ["tempo", "Compact<u16>"],
                ["network_modality", "Compact<u16>"],
                ["network_connect", "Vec<[u16; 2]>"],
                ["emission_values", "Compact<u64>"],
                ["burn", "Compact<u64>"],
            ]
        },
        "DelegateInfo": {
            "type": "struct",
            "type_mapping": [
                ["delegate_ss58", "AccountId"],
                ["take", "Compact<u16>"],
                ["nominators", "Vec<(AccountId, Compact<u64>)>"],
                ["owner_ss58", "AccountId"],
                ["registrations", "Vec<Compact<u16>>"],
                ["validator_permits", "Vec<Compact<u16>>"],
                ["return_per_1000", "Compact<u64>"],
                ["total_daily_return", "Compact<u64>"],
            ],
        },
        "NeuronInfo": {
            "type": "struct",
            "type_mapping": [
                ["hotkey", "AccountId"],
                ["coldkey", "AccountId"],
                ["uid", "Compact<u16>"],
                ["netuid", "Compact<u16>"],
                ["active", "bool"],
                ["axon_info", "AxonInfo"],
                ["prometheus_info", "PrometheusInfo"],
                ["stake", "Vec<(AccountId, Compact<u64>)>"],
                ["rank", "Compact<u16>"],
                ["emission", "Compact<u64>"],
                ["incentive", "Compact<u16>"],
                ["consensus", "Compact<u16>"],
                ["trust", "Compact<u16>"],
                ["validator_trust", "Compact<u16>"],
                ["dividends", "Compact<u16>"],
                ["last_update", "Compact<u64>"],
                ["validator_permit", "bool"],
                ["weights", "Vec<(Compact<u16>, Compact<u16>)>"],
                ["bonds", "Vec<(Compact<u16>, Compact<u16>)>"],
                ["pruning_score", "Compact<u16>"]
            ],
        },
        "NeuronInfoLite": {
            "type": "struct",
            "type_mapping": [
                ["hotkey", "AccountId"],
                ["coldkey", "AccountId"],
                ["uid", "Compact<u16>"],
                ["netuid", "Compact<u16>"],
                ["active", "bool"],
                ["axon_info", "AxonInfo"],
                ["prometheus_info", "PrometheusInfo"],
                ["stake", "Vec<(AccountId, Compact<u64>)>"],
                ["rank", "Compact<u16>"],
                ["emission", "Compact<u64>"],
                ["incentive", "Compact<u16>"],
                ["consensus", "Compact<u16>"],
                ["trust", "Compact<u16>"],
                ["validator_trust", "Compact<u16>"],
                ["dividends", "Compact<u16>"],
                ["last_update", "Compact<u64>"],
                ["validator_permit", "bool"],
                ["pruning_score", "Compact<u16>"]
            ],
        },
        "AxonInfo": {
            "type": "struct",
            "type_mapping": [
                ["block", "u64"],
                ["version", "u32"],
                ["ip", "u128"],
                ["port", "u16"],
                ["ip_type", "u8"],
                ["protocol", "u8"],
                ["placeholder1", "u8"],
                ["placeholder2", "u8"],
            ],
        },
        "PrometheusInfo": {
            "type": "struct",
            "type_mapping": [
                ["block", "u64"],
                ["version", "u32"],
                ["ip", "u128"],
                ["port", "u16"],
                ["ip_type", "u8"],
            ],
        },
    }   
}

class ChainDataType(Enum):
    NeuronInfo = 1
    SubnetInfo = 2
    DelegateInfo = 3
    NeuronInfoLite = 4
    DelegatedInfo = 5

# Constants
RAOPERTAO = 1e9
U16_MAX = 65535
U64_MAX = 18446744073709551615

def from_scale_encoding( vec_u8: List[int], type_name: ChainDataType, is_vec: bool = False, is_option: bool = False ) -> Optional[Dict]:
    as_bytes = bytes(vec_u8)
    as_scale_bytes = ScaleBytes(as_bytes)
    rpc_runtime_config = RuntimeConfiguration()
    rpc_runtime_config.update_type_registry(load_type_registry_preset("legacy"))
    rpc_runtime_config.update_type_registry(custom_rpc_type_registry)

    type_string = type_name.name
    if type_name == ChainDataType.DelegatedInfo:
        # DelegatedInfo is a tuple of (DelegateInfo, Compact<u64>)
        type_string = f'({ChainDataType.DelegateInfo.name}, Compact<u64>)'
    if is_option:
        type_string = f'Option<{type_string}>'
    if is_vec:
        type_string = f'Vec<{type_string}>'

    obj = rpc_runtime_config.create_scale_object(
        type_string,
        data=as_scale_bytes
    )

    return obj.decode()

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
    stake: Balance
    # mapping of coldkey to amount staked to this Neuron
    stake_dict: Dict[str, Balance]
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
    weights: List[List[int]]
    bonds: List[List[int]]
    prometheus_info: 'PrometheusInfo'
    axon_info: 'AxonInfo'
    pruning_score: int
    is_null: bool = False

    @classmethod
    def fix_decoded_values(cls, neuron_info_decoded: Any) -> 'NeuronInfo':
        r""" Fixes the values of the NeuronInfo object.
        """
        neuron_info_decoded['hotkey'] = ss58_encode(neuron_info_decoded['hotkey'], bittensor.__ss58_format__)
        neuron_info_decoded['coldkey'] = ss58_encode(neuron_info_decoded['coldkey'], bittensor.__ss58_format__)
        stake_dict =  { ss58_encode( coldkey, bittensor.__ss58_format__): bittensor.Balance.from_rao(int(stake)) for coldkey, stake in neuron_info_decoded['stake'] }
        neuron_info_decoded['stake_dict'] = stake_dict
        neuron_info_decoded['stake'] = sum(stake_dict.values())
        neuron_info_decoded['total_stake'] = neuron_info_decoded['stake']
        neuron_info_decoded['weights'] = [[int(weight[0]), int(weight[1])] for weight in neuron_info_decoded['weights']]
        neuron_info_decoded['bonds'] = [[int(bond[0]), int(bond[1])] for bond in neuron_info_decoded['bonds']]
        neuron_info_decoded['rank'] = bittensor.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['rank'])
        neuron_info_decoded['emission'] = neuron_info_decoded['emission'] / RAOPERTAO
        neuron_info_decoded['incentive'] = bittensor.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['incentive'])
        neuron_info_decoded['consensus'] = bittensor.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['consensus'])
        neuron_info_decoded['trust'] = bittensor.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['trust'])
        neuron_info_decoded['validator_trust'] = bittensor.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['validator_trust'])
        neuron_info_decoded['dividends'] = bittensor.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['dividends'])
        neuron_info_decoded['prometheus_info'] = PrometheusInfo.fix_decoded_values(neuron_info_decoded['prometheus_info'])
        neuron_info_decoded['axon_info'] = AxonInfo.fix_decoded_values(neuron_info_decoded['axon_info'])

        return cls(**neuron_info_decoded)
    
    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> 'NeuronInfo':
        r""" Returns a NeuronInfo object from a vec_u8.
        """
        if len(vec_u8) == 0:
            return NeuronInfo._null_neuron()
        
        decoded = from_scale_encoding(vec_u8, ChainDataType.NeuronInfo)
        if decoded is None:
            return NeuronInfo._null_neuron()
        
        decoded = NeuronInfo.fix_decoded_values(decoded)

        return decoded
    
    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List['NeuronInfo']:
        r""" Returns a list of NeuronInfo objects from a vec_u8.
        """
        
        decoded_list = from_scale_encoding(vec_u8, ChainDataType.NeuronInfo, is_vec=True)
        if decoded_list is None:
            return []

        decoded_list = [NeuronInfo.fix_decoded_values(decoded) for decoded in decoded_list]
        return decoded_list


    @staticmethod
    def _null_neuron() -> 'NeuronInfo':
        neuron = NeuronInfo(
            uid = 0,
            netuid = 0,
            active =  0,
            stake = Balance.from_rao(0),
            stake_dict = {},
            total_stake = Balance.from_rao(0),
            rank = 0,
            emission = 0,
            incentive = 0,
            consensus = 0,
            trust = 0,
            validator_trust = 0,
            dividends = 0,
            last_update = 0,
            validator_permit = False,
            weights = [],
            bonds = [],
            prometheus_info = None,
            axon_info = None,
            is_null = True,
            coldkey = "000000000000000000000000000000000000000000000000",
            hotkey = "000000000000000000000000000000000000000000000000",
            pruning_score = 0,
        )
        return neuron

    @staticmethod
    def _neuron_dict_to_namespace(neuron_dict) -> 'NeuronInfo':
        # TODO: Legacy: remove?
        if neuron_dict['hotkey'] == '5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM':
            return NeuronInfo._null_neuron()
        else:
            neuron = NeuronInfo( **neuron_dict )
            neuron.stake_dict = { hk: Balance.from_rao(stake) for hk, stake in neuron.stake.items() }
            neuron.stake = Balance.from_rao(neuron.total_stake)
            neuron.total_stake = neuron.stake
            neuron.rank = neuron.rank / U16_MAX
            neuron.trust = neuron.trust / U16_MAX
            neuron.consensus = neuron.consensus / U16_MAX
            neuron.validator_trust = neuron.validator_trust / U16_MAX
            neuron.incentive = neuron.incentive / U16_MAX
            neuron.dividends = neuron.dividends / U16_MAX
            neuron.emission = neuron.emission / RAOPERTAO
                
            return neuron
        
@dataclass
class NeuronInfoLite:
    r"""
    Dataclass for neuron metadata, but without the weights and bonds.
    """
    hotkey: str
    coldkey: str
    uid: int
    netuid: int
    active: int    
    stake: Balance
    # mapping of coldkey to amount staked to this Neuron
    stake_dict: Dict[str, Balance]
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
    #weights: List[List[int]]
    #bonds: List[List[int]] No weights or bonds in lite version
    prometheus_info: 'PrometheusInfo'
    axon_info: 'AxonInfo'
    pruning_score: int
    is_null: bool = False

    @classmethod
    def fix_decoded_values(cls, neuron_info_decoded: Any) -> 'NeuronInfoLite':
        r""" Fixes the values of the NeuronInfoLite object.
        """
        neuron_info_decoded['hotkey'] = ss58_encode(neuron_info_decoded['hotkey'], bittensor.__ss58_format__)
        neuron_info_decoded['coldkey'] = ss58_encode(neuron_info_decoded['coldkey'], bittensor.__ss58_format__)
        stake_dict =  { ss58_encode( coldkey, bittensor.__ss58_format__): bittensor.Balance.from_rao(int(stake)) for coldkey, stake in neuron_info_decoded['stake'] }
        neuron_info_decoded['stake_dict'] = stake_dict
        neuron_info_decoded['stake'] = sum(stake_dict.values())
        neuron_info_decoded['total_stake'] = neuron_info_decoded['stake']
        # Don't need weights and bonds in lite version
        #neuron_info_decoded['weights'] = [[int(weight[0]), int(weight[1])] for weight in neuron_info_decoded['weights']]
        #neuron_info_decoded['bonds'] = [[int(bond[0]), int(bond[1])] for bond in neuron_info_decoded['bonds']]
        neuron_info_decoded['rank'] = bittensor.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['rank'])
        neuron_info_decoded['emission'] = neuron_info_decoded['emission'] / RAOPERTAO
        neuron_info_decoded['incentive'] = bittensor.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['incentive'])
        neuron_info_decoded['consensus'] = bittensor.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['consensus'])
        neuron_info_decoded['trust'] = bittensor.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['trust'])
        neuron_info_decoded['validator_trust'] = bittensor.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['validator_trust'])
        neuron_info_decoded['dividends'] = bittensor.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['dividends'])
        neuron_info_decoded['prometheus_info'] = PrometheusInfo.fix_decoded_values(neuron_info_decoded['prometheus_info'])
        neuron_info_decoded['axon_info'] = AxonInfo.fix_decoded_values(neuron_info_decoded['axon_info'])

        return cls(**neuron_info_decoded)
    
    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> 'NeuronInfoLite':
        r""" Returns a NeuronInfoLite object from a vec_u8.
        """
        if len(vec_u8) == 0:
            return NeuronInfoLite._null_neuron()
        
        decoded = from_scale_encoding(vec_u8, ChainDataType.NeuronInfoLite)
        if decoded is None:
            return NeuronInfoLite._null_neuron()
        
        decoded = NeuronInfoLite.fix_decoded_values(decoded)

        return decoded
    
    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List['NeuronInfoLite']:
        r""" Returns a list of NeuronInfoLite objects from a vec_u8.
        """
        
        decoded_list = from_scale_encoding(vec_u8, ChainDataType.NeuronInfoLite, is_vec=True)
        if decoded_list is None:
            return []

        decoded_list = [NeuronInfoLite.fix_decoded_values(decoded) for decoded in decoded_list]
        return decoded_list


    @staticmethod
    def _null_neuron() -> 'NeuronInfoLite':
        neuron = NeuronInfoLite(
            uid = 0,
            netuid = 0,
            active =  0,
            stake = Balance.from_rao(0),
            stake_dict = {},
            total_stake = Balance.from_rao(0),
            rank = 0,
            emission = 0,
            incentive = 0,
            consensus = 0,
            trust = 0,
            validator_trust = 0,
            dividends = 0,
            last_update = 0,
            validator_permit = False,
            #weights = [], // No weights or bonds in lite version
            #bonds = [],
            prometheus_info = None,
            axon_info = None,
            is_null = True,
            coldkey = "000000000000000000000000000000000000000000000000",
            hotkey = "000000000000000000000000000000000000000000000000",
            pruning_score = 0,
        )
        return neuron

    @staticmethod
    def _neuron_dict_to_namespace(neuron_dict) -> 'NeuronInfoLite':
        # TODO: Legacy: remove?
        if neuron_dict['hotkey'] == '5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM':
            return NeuronInfoLite._null_neuron()
        else:
            neuron = NeuronInfoLite( **neuron_dict )
            neuron.stake = Balance.from_rao(neuron.total_stake)
            neuron.stake_dict = { hk: Balance.from_rao(stake) for hk, stake in neuron.stake.items() }
            neuron.total_stake = neuron.stake
            neuron.rank = neuron.rank / U16_MAX
            neuron.trust = neuron.trust / U16_MAX
            neuron.consensus = neuron.consensus / U16_MAX
            neuron.validator_trust = neuron.validator_trust / U16_MAX
            neuron.incentive = neuron.incentive / U16_MAX
            neuron.dividends = neuron.dividends / U16_MAX
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
    def fix_decoded_values(cls, axon_info_decoded: Dict) -> 'AxonInfo':
        r""" Returns an AxonInfo object from an axon_info_decoded dictionary.
        """
        axon_info_decoded['ip'] = bittensor.utils.networking.int_to_ip(int(axon_info_decoded['ip']))
                                                                       
        return cls(**axon_info_decoded)

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
    def fix_decoded_values(cls, prometheus_info_decoded: Dict) -> 'PrometheusInfo':
        r""" Returns a PrometheusInfo object from a prometheus_info_decoded dictionary.
        """
        prometheus_info_decoded['ip'] = bittensor.utils.networking.int_to_ip(int(prometheus_info_decoded['ip']))
        
        return cls(**prometheus_info_decoded)
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
    validator_permits: List[int] # List of subnets that the delegate is allowed to validate on
    registrations: List[int] # List of subnets that the delegate is registered on
    return_per_1000: bittensor.Balance # Return per 1000 tao of the delegate over a day
    total_daily_return: bittensor.Balance # Total daily return of the delegate

    @classmethod
    def fix_decoded_values(cls, decoded: Any) -> 'DelegateInfo':
        r""" Fixes the decoded values.
        """
        
        return cls(
            hotkey_ss58 = ss58_encode(decoded['delegate_ss58'], bittensor.__ss58_format__),
            owner_ss58 = ss58_encode(decoded['owner_ss58'], bittensor.__ss58_format__),
            take = bittensor.utils.U16_NORMALIZED_FLOAT(decoded['take']),
            nominators = [
                (ss58_encode(nom[0], bittensor.__ss58_format__), Balance.from_rao(nom[1]))
                for nom in decoded['nominators']
            ],
            total_stake = Balance.from_rao(sum([nom[1] for nom in decoded['nominators']])),
            validator_permits = decoded['validator_permits'],
            registrations = decoded['registrations'],
            return_per_1000 = bittensor.Balance.from_rao(decoded['return_per_1000']),
            total_daily_return = bittensor.Balance.from_rao(decoded['total_daily_return']),
        )

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional['DelegateInfo']:
        r""" Returns a DelegateInfo object from a vec_u8.
        """
        if len(vec_u8) == 0:
            return None
        
        decoded = from_scale_encoding(vec_u8, ChainDataType.DelegateInfo)

        if decoded is None:
            return None
        
        decoded = DelegateInfo.fix_decoded_values(decoded)

        return decoded
    
    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List['DelegateInfo']:
        r""" Returns a list of DelegateInfo objects from a vec_u8.
        """
        decoded = from_scale_encoding(vec_u8, ChainDataType.DelegateInfo, is_vec=True)

        if decoded is None:
            return []
        
        decoded = [DelegateInfo.fix_decoded_values(d) for d in decoded]

        return decoded
    
    @classmethod
    def delegated_list_from_vec_u8(cls, vec_u8: List[int]) -> List[Tuple['DelegateInfo', Balance]]:
        r""" Returns a list of Tuples of DelegateInfo objects, and Balance, from a vec_u8.
        This is the list of delegates that the user has delegated to, and the amount of stake delegated.
        """
        decoded = from_scale_encoding(vec_u8, ChainDataType.DelegatedInfo, is_vec=True)

        if decoded is None:
            return []
        
        decoded = [(DelegateInfo.fix_decoded_values(d), Balance.from_rao(s)) for d, s in decoded]

        return decoded

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
    # netuid -> topk percentile prunning score requirement (u16:MAX normalized.)
    connection_requirements: Dict[str, float]
    emission_value: float
    burn: Balance

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional['SubnetInfo']:
        r""" Returns a SubnetInfo object from a vec_u8.
        """
        if len(vec_u8) == 0:
            return None

        decoded = from_scale_encoding(vec_u8, ChainDataType.SubnetInfo)

        if decoded is None:
            return None
        
        return SubnetInfo.fix_decoded_values(decoded)
    
    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List['SubnetInfo']:
        r""" Returns a list of SubnetInfo objects from a vec_u8.
        """
        decoded = from_scale_encoding(vec_u8, ChainDataType.SubnetInfo, is_vec=True, is_option=True)

        if decoded is None:
            return []
        
        decoded = [SubnetInfo.fix_decoded_values(d) for d in decoded]

        return decoded

    @classmethod
    def fix_decoded_values(cls, decoded: Dict) -> 'SubnetInfo':
        r""" Returns a SubnetInfo object from a decoded SubnetInfo dictionary.
        """
        return SubnetInfo(
            netuid = decoded['netuid'],
            rho = decoded['rho'],
            kappa = decoded['kappa'],
            difficulty = decoded['difficulty'],
            immunity_period = decoded['immunity_period'],
            validator_batch_size = decoded['validator_batch_size'],
            validator_sequence_length = decoded['validator_sequence_length'],
            validator_epochs_per_reset = decoded['validator_epochs_per_reset'],
            validator_epoch_length = decoded['validator_epoch_length'],
            max_allowed_validators = decoded['max_allowed_validators'],
            min_allowed_weights = decoded['min_allowed_weights'],
            max_weight_limit = decoded['max_weights_limit'],
            scaling_law_power = decoded['scaling_law_power'],
            synergy_scaling_law_power= decoded['synergy_scaling_law_power'],
            subnetwork_n = decoded['subnetwork_n'],
            max_n = decoded['max_allowed_uids'],
            blocks_since_epoch = decoded['blocks_since_last_step'],
            tempo = decoded['tempo'],
            modality = decoded['network_modality'],
            connection_requirements = {
                str(int(netuid)): bittensor.utils.U16_NORMALIZED_FLOAT(int(req)) for netuid, req in decoded['network_connect']
            },
            emission_value= decoded['emission_values'],
            burn = Balance(0)#Balance.from_rao(decoded['burn'])
        )
    
    def to_parameter_dict( self ) -> 'torch.nn.ParameterDict':
        r""" Returns a torch tensor of the subnet info.
        """
        return torch.nn.ParameterDict( 
            self.__dict__
        )
    
    @classmethod
    def from_parameter_dict( cls, parameter_dict: 'torch.nn.ParameterDict' ) -> 'SubnetInfo':
        r""" Returns a SubnetInfo object from a torch parameter_dict.
        """
        return cls( **dict(parameter_dict) )
