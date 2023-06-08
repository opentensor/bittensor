# The MIT License (MIT)
# Copyright © 2022-2023 Opentensor Foundation

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

from random import randint
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
from unittest.mock import MagicMock
from dataclasses import dataclass

import bittensor
from bittensor.utils import RAOPERTAO, U16_NORMALIZED_FLOAT
from bittensor.utils.registration import POWSolution

from .chain_data import (NeuronInfo, NeuronInfoLite, PrometheusInfo,
                         SubnetInfo, axon_info)
from .errors import *
from .subtensor_impl import Subtensor

BlockNumber = int

class AxonInfoDict(TypedDict):
    block: int
    version: int
    ip: int # integer representation of ip address
    port: int
    ip_type: int
    protocol: int
    placeholder1: int # placeholder for future use
    placeholder2: int

class PrometheusInfoDict(TypedDict):
    block: int
    version: int
    ip: int # integer representation of ip address
    port: int
    ip_type: int

@dataclass
class MockSubtensorValue:
    value: Any

class MockMapResult:
    records: Optional[List[Tuple[MockSubtensorValue, MockSubtensorValue]]]

    def __init__(self, records: Optional[List[Tuple[Union[Any, MockSubtensorValue], Union[Any, MockSubtensorValue]]]] = None):
        _records = [
                (MockSubtensorValue( value=record[0] ), MockSubtensorValue( value=record[1] )) 
                    # Make sure record is a tuple of MockSubtensorValue (dict with value attr)
                    if not (isinstance(record, tuple) and all(isinstance(item, dict) and hasattr(item, 'value') for item in record))
                else record 
            for record in records 
        ]
        
        self.records = _records

    def __iter__(self):
        return iter(self.records)

class MockSystemState(TypedDict):
    Account: Dict[str, Dict[str, 'bittensor.Balance']] # address -> block -> balance

class MockSubtensorState(TypedDict):
    Rho: Dict[str, Dict[BlockNumber, int]] # netuid -> block -> rho
    Kappa: Dict[str, Dict[BlockNumber, int]] # netuid -> block -> kappa
    Difficulty: Dict[str, Dict[BlockNumber, int]] # netuid -> block -> difficulty
    ImmunityPeriod: Dict[str, Dict[BlockNumber, int]] # netuid -> block -> immunity_period
    ValidatorBatchSize: Dict[str, Dict[BlockNumber, int]] # netuid -> block -> validator_batch_size
    Active: Dict[str, Dict[BlockNumber, bool]] # (netuid, uid), block -> active

    NetworksAdded: Dict[str, Dict[BlockNumber, bool]] # netuid -> block -> added

class MockChainState(TypedDict):
    System: MockSystemState
    SubtensorModule: MockSubtensorState
    
class MockSubtensor(Subtensor):
    """
    A Mock Subtensor class for running tests. 
    This should mock only methods that make queries to the chain.
    e.g. We mock `Subtensor.query_subtensor` instead of all query methods.

    This class will also store a local (mock) state of the chain.
    """
    chain_state: MockChainState
    block_number: int

    @classmethod
    def reset(cls) -> None:
        bittensor.__GLOBAL_MOCK_STATE__.clear()

        _ = cls()

    def setup(self) -> None:
        if not hasattr(self, 'chain_state') or getattr(self, 'chain_state') is None:
            self.chain_state = {
                'System': {
                    'Account': {}
                },
                'Balances': {
                    'ExistentialDeposit': {
                        0: 500
                    },
                },
                'SubtensorModule': {
                    'NetworksAdded': {},
                    'Rho': {},
                    'Kappa': {},
                    'Difficulty': {},
                    'ImmunityPeriod': {},
                    'ValidatorBatchSize': {},
                    'ValidatorSequenceLen': {},
                    'ValidatorEpochsPerReset': {},
                    'ValidatorEpochLen': {},
                    'MaxAllowedValidators': {},
                    'MinAllowedWeights': {},
                    'MaxWeightsLimit': {},
                    'SynergyScalingLawPower': {},
                    'ScalingLawPower': {},
                    'SubnetworkN': {},
                    'MaxAllowedUids': {},
                    'NetworkModality': {},
                    'BlocksSinceLastStep': {},
                    'Tempo': {},
                    'NetworkConnect': {},
                    'EmissionValue': {},
                    'Burn': {},

                    'Active': {},

                    'UIDs': {},
                    'Keys': {},
                    'Owner': {},
                    'LastUpdate': {},
                    
                    'Rank': {},
                    'Emission': {},
                    'Incentive': {},
                    'Consensus': {},
                    'Trust': {},
                    'ValidatorTrust': {},
                    'Dividends': {},
                    'PruningScores': {},
                    'ValidatorPermit': {},

                    'Weights': {},
                    'Bonds': {},
                    
                    'Stake': {},
                    'TotalStake': {
                        0: 0
                    },

                    'Delegates': {},

                    'Axons': {},
                    'Prometheus': {},
                },
            }

            self.block_number = 0

            self.network = 'mock'
            self.chain_endpoint = 'mock_endpoint'
            self.substrate = MagicMock()

    def __init__(self) -> None:
        self.__dict__ = bittensor.__GLOBAL_MOCK_STATE__
        
        if not hasattr(self, 'chain_state') or getattr(self, 'chain_state') is None:
            self.setup()
        

    def create_subnet( self, netuid: int ) -> None:
        subtensor_state = self.chain_state['SubtensorModule']
        if netuid not in subtensor_state['NetworksAdded']:
            # Per Subnet
            subtensor_state['Rho'][netuid] = {}
            subtensor_state['Rho'][netuid][0] = 10
            subtensor_state['Kappa'][netuid] = {}
            subtensor_state['Kappa'][netuid][0] = 32_767
            subtensor_state['Difficulty'][netuid] = {}
            subtensor_state['Difficulty'][netuid][0] = 10_000_000
            subtensor_state['ImmunityPeriod'][netuid] = {}
            subtensor_state['ImmunityPeriod'][netuid][0] = 4096
            subtensor_state['ValidatorBatchSize'][netuid] = {}
            subtensor_state['ValidatorBatchSize'][netuid][0] = 32
            subtensor_state['ValidatorSequenceLen'][netuid] = {}
            subtensor_state['ValidatorSequenceLen'][netuid][0] = 256
            subtensor_state['ValidatorEpochsPerReset'][netuid] = {}
            subtensor_state['ValidatorEpochsPerReset'][netuid][0] = 60
            subtensor_state['ValidatorEpochLen'][netuid] = {}
            subtensor_state['ValidatorEpochLen'][netuid][0] = 100
            subtensor_state['MaxAllowedValidators'][netuid] = {}
            subtensor_state['MaxAllowedValidators'][netuid][0] = 128
            subtensor_state['MinAllowedWeights'][netuid] = {}
            subtensor_state['MinAllowedWeights'][netuid][0] = 1024
            subtensor_state['MaxWeightsLimit'][netuid] = {}
            subtensor_state['MaxWeightsLimit'][netuid][0] = 1_000
            subtensor_state['SynergyScalingLawPower'][netuid] = {}
            subtensor_state['SynergyScalingLawPower'][netuid][0] = 50
            subtensor_state['ScalingLawPower'][netuid] = {}
            subtensor_state['ScalingLawPower'][netuid][0] = 50
            subtensor_state['SubnetworkN'][netuid] = {}
            subtensor_state['SubnetworkN'][netuid][0] = 0
            subtensor_state['MaxAllowedUids'][netuid] = {}
            subtensor_state['MaxAllowedUids'][netuid][0] = 4096
            subtensor_state['NetworkModality'][netuid] = {}
            subtensor_state['NetworkModality'][netuid][0] = 0
            subtensor_state['BlocksSinceLastStep'][netuid] = {}
            subtensor_state['BlocksSinceLastStep'][netuid][0] = 0
            subtensor_state['Tempo'][netuid] = {}
            subtensor_state['Tempo'][netuid][0] = 99
            subtensor_state['NetworkConnect'][netuid] = {}
            subtensor_state['NetworkConnect'][netuid][0] = {}
            subtensor_state['EmissionValue'][netuid] = {}
            subtensor_state['EmissionValue'][netuid][0] = 0
            subtensor_state['Burn'][netuid] = {}
            subtensor_state['Burn'][netuid][0] = 0

            # Per-UID/Hotkey

            subtensor_state['UIDs'][netuid] = {}
            subtensor_state['Keys'][netuid] = {}
            subtensor_state['Owner'][netuid] = {}
            
            subtensor_state['LastUpdate'][netuid] = {}
            subtensor_state['Active'][netuid] = {}
            subtensor_state['Rank'][netuid] = {}
            subtensor_state['Emission'][netuid] = {}
            subtensor_state['Incentive'][netuid] = {}
            subtensor_state['Consensus'][netuid] = {}
            subtensor_state['Trust'][netuid] = {}
            subtensor_state['ValidatorTrust'][netuid] = {}
            subtensor_state['Dividends'][netuid] = {}
            subtensor_state['PruningScores'][netuid] = {}
            subtensor_state['PruningScores'][netuid][0] = {}
            subtensor_state['ValidatorPermit'][netuid] = {}

            subtensor_state['Weights'][netuid] = {}
            subtensor_state['Bonds'][netuid] = {}

            subtensor_state['Axons'][netuid] = {}
            subtensor_state['Prometheus'][netuid] = {}
            subtensor_state['Stake'][netuid] = {}

            subtensor_state['NetworksAdded'][netuid] = {}
            subtensor_state['NetworksAdded'][netuid][0] = True

        else:
            raise Exception("Subnet already exists")
    
    def _register_neuron(
        self,
        netuid: int,
        hotkey: str, 
        coldkey: str,
    ) -> int:
        subtensor_state = self.chain_state['SubtensorModule']
        if netuid not in subtensor_state['NetworksAdded']:
            raise Exception("Subnet does not exist")

        subnetwork_n = self._get_most_recent_storage(subtensor_state['SubnetworkN'][netuid])
        
        if subnetwork_n > 0 and any(self._get_most_recent_storage(subtensor_state['Keys'][netuid][uid]) == hotkey for uid in range(subnetwork_n)):
            # already_registered
            raise Exception("Hotkey already registered")
        else:
            # Not found
            if subnetwork_n >= self._get_most_recent_storage(subtensor_state['MaxAllowedUids'][netuid]):
                # Subnet full, replace neuron randomly
                uid = randint(0, subnetwork_n-1)
            else:
                # Subnet not full, add new neuron
                # Append as next uid and increment subnetwork_n
                uid = subnetwork_n
                subtensor_state['SubnetworkN'][netuid][self.block_number] = subnetwork_n 

            subtensor_state['Stake'][hotkey] = {}
            subtensor_state['Stake'][hotkey][coldkey] = {}
            subtensor_state['Stake'][hotkey][coldkey][self.block_number] = bittensor.Balance(0)
            
            subtensor_state['UIDs'][netuid][hotkey] = {}
            subtensor_state['UIDs'][netuid][hotkey][self.block_number] = uid

            subtensor_state['Keys'][netuid][uid] = {}
            subtensor_state['Keys'][netuid][uid][self.block_number] = hotkey

            subtensor_state['Owner'][hotkey] = {}
            subtensor_state['Owner'][hotkey][self.block_number] = coldkey

            subtensor_state['LastUpdate'][netuid][uid] = {}
            subtensor_state['LastUpdate'][netuid][uid][self.block_number] = self.block_number
            
            subtensor_state['Rank'][netuid][uid] = {}
            subtensor_state['Rank'][netuid][uid][self.block_number] = 0.0

            subtensor_state['Emission'][netuid][uid] = {}
            subtensor_state['Emission'][netuid][uid][self.block_number] = 0.0

            subtensor_state['Incentive'][netuid][uid] = {}
            subtensor_state['Incentive'][netuid][uid][self.block_number] = 0.0

            subtensor_state['Consensus'][netuid][uid] = {}
            subtensor_state['Consensus'][netuid][uid][self.block_number] = 0.0

            subtensor_state['Trust'][netuid][uid] = {}
            subtensor_state['Trust'][netuid][uid][self.block_number] = 0.0

            subtensor_state['ValidatorTrust'][netuid][uid] = {}
            subtensor_state['ValidatorTrust'][netuid][uid][self.block_number] = 0.0

            subtensor_state['Dividends'][netuid][uid] = {}
            subtensor_state['Dividends'][netuid][uid][self.block_number] = 0.0

            subtensor_state['PruningScores'][netuid][uid] = {}
            subtensor_state['PruningScores'][netuid][uid][self.block_number] = 0.0

            subtensor_state['ValidatorPermit'][netuid][uid] = {}
            subtensor_state['ValidatorPermit'][netuid][uid][self.block_number] = False

            subtensor_state['Weights'][netuid][uid] = {}
            subtensor_state['Weights'][netuid][uid][self.block_number] = []

            subtensor_state['Bonds'][netuid][uid] = {}
            subtensor_state['Bonds'][netuid][uid][self.block_number] = []

            subtensor_state['Axons'][netuid][hotkey] = {}
            subtensor_state['Axons'][netuid][hotkey][self.block_number] = {}

            subtensor_state['Prometheus'][netuid][hotkey] = {}
            subtensor_state['Prometheus'][netuid][hotkey][self.block_number] = {}
            
            return uid

    def force_register_neuron(
        self,
        netuid: int,
        hotkey: str, 
        coldkey: str,
        stake: Union['bittensor.Balance', float] = bittensor.Balance(0),
    ) -> int:
        """
        Force register a neuron on the mock chain, returning the UID.
        """
        if isinstance(stake, float):
            stake = bittensor.Balance.from_tao(stake)

        subtensor_state = self.chain_state['SubtensorModule']
        if netuid not in subtensor_state['NetworksAdded']:
            raise Exception("Subnet does not exist")

        self._register_neuron(
            netuid=netuid,
            hotkey=hotkey,
            coldkey=coldkey,
        )

        subtensor_state['TotalStake'][self.block_number] = self._get_most_recent_storage(subtensor_state['TotalStake']) + stake
        subtensor_state['Stake'][hotkey][coldkey][self.block_number] = stake
            

    def force_set_balance(
        self,
        ss58_address: str,
        balance: Union['bittensor.Balance', float] = bittensor.Balance(0),
        stake: Union['bittensor.Balance', float] = bittensor.Balance(0),
    ) -> None:
        if isinstance(stake, float):
            stake = bittensor.Balance.from_tao(stake)
            
        if isinstance(balance, float):
            balance = bittensor.Balance.from_tao(balance)

        if ss58_address not in self.chain_state['System']['Account']:
            self.chain_state['System']['Account'][ss58_address] = {}

        self.chain_state['System']['Account'][ss58_address][self.block_number] = {
            'data': {
                'free': balance.rao,
            }
        }
        
        
    def do_block_step( self ) -> None:
        self.block_number += 1

        # Doesn't do epoch
        subtensor_state = self.chain_state['SubtensorModule']
        for subnet in subtensor_state['NetworksAdded']:
            subtensor_state['BlocksSinceLastStep'][subnet][self.block_number] = self._get_most_recent_storage(subtensor_state['BlocksSinceLastStep'][subnet]) + 1

    def query_subtensor( self, name: str, block: Optional[int] = None, params: Optional[List[object]] = [] ) -> Optional[object]:
        if block:
            if self.block_number < block:
                raise Exception("Cannot query block in the future")
            
        else:
            block = self.block_number

        state = self.chain_state['SubtensorModule'][name]
        if state is not None:
            # Use prefix
            if len(params) > 0:
                while state is not None and len(params) > 0:
                    state = state.get(params.pop(0), None)
                    if state is None:
                        return None
                    
            # Use block
            state_at_block = state.get(block, None)
            while state_at_block is None and block > 0:
                block -= 1
                state_at_block = self.state.get(block, None)
            if state_at_block is not None:
                return SimpleNamespace(
                    value=state_at_block
                )

            return None
        else:
            return None
            
    def query_map_subtensor( self, name: str, block: Optional[int] = None, params: Optional[List[object]] = [] ) -> Optional[MockMapResult]:
        """
        Note: Double map requires one param
        """
        if block:
            if self.block_number < block:
                raise Exception("Cannot query block in the future")
            
        else:
            block = self.block_number

        state = self.chain_state['SubtensorModule'][name]
        if state is not None:
            # Use prefix
            if len(params) > 0:
                while state is not None and len(params) > 0:
                    state = state.get(params.pop(0), None)
                    if state is None:
                        return MockMapResult([])
            
            # Check if single map or double map
            if len(state.keys()) == 0:
                return MockMapResult([])
        
            inner = list(state.values())[0]
            # Should have at least one key
            if len(inner.keys()) == 0:
                raise Exception("Invalid state")
            
            # Check if double map
            if isinstance(list(inner.values())[0], dict):
                # is double map
                raise ChainQueryError("Double map requires one param")
            
            # Iterate over each key and add value to list, max at block
            records = []
            for key in state:
                result = self._get_most_recent_storage(state[key], block)
                if result is None:
                    continue # Skip if no result for this key at `block` or earlier

                records.append((key, result))

            return MockMapResult(records)
        else:
            return MockMapResult([])

    def query_constant( self, module_name: str, constant_name: str, block: Optional[int] = None ) -> Optional[object]:
        if block:
            if self.block_number < block:
                raise Exception("Cannot query block in the future")
            
        else:
            block = self.block_number

        state = self.chain_state.get(module_name, None)
        if state is not None:
            if constant_name in state:
                state = state[constant_name]
            else:
                return None
                    
            # Use block
            state_at_block = self._get_most_recent_storage(state, block)
            if state_at_block is not None:
                return SimpleNamespace(value=state_at_block)

            return state_at_block # Can be None
        else:
            return None

    def get_current_block( self ) -> int:
        return self.block_number

    # ==== Balance RPC methods ====

    def get_balance(self, address: str, block: int = None) -> 'bittensor.Balance':
        if block:
            if self.block_number < block:
                raise Exception("Cannot query block in the future")
            
        else:
            block = self.block_number

        state = self.chain_state['System']['Account']
        if state is not None:
            if address in state:
                state = state[address]
            else:
                return bittensor.Balance(0)
                    
            # Use block
            state_at_block = self._get_most_recent_storage(state, block)
            return state_at_block # Can be None
        else:
            return None

    def get_balances(self, block: int = None) -> Dict[str, 'bittensor.Balance']:
        balances = {}
        for address in self.chain_state['System']['Account']:
            balances[address] = self.get_balance(address, block)

    # ==== Neuron RPC methods ====

    def neuron_for_uid( self, uid: int, netuid: int, block: Optional[int] = None ) -> Optional[NeuronInfo]:
        if block:
            if self.block_number < block:
                raise Exception("Cannot query block in the future")
            
        else:
            block = self.block_number

        if netuid not in self.chain_state['SubtensorModule']['NetworksAdded']:
            return None
        
        neuron_info = self._neuron_subnet_exists( uid, netuid, block )
        if neuron_info is None:
            return None
        
        else:
            return neuron_info

    def neurons(self, netuid: int, block: Optional[int] = None ) -> List[NeuronInfo]:
        if netuid not in self.chain_state['SubtensorModule']['NetworksAdded']:
            raise Exception("Subnet does not exist")
        
        neurons = []
        subnet_n = self._get_most_recent_storage( self.chain_state['SubtensorModule']['SubnetworkN'][netuid], block )
        for uid in range( subnet_n ):
            neuron_info = self.neuron_for_uid( uid, netuid, block )
            if neuron_info is not None:
                neurons.append(neuron_info)

        return neurons

    @staticmethod
    def _get_most_recent_storage( storage: Dict[BlockNumber, Any], block_number: Optional[int] = None ) -> Any:
        if block_number is None:
            items = list(storage.items())
            items.sort(key=lambda x: x[0], reverse=True)
            return items[0][1]
        
        else:
            while block_number >= 0:
                if block_number in storage:
                    return storage[block_number]
                
                block_number -= 1
            
            return None
        
    def _get_axon_info( self, netuid: int, hotkey: str, block: Optional[int] = None ) -> Optional[AxonInfoDict]:
        # Axons [netuid][hotkey][block_number]
        subtensor_state = self.chain_state['SubtensorModule']
        if netuid not in subtensor_state['Axons']:
            return None
        
        if hotkey not in subtensor_state['Axons'][netuid]:
            return None
        
        return self._get_most_recent_storage(subtensor_state['Axons'][netuid][hotkey], block)

    def _get_prometheus_info( self, netuid: int, hotkey: str, block: Optional[int] = None ) -> Optional[PrometheusInfoDict]:
        subtensor_state = self.chain_state['SubtensorModule']
        if netuid not in subtensor_state['Prometheus']:
            return None
        
        if hotkey not in subtensor_state['Prometheus'][netuid]:
            return None
        
        return self._get_most_recent_storage(subtensor_state['Prometheus'][netuid][hotkey], block)

    def _neuron_subnet_exists( self, uid: int, netuid: int, block: Optional[int] = None ) -> Optional[NeuronInfo]:
        subtensor_state = self.chain_state['SubtensorModule']
        if netuid not in subtensor_state['NetworksAdded']:
            return None
        
        if self._get_most_recent_storage(subtensor_state['SubnetworkN'][netuid]) <= uid:
            return None
        
        hotkey = self._get_most_recent_storage(subtensor_state['Keys'][netuid][uid])
        if hotkey is None:
            return None


        axon_info_ = self._get_axon_info( netuid, hotkey, block )

        prometheus_info = self._get_prometheus_info( netuid, hotkey, block )


        coldkey = self._get_most_recent_storage(subtensor_state['Owner'][hotkey], block)
        active = self._get_most_recent_storage(subtensor_state['Active'][netuid][uid], block)
        rank = self._get_most_recent_storage(subtensor_state['Rank'][netuid][uid], block)
        emission = self._get_most_recent_storage(subtensor_state['Emission'][netuid][uid], block)
        incentive = self._get_most_recent_storage(subtensor_state['Incentive'][netuid][uid], block)
        consensus = self._get_most_recent_storage(subtensor_state['Consensus'][netuid][uid], block)
        trust = self._get_most_recent_storage(subtensor_state['Trust'][netuid][uid], block)
        validator_trust = self._get_most_recent_storage(subtensor_state['ValidatorTrust'][netuid][uid], block)
        dividends = self._get_most_recent_storage(subtensor_state['Dividends'][netuid][uid], block)
        pruning_score = self._get_most_recent_storage(subtensor_state['PruningScores'][netuid][uid], block)
        last_update = self._get_most_recent_storage(subtensor_state['LastUpdate'][netuid][uid], block)
        validator_permit = self._get_most_recent_storage(subtensor_state['ValidatorPermit'][netuid][uid], block)
        
        weights = self._get_most_recent_storage(subtensor_state['Weights'][netuid][uid], block)
        bonds = self._get_most_recent_storage(subtensor_state['Bonds'][netuid][uid], block)

        stake_dict = [(coldkey, bittensor.Balance.from_rao(self._get_most_recent_storage(
            subtensor_state['Stake'][hotkey][coldkey], block
        ))) for coldkey in subtensor_state['Stake'][hotkey]]

        stake = sum(stake_dict.values())


        weights = [[int(weight[0]), int(weight[1])] for weight in weights]
        bonds = [[int(bond[0]), int(bond[1])] for bond in bonds]
        rank = U16_NORMALIZED_FLOAT(rank)
        emission = emission / RAOPERTAO
        incentive = U16_NORMALIZED_FLOAT(incentive)
        consensus = U16_NORMALIZED_FLOAT(consensus)
        trust = U16_NORMALIZED_FLOAT(trust)
        validator_trust = U16_NORMALIZED_FLOAT(validator_trust)
        dividends = U16_NORMALIZED_FLOAT(dividends)
        prometheus_info = PrometheusInfo.fix_decoded_values(prometheus_info)
        axon_info_ = axon_info.from_neuron_info( {
            'hotkey': hotkey,
            'coldkey': coldkey,
            'axon_info': axon_info_,
        })

        neuron_info = NeuronInfo(
            hotkey = hotkey,
            coldkey = coldkey,
            uid = uid,
            netuid = netuid,
            active = active,
            rank = rank,
            emission = emission,
            incentive = incentive,
            consensus = consensus,
            trust = trust,
            validator_trust = validator_trust,
            dividends = dividends,
            pruning_score = pruning_score,
            last_update = last_update,
            validator_permit = validator_permit,
            stake = stake,
            stake_dict = stake_dict,
            total_stake=stake,
            prometheus_info=prometheus_info,
            axon_info=axon_info_,
            weights = weights,
            bonds = bonds,
            is_null=False,
        )

        return neuron_info


    def neuron_for_uid_lite( self, uid: int, netuid: int, block: Optional[int] = None ) -> Optional[NeuronInfoLite]:
        if block:
            if self.block_number < block:
                raise Exception("Cannot query block in the future")
            
        else:
            block = self.block_number

        if netuid not in self.chain_state['SubtensorModule']['NetworksAdded']:
            raise Exception("Subnet does not exist")
        
        neuron_info = self._neuron_subnet_exists( uid, netuid, block )
        if neuron_info is None:
            return None
        
        else:
            del neuron_info['weights']
            del neuron_info['bonds']

            neuron_info_lite = NeuronInfoLite(
                **neuron_info
            )
            return neuron_info_lite

    def neurons_lite(self, netuid: int, block: Optional[int] = None ) -> List[NeuronInfoLite]:
        if netuid not in self.chain_state['SubtensorModule']['NetworksAdded']:
            raise Exception("Subnet does not exist")
        
        neurons = []
        subnet_n = self._get_most_recent_storage( self.chain_state['SubtensorModule']['SubnetworkN'][netuid] )
        for uid in range(subnet_n):
            neuron_info = self.neuron_for_uid_lite( uid, netuid, block )
            if neuron_info is not None:
                neurons.append(neuron_info)

        return neurons
    
    # Extrinsics
    def do_delegation(
        self,
        wallet: 'bittensor.Wallet',
        delegate_ss58: str,
        amount: 'bittensor.Balance',
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        # Check if delegate
        if not self.is_hotkey_delegate(
            hotkey_ss58 = delegate_ss58
        ):
            raise Exception("Not a delegate")
        
        # do stake
        self.do_stake(
            wallet = wallet,
            hotkey_ss58 = delegate_ss58,
            amount = amount,
            wait_for_inclusion = wait_for_inclusion,
            wait_for_finalization = wait_for_finalization,
        )
    

    def do_undelegation(
        self,
        wallet: 'bittensor.Wallet',
        delegate_ss58: str,
        amount: 'bittensor.Balance',
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        # Check if delegate
        if not self.is_hotkey_delegate(
            hotkey_ss58 = delegate_ss58
        ):
            raise Exception("Not a delegate")
        
        # do unstake
        self.do_unstake(
            wallet = wallet,
            hotkey_ss58 = delegate_ss58,
            amount = amount,
            wait_for_inclusion = wait_for_inclusion,
            wait_for_finalization = wait_for_finalization,
        )
    
    def do_nominate(
        self,
        wallet: 'bittensor.Wallet',
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        hotkey_ss58 = wallet.hotkey.ss58_address
        coldkey_ss58 = wallet.coldkeypub.ss58_address

        if hotkey_ss58 not in self.chain_state['SubtensorModule']['Neurons']:
            pass
    
    def get_transfer_fee(
        self,
        wallet: 'bittensor.Wallet',
        dest: str,
        value: Union['bittensor.Balance', float, int],
    ) -> 'bittensor.Balance':    
        return bittensor.Balance( 700 )
    
    def do_transfer(
        self,
        wallet: 'bittensor.Wallet',
        dest: str,
        transfer_balance: 'bittensor.Balance',
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        bal = self.get_balance(wallet.coldkeypub.ss58_address)
        dest_bal = self.get_balance(dest)
        transfer_fee = self.get_transfer_fee(wallet, dest, transfer_balance)

        existential_deposit = self.get_existential_deposit()

        if bal < transfer_balance + existential_deposit + transfer_fee:
            raise Exception("Insufficient balance")
        
        # Remove from the free balance
        self.chain_state['System']['Account'][wallet.coldkeypub.ss58_address]['data']['free'][self.block_number] = (bal - transfer_balance - transfer_fee).rao

        # Add to the free balance
        if dest not in self.chain_state['System']['Account']:
            self.chain_state['System']['Account'][dest] = {
                'data': {
                    'free': {},
                }
            }

        self.chain_state['System']['Account'][dest]['data']['free'][self.block_number] = (dest_bal + transfer_balance).rao

    def do_pow_register(
        self,
        netuid: int,
        wallet: 'bittensor.Wallet',
        pow_result: 'POWSolution',
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        # Assume pow result is valid

        subtensor_state = self.chain_state['SubtensorModule']
        if netuid not in subtensor_state['NetworksAdded']:
            raise Exception("Subnet does not exist")

        self._register_neuron(
            netuid=netuid,
            hotkey=wallet.hotkey.ss58_address,
            coldkey=wallet.coldkeypub.ss58_address,
        )

    def do_burned_register(
        self,
        netuid: int,
        wallet: 'bittensor.Wallet',
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        subtensor_state = self.chain_state['SubtensorModule']
        if netuid not in subtensor_state['NetworksAdded']:
            raise Exception("Subnet does not exist")
        
        bal = self.get_balance( wallet.coldkeypub.ss58_address )
        burn = self.burn( netuid=netuid )
        existential_deposit = self.get_existential_deposit( )

        if bal < burn + existential_deposit:
            raise Exception("Insufficient funds")

        self._register_neuron(
            netuid=netuid,
            hotkey=wallet.hotkey.ss58_address,
            coldkey=wallet.coldkeypub.ss58_address,
        )

        # Burn the funds
        self.chain_state['System']['Account'][wallet.coldkeypub.ss58_address]['data']['free'][self.block_number] = (bal - burn).rao

    def do_stake(
        self,
        wallet: 'bittensor.Wallet',
        hotkey_ss58: str,
        amount: 'bittensor.Balance',
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        subtensor_state = self.chain_state['SubtensorModule']
        
        bal = self.get_balance( wallet.coldkeypub.ss58_address )
        curr_stake = self.get_stake_for_coldkey_and_hotkey(
            hotkey_ss58=hotkey_ss58,
            coldkey_ss58=wallet.coldkeypub.ss58_address,
        )
        if curr_stake is None:
            curr_stake = bittensor.Balance(0)
        existential_deposit = self.get_existential_deposit( )

        if bal < amount + existential_deposit:
            raise Exception("Insufficient funds")
        
        stake_state = subtensor_state['Stake']

        # Stake the funds
        if not hotkey_ss58 in stake_state:
            stake_state[hotkey_ss58] = {}
        if not wallet.coldkeypub.ss58_address in stake_state[hotkey_ss58]:
            stake_state[hotkey_ss58][wallet.coldkeypub.ss58_address] = {}
        
        stake_state[hotkey_ss58][wallet.coldkeypub.ss58_address][self.block_number] = amount.rao
        # Remove from free balance
        self.chain_state['System']['Account'][wallet.coldkeypub.ss58_address]['data']['free'][self.block_number] = (bal - amount).rao

    def do_unstake(
        self,
        wallet: 'bittensor.Wallet',
        hotkey_ss58: str,
        amount: 'bittensor.Balance',
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        subtensor_state = self.chain_state['SubtensorModule']
        
        bal = self.get_balance( wallet.coldkeypub.ss58_address )
        curr_stake = self.get_stake_for_coldkey_and_hotkey(
            hotkey_ss58=hotkey_ss58,
            coldkey_ss58=wallet.coldkeypub.ss58_address,
        )
        if curr_stake is None:
            curr_stake = bittensor.Balance(0)

        if curr_stake < amount:
            raise Exception("Insufficient funds")
        
        stake_state = subtensor_state['Stake']

        if curr_stake.rao == 0:
            return True
        
        # Unstake the funds
        # We know that the hotkey has stake, so we can just remove it
        stake_state[hotkey_ss58][wallet.coldkeypub.ss58_address][self.block_number] = (curr_stake - amount).rao
        # Add to the free balance
        if wallet.coldkeypub.ss58_address not in self.chain_state['System']['Account']:
            self.chain_state['System']['Account'][wallet.coldkeypub.ss58_address] = {
                'data': {
                    'free': {},
                }
            }

        self.chain_state['System']['Account'][wallet.coldkeypub.ss58_address]['data']['free'][self.block_number] = (bal + amount).rao

