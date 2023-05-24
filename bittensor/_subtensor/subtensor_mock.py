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

from typing import Optional, Tuple, Dict, Union, TypedDict, List, Any
from random import randint

from substrateinterface import SubstrateInterface
from bittensor import Balance, NeuronInfo, NeuronInfoLite, SubnetInfo, PrometheusInfo, __version_as_int__, axon_info
from bittensor.utils import U16_NORMALIZED_FLOAT, RAOPERTAO

from subtensor_impl import Subtensor

BlockNumber = str # str(int)

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

class MockMapResult:
    records: Optional[List[Any]]

    def __init__(self, records: Optional[List[Any]] = None):
        self.records = records

    def __iter__(self):
        return iter(self.records)

class MockSystemState(TypedDict):
    Account: Dict[str, Dict[str, Balance]] # address -> block -> balance

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
    __shared_state = {}

    chain_state: MockChainState
    block_number: int

    def __init__(self) -> None:
        self.__dict__ = self.__shared_state
        
        if not hasattr(self, 'chain_state'):
            self.chain_state = {
                'System': {
                    'Account': {}
                },
                'SubtensorModule': {
                    'NetworksAdded': {},
                },
            }

            self.block_number = 0

    def create_subnet( self, netuid: int ) -> None:
        subtensor_state = self.chain_state['SubtensorModule']
        if str(netuid) not in subtensor_state['NetworksAdded']:
            subtensor_state['Rho'][str(netuid)] = 10
            subtensor_state['SubtensorModule']['Kappa'][str(netuid)] = 32_767
            subtensor_state['Difficulty'][str(netuid)] = 10_000_000
            subtensor_state['ImmunityPeriod'][str(netuid)] = 4096
            subtensor_state['ValidatorBatchSize'][str(netuid)] = 32
            subtensor_state['ValidatorSequenceLen'][str(netuid)] = 256
            subtensor_state['ValidatorEpochsPerReset'][str(netuid)] = 60
            subtensor_state['ValidatorEpochLen'][str(netuid)] = 100
            subtensor_state['MaxAllowedValidators'][str(netuid)] = 128
            subtensor_state['MinAllowedWeights'][str(netuid)] = 1024
            subtensor_state['MaxWeightsLimit'][str(netuid)] = 1_000
            subtensor_state['SynergyScalingLawPower'][str(netuid)] = 50
            subtensor_state['ScalingLawPower'][str(netuid)] = 50
            subtensor_state['SubnetworkN'][str(netuid)] = 0
            subtensor_state['MaxAllowedUids'][str(netuid)] = 4096
            subtensor_state['NetworkModality'][str(netuid)] = 0
            subtensor_state['BlocksSinceLastStep'][str(netuid)] = 0
            subtensor_state['Tempo'][str(netuid)] = 99
            subtensor_state['NetworkConnect'][str(netuid)] = {}
            subtensor_state['EmissionValue'][str(netuid)] = 0
            subtensor_state['Burn'][str(netuid)] = 0

            subtensor_state['Active'][str(netuid)] = {}

            subtensor_state['UIDs'][str(netuid)] = {}
            subtensor_state['Keys'][str(netuid)] = {}
            subtensor_state['LastUpdate'][str(netuid)] = {}
            
            subtensor_state['Rank'][str(netuid)] = {}
            subtensor_state['Emission'][str(netuid)] = {}
            subtensor_state['Incentive'][str(netuid)] = {}
            subtensor_state['Consensus'][str(netuid)] = {}
            subtensor_state['Trust'][str(netuid)] = {}
            subtensor_state['ValidatorTrust'][str(netuid)] = {}
            subtensor_state['Dividends'][str(netuid)] = {}
            subtensor_state['PruningScores'][str(netuid)] = {}
            subtensor_state['ValidatorPermit'][str(netuid)] = {}

            subtensor_state['Weights'][str(netuid)] = {}
            subtensor_state['Bonds'][str(netuid)] = {}

            subtensor_state['TotalStake'][str(self.block_number)] = Balance(0)

        else:
            raise Exception("Subnet already exists")
        
    def force_register_neuron(
        self,
        netuid: int,
        hotkey: str, 
        coldkey: str,
        stake: Union[Balance, float] = Balance(0),
    ) -> int:
        """
        Force register a neuron on the mock chain, returning the UID.
        """
        if isinstance(stake, float):
            stake = Balance.from_tao(stake)

        subtensor_state = self.chain_state['SubtensorModule']
        if str(netuid) not in subtensor_state['NetworksAdded']:
            raise Exception("Subnet does not exist")

        subnetwork_n = subtensor_state['SubnetworkN'][str(netuid)][-1]
        for uid in range(subnetwork_n): # Get last block
            if subtensor_state['Keys'][str(netuid)][uid][-1] == hotkey:
                # already_registered 
                break

        else:
            # Not found
            if subnetwork_n >= subtensor_state['MaxAllowedUids'][str(netuid)][-1]:
                # Subnet full, replace neuron randomly
                uid = randint(0, subnetwork_n-1)
            else:
                # Subnet not full, add new neuron
                # Append as next uid and increment subnetwork_n
                uid = subnetwork_n
                subtensor_state['SubnetworkN'][str(netuid)][str(self.block_number)] = subnetwork_n + 1

            subtensor_state['Stake'][hotkey][coldkey][str(self.block_number)] = stake
            subtensor_state['UIDs'][str(netuid)][hotkey][str(self.block_number)] = uid
            subtensor_state['Keys'][str(netuid)][str(uid)][str(self.block_number)] = hotkey
            subtensor_state['Owner'][hotkey][str(self.block_number)] = coldkey
            subtensor_state['LastUpdate'][str(netuid)][str(uid)][str(self.block_number)] = self.block_number
            
            subtensor_state['Rank'][str(netuid)][str(uid)][str(self.block_number)] = 0.0
            subtensor_state['Emission'][str(netuid)][str(uid)][str(self.block_number)] = 0.0
            subtensor_state['Incentive'][str(netuid)][str(uid)][str(self.block_number)] = 0.0
            subtensor_state['Consensus'][str(netuid)][str(uid)][str(self.block_number)] = 0.0
            subtensor_state['Trust'][str(netuid)][str(uid)][str(self.block_number)] = 0.0
            subtensor_state['ValidatorTrust'][str(netuid)][str(uid)][str(self.block_number)] = 0.0
            subtensor_state['Dividends'][str(netuid)][str(uid)][str(self.block_number)] = 0.0
            subtensor_state['PruningScores'][str(netuid)][str(uid)][str(self.block_number)] = 0.0
            subtensor_state['ValidatorPermit'][str(netuid)][str(uid)][str(self.block_number)] = False

            subtensor_state['Weights'][str(netuid)][str(uid)][str(self.block_number)] = []
            subtensor_state['Bonds'][str(netuid)][str(uid)][str(self.block_number)] = []

            subtensor_state['TotalStake'][str(self.block_number)] = subtensor_state['TotalStake'][-1] + stake

            subtensor_state['Axons'][str(netuid)][hotkey][str(self.block_number)] = {}
            subtensor_state['Prometheus'][str(netuid)][hotkey][str(self.block_number)] = {}



        raise Exception("Hotkey already registered")
            

    def force_set_balance(
        self,
        ss58_address: str,
        balance: Union[Balance, float] = Balance(0),
        stake: Union[Balance, float] = Balance(0),
    ) -> None:
        if isinstance(stake, float):
            stake = Balance.from_tao(stake)
            
        if isinstance(balance, float):
            balance = Balance.from_tao(balance)

        if ss58_address not in self.chain_state['System']['Account']:
            self.chain_state['System']['Account'][ss58_address] = {}

        self.chain_state['System']['Account'][ss58_address][str(self.block_number)] = {
            'data': {
                'free': balance.rao,
            }
        }
        
        
    def do_block_step( self ) -> None:
        self.block_number += 1

        # Doesn't do epoch
        subtensor_state = self.chain_state['SubtensorModule']
        for subnet in subtensor_state['NetworksAdded']:
            subtensor_state['BlocksSinceLastStep'][subnet][str(self.block_number)] = subtensor_state['BlocksSinceLastStep'][subnet][-1] + 1

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
            state_at_block = state.get(str(block), None)
            while state_at_block is None and block > 0:
                block -= 1
                state_at_block = self.state.get(str(block), None)

            return state_at_block # Can be None
        else:
            return None
            
    def query_map_subtensor( self, name: str, block: Optional[int] = None, params: Optional[List[object]] = [] ) -> Optional[MockMapResult]:
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
            state_at_block = state.get(str(block), None)
            while state_at_block is None and block > 0:
                block -= 1
                state_at_block = self.state.get(str(block), None)
                if state_at_block is not None:
                    break
            else:
                return None

            return MockMapResult(state_at_block)
        else:
            return None

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
            state_at_block = state.get(str(block), None)
            while state_at_block is None and block > 0:
                block -= 1
                state_at_block = self.state.get(str(block), None)

            return state_at_block # Can be None
        else:
            return None

    def get_current_block( self ) -> int:
        return self.block_number

    # ==== Balance RPC methods ====

    def get_balance(self, address: str, block: int = None) -> Balance:
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
                return Balance(0)
                    
            # Use block
            state_at_block = state.get(str(block), None)
            while state_at_block is None and block > 0:
                block -= 1
                state_at_block = self.state.get(str(block), None)

            return state_at_block # Can be None
        else:
            return None

    def get_balances(self, block: int = None) -> Dict[str, Balance]:
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

        if str(netuid) not in self.chain_state['SubtensorModule']['NetworksAdded']:
            return None
        
        neuron_info = self._neuron_subnet_exists( uid, netuid, block )
        if neuron_info is None:
            return None
        
        else:
            return neuron_info

    def neurons(self, netuid: int, block: Optional[int] = None ) -> List[NeuronInfo]:
        if str(netuid) not in self.chain_state['SubtensorModule']['NetworksAdded']:
            raise Exception("Subnet does not exist")
        
        neurons = []
        for uid in range(self.chain_state['SubtensorModule']['SubnetworkN'][str(netuid)][-1]):
            neuron_info = self.neuron_for_uid( uid, netuid, block )
            if neuron_info is not None:
                neurons.append(neuron_info)

        return neurons

    @staticmethod
    def _get_most_recent_storage( storage: Dict[BlockNumber, Any], block_number: Optional[int] ) -> Any:
        if block_number is None:
            return storage[-1]
        
        else:
            while block_number > 0:
                if str(block_number) in storage:
                    return storage[str(block_number)]
                
                block_number -= 1
            
            return None
        
    def _get_axon_info( self, netuid: int, hotkey: str, block: Optional[int] = None ) -> Optional[AxonInfoDict]:
        # Axons [netuid][hotkey][block_number]
        subtensor_state = self.chain_state['SubtensorModule']
        if str(netuid) not in subtensor_state['Axons']:
            return None
        
        return self._get_most_recent_storage(subtensor_state['Axons'][str(netuid)][hotkey], block)

    def _get_prometheus_info( self, netuid: int, hotkey: str, block: Optional[int] = None ) -> Optional[PrometheusInfoDict]:
        subtensor_state = self.chain_state['SubtensorModule']
        if str(netuid) not in subtensor_state['Prometheus']:
            return None
        
        return self._get_most_recent_storage(subtensor_state['Prometheus'][str(netuid)][hotkey], block)

    def _neuron_subnet_exists( self, uid: int, netuid: int, block: Optional[int] = None ) -> Optional[NeuronInfo]:
        subtensor_state = self.chain_state['SubtensorModule']
        if str(netuid) not in subtensor_state['NetworksAdded']:
            return None
        
        if subtensor_state['SubnetworkN'][str(netuid)][-1] <= uid:
            return None
        
        hotkey = subtensor_state['Keys'][str(netuid)][str(uid)][-1]
        if hotkey is None:
            return None


        axon_info_ = self._get_axon_info( netuid, hotkey, block )

        prometheus_info = self._get_prometheus_info( netuid, hotkey, block )


        coldkey = self._get_most_recent_storage(subtensor_state['Owner'][hotkey], block)
        active = self._get_most_recent_storage(subtensor_state['Active'][str(netuid)][str(uid)], block)
        rank = self._get_most_recent_storage(subtensor_state['Rank'][str(netuid)][str(uid)], block)
        emission = self._get_most_recent_storage(subtensor_state['Emission'][str(netuid)][str(uid)], block)
        incentive = self._get_most_recent_storage(subtensor_state['Incentive'][str(netuid)][str(uid)], block)
        consensus = self._get_most_recent_storage(subtensor_state['Consensus'][str(netuid)][str(uid)], block)
        trust = self._get_most_recent_storage(subtensor_state['Trust'][str(netuid)][str(uid)], block)
        validator_trust = self._get_most_recent_storage(subtensor_state['ValidatorTrust'][str(netuid)][str(uid)], block)
        dividends = self._get_most_recent_storage(subtensor_state['Dividends'][str(netuid)][str(uid)], block)
        pruning_score = self._get_most_recent_storage(subtensor_state['PruningScores'][str(netuid)][str(uid)], block)
        last_update = self._get_most_recent_storage(subtensor_state['LastUpdate'][str(netuid)][str(uid)], block)
        validator_permit = self._get_most_recent_storage(subtensor_state['ValidatorPermit'][str(netuid)][str(uid)], block)
        
        weights = self._get_most_recent_storage(subtensor_state['Weights'][str(netuid)][str(uid)], block)
        bonds = self._get_most_recent_storage(subtensor_state['Bonds'][str(netuid)][str(uid)], block)

        stake_dict = [(coldkey, Balance.from_rao(self._get_most_recent_storage(
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

        if str(netuid) not in self.chain_state['SubtensorModule']['NetworksAdded']:
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
        if str(netuid) not in self.chain_state['SubtensorModule']['NetworksAdded']:
            raise Exception("Subnet does not exist")
        
        neurons = []
        for uid in range(self.chain_state['SubtensorModule']['SubnetworkN'][str(netuid)][-1]):
            neuron_info = self.neuron_for_uid_lite( uid, netuid, block )
            if neuron_info is not None:
                neurons.append(neuron_info)

        return neurons
        