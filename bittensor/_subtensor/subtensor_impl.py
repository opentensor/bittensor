# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
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

# Imports
import torch
import bittensor
import scalecodec
from retry import retry
from typing import List, Dict, Union, Optional, Tuple
from substrateinterface import SubstrateInterface
from bittensor.utils.balance import Balance
from bittensor.utils import U16_NORMALIZED_FLOAT, U64_MAX, RAOPERTAO, U16_MAX

# Local imports.
from .chain_data import NeuronInfo, AxonInfo, DelegateInfo, PrometheusInfo, SubnetInfo, NeuronInfoLite
from .errors import *
from .extrinsics.staking import add_stake_extrinsic, add_stake_multiple_extrinsic
from .extrinsics.unstaking import unstake_extrinsic, unstake_multiple_extrinsic
from .extrinsics.serving import serve_extrinsic, serve_axon_extrinsic
from .extrinsics.registration import register_extrinsic, burned_register_extrinsic
from .extrinsics.transfer import transfer_extrinsic
from .extrinsics.set_weights import set_weights_extrinsic
from .extrinsics.prometheus import prometheus_extrinsic
from .extrinsics.delegation import delegate_extrinsic, nominate_extrinsic,undelegate_extrinsic

# Logging
from loguru import logger
logger = logger.opt(colors=True)

class Subtensor:
    """
    Handles interactions with the subtensor chain.
    """
    def __init__( 
        self, 
        substrate: 'SubstrateInterface',
        network: str,
        chain_endpoint: str,
    ):
        r""" Initializes a subtensor chain interface.
            Args:
                substrate (:obj:`SubstrateInterface`, `required`): 
                    substrate websocket client.
                network (default='local', type=str)
                    The subtensor network flag. The likely choices are:
                            -- local (local running network)
                            -- nobunaga (staging network)
                            -- nakamoto (main network)
                    If this option is set it overloads subtensor.chain_endpoint with 
                    an entry point node from that network.
                chain_endpoint (default=None, type=str)
                    The subtensor endpoint flag. If set, overrides the network argument.
        """
        self.network = network
        self.chain_endpoint = chain_endpoint
        self.substrate = substrate

    def __str__(self) -> str:
        if self.network == self.chain_endpoint:
            # Connecting to chain endpoint without network known.
            return "Subtensor({})".format( self.chain_endpoint )
        else:
            # Connecting to network with endpoint known.
            return "Subtensor({}, {})".format( self.network, self.chain_endpoint )

    def __repr__(self) -> str:
        return self.__str__()

    #####################
    #### Delegation #####
    #####################
    def nominate( 
        self,
        wallet: 'bittensor.Wallet', 
        wait_for_finalization: bool = False, 
        wait_for_inclusion: bool = True 
    ) -> bool:
        """ Becomes a delegate for the hotkey."""
        return nominate_extrinsic( 
            subtensor = self, 
            wallet = wallet, 
            wait_for_finalization = wait_for_finalization,
            wait_for_inclusion = wait_for_inclusion
        )

    def delegate(
        self, 
        wallet: 'bittensor.wallet',
        delegate_ss58: Optional[str] = None,
        amount: Union[Balance, float] = None, 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """ Adds the specified amount of stake to the passed delegate using the passed wallet. """
        return delegate_extrinsic( 
            subtensor = self, 
            wallet = wallet,
            delegate_ss58 = delegate_ss58, 
            amount = amount, 
            wait_for_inclusion = wait_for_inclusion,
            wait_for_finalization = wait_for_finalization, 
            prompt = prompt
        )

    def undelegate(
        self, 
        wallet: 'bittensor.wallet',
        delegate_ss58: Optional[str] = None,
        amount: Union[Balance, float] = None, 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """ Removes the specified amount of stake from the passed delegate using the passed wallet. """
        return undelegate_extrinsic( 
            subtensor = self, 
            wallet = wallet,
            delegate_ss58 = delegate_ss58, 
            amount = amount, 
            wait_for_inclusion = wait_for_inclusion,
            wait_for_finalization = wait_for_finalization, 
            prompt = prompt
        )

    #####################
    #### Set Weights ####
    #####################
    def set_weights(
        self,
        wallet: 'bittensor.wallet',
        netuid: int,
        uids: Union[torch.LongTensor, list],
        weights: Union[torch.FloatTensor, list],
        version_key: int = bittensor.__version_as_int__,
        wait_for_inclusion:bool = False,
        wait_for_finalization:bool = False,
        prompt:bool = False
    ) -> bool:
        return set_weights_extrinsic( 
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

    ######################
    #### Registration ####
    ######################
    def register (
        self,
        wallet: 'bittensor.Wallet',
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False,
        max_allowed_attempts: int = 3,
        output_in_place: bool = True,
        cuda: bool = False,
        dev_id: Union[List[int], int] = 0,
        TPB: int = 256,
        num_processes: Optional[int] = None,
        update_interval: Optional[int] = None,
        log_verbose: bool = False,
    ) -> bool:
        """ Registers the wallet to chain."""
        return register_extrinsic( 
            subtensor = self, 
            wallet = wallet, 
            netuid = netuid, 
            wait_for_inclusion = wait_for_inclusion, 
            wait_for_finalization = wait_for_finalization, 
            prompt = prompt,
            max_allowed_attempts = max_allowed_attempts,
            output_in_place = output_in_place,
            cuda = cuda,
            dev_id = dev_id,
            TPB = TPB,
            num_processes = num_processes,
            update_interval = update_interval,
            log_verbose = log_verbose,
        )
    
    def burned_register (
        self,
        wallet: 'bittensor.Wallet',
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False
    ) -> bool:
        """ Registers the wallet to chain by recycling TAO."""
        return burned_register_extrinsic( 
            subtensor = self, 
            wallet = wallet, 
            netuid = netuid, 
            wait_for_inclusion = wait_for_inclusion, 
            wait_for_finalization = wait_for_finalization, 
            prompt = prompt
        )

    ##################
    #### Transfer ####
    ##################
    def transfer(
        self,
        wallet: 'bittensor.wallet',
        dest: str, 
        amount: Union[Balance, float], 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """ Transfers funds from this wallet to the destination public key address"""
        return transfer_extrinsic(
            subtensor = self,
            wallet = wallet,
            dest = dest,
            amount = amount,
            wait_for_inclusion = wait_for_inclusion,
            wait_for_finalization = wait_for_finalization,
            prompt = prompt
        )
    
    def get_existential_deposit(
        self,
        block: Optional[int] = None,
    ) -> Optional[Balance]:
        """ Returns the existential deposit for the chain. """
        result = self.query_constant(
            module_name='Balances',
            constant_name='ExistentialDeposit',
            block = block,
        )
        
        if result is None:
            return None
        
        return Balance.from_rao(result.value)

    #################
    #### Serving ####
    #################
    def serve (
        self,
        wallet: 'bittensor.wallet',
        ip: str, 
        port: int, 
        protocol: int, 
        netuid: int,
        placeholder1: int = 0,
        placeholder2: int = 0,
        wait_for_inclusion: bool = False,
        wait_for_finalization = True,
        prompt: bool = False,
    ) -> bool:
        return serve_extrinsic( self, wallet, ip, port, protocol, netuid , placeholder1, placeholder2, wait_for_inclusion, wait_for_finalization)

    def serve_axon (
        self,
        axon: 'bittensor.Axon',
        use_upnpc: bool = False,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False,
    ) -> bool:
        return serve_axon_extrinsic( self, axon, use_upnpc, wait_for_inclusion, wait_for_finalization)

    def serve_prometheus (
        self,
        wallet: 'bittensor.wallet',
        port: int,
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        return prometheus_extrinsic( self, wallet = wallet, port = port, netuid = netuid, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization)
    #################
    #### Staking ####
    #################
    def add_stake(
        self, 
        wallet: 'bittensor.wallet',
        hotkey_ss58: Optional[str] = None,
        amount: Union[Balance, float] = None, 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """ Adds the specified amount of stake to passed hotkey uid. """
        return add_stake_extrinsic( 
            subtensor = self, 
            wallet = wallet,
            hotkey_ss58 = hotkey_ss58, 
            amount = amount, 
            wait_for_inclusion = wait_for_inclusion,
            wait_for_finalization = wait_for_finalization, 
            prompt = prompt
        )

    def add_stake_multiple (
        self, 
        wallet: 'bittensor.wallet',
        hotkey_ss58s: List[str],
        amounts: List[Union[Balance, float]] = None, 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """ Adds stake to each hotkey_ss58 in the list, using each amount, from a common coldkey."""
        return add_stake_multiple_extrinsic( self, wallet, hotkey_ss58s, amounts, wait_for_inclusion, wait_for_finalization, prompt)

    ###################
    #### Unstaking ####
    ###################
    def unstake_multiple (
        self,
        wallet: 'bittensor.wallet',
        hotkey_ss58s: List[str],
        amounts: List[Union[Balance, float]] = None, 
        wait_for_inclusion: bool = True, 
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """ Removes stake from each hotkey_ss58 in the list, using each amount, to a common coldkey. """
        return unstake_multiple_extrinsic( self, wallet, hotkey_ss58s, amounts, wait_for_inclusion, wait_for_finalization, prompt)

   

    def unstake (
        self,
        wallet: 'bittensor.wallet',
        hotkey_ss58: Optional[str] = None,
        amount: Union[Balance, float] = None, 
        wait_for_inclusion:bool = True, 
        wait_for_finalization:bool = False,
        prompt: bool = False,
    ) -> bool:
        """ Removes stake into the wallet coldkey from the specified hotkey uid."""
        return unstake_extrinsic( self, wallet, hotkey_ss58, amount, wait_for_inclusion, wait_for_finalization, prompt )


    ########################
    #### Standard Calls ####
    ########################

    """ Queries subtensor named storage with params and block. """
    def query_subtensor( self, name: str, block: Optional[int] = None, params: Optional[List[object]] = [] ) -> Optional[object]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='SubtensorModule',
                    storage_function = name,
                    params = params,
                    block_hash = None if block == None else substrate.get_block_hash(block)
                )
        return make_substrate_call_with_retry()

    """ Queries subtensor map storage with params and block. """
    def query_map_subtensor( self, name: str, block: Optional[int] = None, params: Optional[List[object]] = [] ) -> Optional[object]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query_map(
                    module='SubtensorModule',
                    storage_function = name,
                    params = params,
                    block_hash = None if block == None else substrate.get_block_hash(block)
                )
        return make_substrate_call_with_retry()
    
    """ Gets a constant from subtensor with module_name, constant_name, and block. """
    def query_constant( self, module_name: str, constant_name: str, block: Optional[int] = None ) -> Optional[object]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.get_constant(
                    module_name=module_name,
                    constant_name=constant_name,
                    block_hash = None if block == None else substrate.get_block_hash(block)
                )
        return make_substrate_call_with_retry()
      
    #####################################
    #### Hyper parameter calls. ####
    #####################################

    """ Returns network Rho hyper parameter """
    def rho (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor( "Rho", block, [netuid] ).value

    """ Returns network Kappa hyper parameter """
    def kappa (self, netuid: int, block: Optional[int] = None ) -> Optional[float]:
        if not self.subnet_exists( netuid, block ): return None
        return U16_NORMALIZED_FLOAT( self.query_subtensor( "Kappa", block, [netuid] ).value )

    """ Returns network Difficulty hyper parameter """
    def difficulty (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor( "Difficulty", block, [netuid] ).value
    
    """ Returns network Burn hyper parameter """
    def burn (self, netuid: int, block: Optional[int] = None ) -> Optional[bittensor.Balance]:
        if not self.subnet_exists( netuid, block ): return None
        return bittensor.Balance.from_rao( self.query_subtensor( "Burn", block, [netuid] ).value )

    """ Returns network ImmunityPeriod hyper parameter """
    def immunity_period (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor("ImmunityPeriod", block, [netuid] ).value

    """ Returns network ValidatorBatchSize hyper parameter """
    def validator_batch_size (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor("ValidatorBatchSize", block, [netuid] ).value

    """ Returns network ValidatorPruneLen hyper parameter """
    def validator_prune_len (self, netuid: int, block: Optional[int] = None ) -> int:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor("ValidatorPruneLen", block, [netuid] ).value

    """ Returns network ValidatorLogitsDivergence hyper parameter """
    def validator_logits_divergence (self, netuid: int, block: Optional[int] = None ) -> Optional[float]:
        if not self.subnet_exists( netuid, block ): return None
        return U16_NORMALIZED_FLOAT(self.query_subtensor("ValidatorLogitsDivergence", block, [netuid]).value)

    """ Returns network ValidatorSequenceLength hyper parameter """
    def validator_sequence_length (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor("ValidatorSequenceLength", block, [netuid] ).value

    """ Returns network ValidatorEpochsPerReset hyper parameter """
    def validator_epochs_per_reset (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor("ValidatorEpochsPerReset", block, [netuid] ).value

    """ Returns network ValidatorEpochLen hyper parameter """
    def validator_epoch_length (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor("ValidatorEpochLen", block, [netuid] ).value

    """ Returns network ValidatorEpochLen hyper parameter """
    def validator_exclude_quantile (self, netuid: int, block: Optional[int] = None ) -> Optional[float]:
        if not self.subnet_exists( netuid, block ): return None
        return U16_NORMALIZED_FLOAT( self.query_subtensor("ValidatorExcludeQuantile", block, [netuid] ).value )

    """ Returns network MaxAllowedValidators hyper parameter """
    def max_allowed_validators(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor( 'MaxAllowedValidators', block, [netuid] ).value
        
    """ Returns network MinAllowedWeights hyper parameter """
    def min_allowed_weights (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor("MinAllowedWeights", block, [netuid] ).value

    """ Returns network MaxWeightsLimit hyper parameter """
    def max_weight_limit (self, netuid: int, block: Optional[int] = None ) -> Optional[float]:
        if not self.subnet_exists( netuid, block ): return None
        return U16_NORMALIZED_FLOAT( self.query_subtensor('MaxWeightsLimit', block, [netuid] ).value )

    """ Returns network ScalingLawPower hyper parameter """
    def scaling_law_power (self, netuid: int, block: Optional[int] = None ) -> Optional[float]:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor('ScalingLawPower', block, [netuid] ).value / 100.

    """ Returns network SynergyScalingLawPower hyper parameter """
    def synergy_scaling_law_power (self, netuid: int, block: Optional[int] = None ) -> Optional[float]:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor('SynergyScalingLawPower', block, [netuid] ).value / 100.

    """ Returns network SubnetworkN hyper parameter """
    def subnetwork_n (self, netuid: int, block: Optional[int] = None ) -> int:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor('SubnetworkN', block, [netuid] ).value

    """ Returns network MaxAllowedUids hyper parameter """
    def max_n (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor('MaxAllowedUids', block, [netuid] ).value

    """ Returns network BlocksSinceLastStep hyper parameter """
    def blocks_since_epoch (self, netuid: int, block: Optional[int] = None) -> int:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor('BlocksSinceLastStep', block, [netuid] ).value

    """ Returns network Tempo hyper parameter """
    def tempo (self, netuid: int, block: Optional[int] = None) -> int:
        if not self.subnet_exists( netuid, block ): return None
        return self.query_subtensor('Tempo', block, [netuid] ).value

    ##########################
    #### Account functions ###
    ##########################

    """ Returns the total stake held on a hotkey including delegative """
    def get_total_stake_for_hotkey( self, ss58_address: str, block: Optional[int] = None ) -> Optional['bittensor.Balance']:
        return bittensor.Balance.from_rao( self.query_subtensor( 'TotalHotkeyStake', block, [ss58_address] ).value )

    """ Returns the total stake held on a coldkey across all hotkeys including delegates"""
    def get_total_stake_for_coldkey( self, ss58_address: str, block: Optional[int] = None ) -> Optional['bittensor.Balance']:
        return bittensor.Balance.from_rao( self.query_subtensor( 'TotalColdkeyStake', block, [ss58_address] ).value )

    """ Returns the stake under a coldkey - hotkey pairing """
    def get_stake_for_coldkey_and_hotkey( self, hotkey_ss58: str, coldkey_ss58: str, block: Optional[int] = None ) -> Optional['bittensor.Balance']:
        return bittensor.Balance.from_rao( self.query_subtensor( 'Stake', block, [hotkey_ss58, coldkey_ss58] ).value )

    """ Returns a list of stake tuples (coldkey, balance) for each delegating coldkey including the owner"""
    def get_stake( self, hotkey_ss58: str, block: Optional[int] = None ) -> List[Tuple[str,'bittensor.Balance']]:
        return [ (r[0].value, bittensor.Balance.from_rao( r[1].value ))  for r in self.query_map_subtensor( 'Stake', block, [hotkey_ss58] ) ]

    """ Returns true if the hotkey is known by the chain and there are accounts. """
    def does_hotkey_exist( self, hotkey_ss58: str, block: Optional[int] = None ) -> bool:
        return (self.query_subtensor( 'Owner', block, [hotkey_ss58 ] ).value != "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM")

    """ Returns the coldkey owner of the passed hotkey """
    def get_hotkey_owner( self, hotkey_ss58: str, block: Optional[int] = None ) -> Optional[str]:
        if self.does_hotkey_exist( hotkey_ss58, block ):
            return self.query_subtensor( 'Owner', block, [hotkey_ss58 ] ).value
        else:
            return None

    """ Returns the axon information for this hotkey account """
    def get_axon_info( self, hotkey_ss58: str, block: Optional[int] = None ) -> Optional[AxonInfo]:
        result = self.query_subtensor( 'Axons', block, [hotkey_ss58 ] )        
        if result != None:
            return AxonInfo(
                ip = bittensor.utils.networking.ip_from_int( result.value.ip ),
                ip_type = result.value.ip_type,
                port = result.value.port,
                protocol = result.value.protocol,
                version = result.value.version,
                placeholder1 = result.value.placeholder1,
                placeholder2 = result.value.placeholder2,
            )
        else:
            return None

    """ Returns the prometheus information for this hotkey account """
    def get_prometheus_info( self, hotkey_ss58: str, block: Optional[int] = None ) -> Optional[AxonInfo]:
        result = self.query_subtensor( 'Prometheus', block, [hotkey_ss58 ] )        
        if result != None:
            return PrometheusInfo (
                ip = bittensor.utils.networking.ip_from_int( result.value.ip ),
                ip_type = result.value.ip_type,
                port = result.value.port,
                version = result.value.version,
                block = result.value.block,
            )
        else:
            return None

    ###########################
    #### Global Parameters ####
    ###########################

    @property
    def block (self) -> int:
        r""" Returns current chain block.
        Returns:
            block (int):
                Current chain block.
        """
        return self.get_current_block()

    def total_issuance (self, block: Optional[int] = None ) -> 'bittensor.Balance':
        return bittensor.Balance.from_rao( self.query_subtensor( 'TotalIssuance', block ).value )

    def total_stake (self,block: Optional[int] = None ) -> 'bittensor.Balance':
        return bittensor.Balance.from_rao( self.query_subtensor( "TotalStake", block ).value )

    def serving_rate_limit (self, block: Optional[int] = None ) -> Optional[int]:
        return self.query_subtensor( "ServingRateLimit", block ).value
    
    def tx_rate_limit (self, block: Optional[int] = None ) -> Optional[int]:
        return self.query_subtensor( "TxRateLimit", block ).value

    #####################################
    #### Network Parameters ####
    #####################################

    def subnet_exists( self, netuid: int, block: Optional[int] = None ) -> bool:
        return self.query_subtensor( 'NetworksAdded', block, [netuid] ).value  

    def get_all_subnet_netuids( self, block: Optional[int] = None ) -> List[int]:
        subnet_netuids = []
        result = self.query_map_subtensor( 'NetworksAdded', block )
        if result.records:
            for netuid, exists in result:  
                if exists:
                    subnet_netuids.append( netuid.value )
            
        return subnet_netuids

    def get_total_subnets( self, block: Optional[int] = None ) -> int:
        return self.query_subtensor( 'TotalNetworks', block ).value      

    def get_subnet_modality( self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        return self.query_subtensor( 'NetworkModality', block, [netuid] ).value   

    def get_subnet_connection_requirement( self, netuid_0: int, netuid_1: int, block: Optional[int] = None) -> Optional[int]:
        return self.query_subtensor( 'NetworkConnect', block, [netuid_0, netuid_1] ).value

    def get_emission_value_by_subnet( self, netuid: int, block: Optional[int] = None ) -> Optional[float]:
        return bittensor.Balance.from_rao( self.query_subtensor( 'EmissionValues', block, [ netuid ] ).value )

    def get_subnet_connection_requirements( self, netuid: int, block: Optional[int] = None) -> Dict[str, int]:
        result = self.query_map_subtensor( 'NetworkConnect', block, [netuid] )
        if result.records:
            requirements = {}
            for tuple in result.records:
                requirements[str(tuple[0].value)] = tuple[1].value
        else:
            return {}

    def get_subnets( self, block: Optional[int] = None ) -> List[int]:
        subnets = []
        result = self.query_map_subtensor( 'NetworksAdded', block )
        if result.records:
            for network in result.records:
                subnets.append( network[0].value )
            return subnets
        else:
            return []

    def get_all_subnets_info( self, block: Optional[int] = None ) -> List[SubnetInfo]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = []
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="subnetInfo_getSubnetsInfo", # custom rpc method
                    params=params
                )
        
        json_body = make_substrate_call_with_retry()
        result = json_body['result']

        if result in (None, []):
            return []
        
        return SubnetInfo.list_from_vec_u8( result )

    def get_subnet_info( self, netuid: int, block: Optional[int] = None ) -> Optional[SubnetInfo]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [netuid]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="subnetInfo_getSubnetInfo", # custom rpc method
                    params=params
                )
        
        json_body = make_substrate_call_with_retry()
        result = json_body['result']

        if result in (None, []):
            return None
        
        return SubnetInfo.from_vec_u8( result )
        
    ####################
    #### Nomination ####
    ####################
    def is_hotkey_delegate( self, hotkey_ss58: str ) -> bool:
        return hotkey_ss58 in [ info.hotkey_ss58 for info in self.get_delegates() ]

    def get_delegate_take( self, hotkey_ss58: str, block: Optional[int] = None ) -> Optional[float]:
        return U16_NORMALIZED_FLOAT( self.query_subtensor( 'Delegates', block, [ hotkey_ss58 ] ).value )

    def get_nominators_for_hotkey( self, hotkey_ss58: str, block: Optional[int] = None ) -> List[Tuple[str, Balance]]:
        result = self.query_map_subtensor( 'Stake', block, [ hotkey_ss58 ] ) 
        if result.records:
            return [(record[0].value, record[1].value) for record in result.records]
        else:
            return 0

    def get_delegate_by_hotkey( self, hotkey_ss58: str, block: Optional[int] = None ) -> Optional[DelegateInfo]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry(encoded_hotkey: List[int]):
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [encoded_hotkey]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="delegateInfo_getDelegate", # custom rpc method
                    params=params
                )

        hotkey_bytes: bytes = bittensor.utils.ss58_address_to_bytes( hotkey_ss58 )
        encoded_hotkey: List[int] = [ int( byte ) for byte in hotkey_bytes ]
        json_body = make_substrate_call_with_retry(encoded_hotkey)
        result = json_body['result']

        if result in (None, []):
            return None
            
        return DelegateInfo.from_vec_u8( result )

    def get_delegates( self, block: Optional[int] = None ) -> List[DelegateInfo]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = []
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="delegateInfo_getDelegates", # custom rpc method
                    params=params
                )
        json_body = make_substrate_call_with_retry()
        result = json_body['result']

        if result in (None, []):
            return []

        return DelegateInfo.list_from_vec_u8( result )
    
    def get_delegated( self, coldkey_ss58: str, block: Optional[int] = None ) -> List[Tuple[DelegateInfo, Balance]]:
        """ Returns the list of delegates that a given coldkey is staked to.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry(encoded_coldkey: List[int]):
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [encoded_coldkey]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="delegateInfo_getDelegated", # custom rpc method
                    params=params
                )

        coldkey_bytes: bytes = bittensor.utils.ss58_address_to_bytes( coldkey_ss58 )
        encoded_coldkey: List[int] = [ int( byte ) for byte in coldkey_bytes ]
        json_body = make_substrate_call_with_retry(encoded_coldkey)
        result = json_body['result']

        if result in (None, []):
            return []

        return DelegateInfo.delegated_list_from_vec_u8( result )


    ########################################
    #### Neuron information per subnet ####
    ########################################

    def is_hotkey_registered_any( self, hotkey_ss58: str, block: Optional[int] = None) -> bool:
        return len( self.get_netuids_for_hotkey( hotkey_ss58, block) ) > 0
    
    def is_hotkey_registered_on_subnet( self, hotkey_ss58: str, netuid: int, block: Optional[int] = None) -> bool:
        return self.get_uid_for_hotkey_on_subnet( hotkey_ss58, netuid, block ) != None

    def is_hotkey_registered( self, hotkey_ss58: str, netuid: int, block: Optional[int] = None) -> bool:
        return self.get_uid_for_hotkey_on_subnet( hotkey_ss58, netuid, block ) != None

    def get_uid_for_hotkey_on_subnet( self, hotkey_ss58: str, netuid: int, block: Optional[int] = None) -> int:
        return self.query_subtensor( 'Uids', block, [ netuid, hotkey_ss58 ] ).value  

    def get_all_uids_for_hotkey( self, hotkey_ss58: str, block: Optional[int] = None) -> List[int]:
        return [ self.get_uid_for_hotkey_on_subnet( hotkey_ss58, netuid, block) for netuid in self.get_netuids_for_hotkey( hotkey_ss58, block)]

    def get_netuids_for_hotkey( self, hotkey_ss58: str, block: Optional[int] = None) -> List[int]:
        result = self.query_map_subtensor( 'IsNetworkMember', block, [ hotkey_ss58 ] )   
        netuids = []
        for netuid, is_member in result.records:
            if is_member:
                netuids.append( netuid.value )
        return netuids

    def get_neuron_for_pubkey_and_subnet( self, hotkey_ss58: str, netuid: int, block: Optional[int] = None ) -> Optional[NeuronInfo]:
        return self.neuron_for_uid( self.get_uid_for_hotkey_on_subnet(hotkey_ss58, netuid, block=block), netuid, block = block)

    def get_all_neurons_for_pubkey( self, hotkey_ss58: str, block: Optional[int] = None ) -> List[NeuronInfo]:
        netuids = self.get_netuids_for_hotkey( hotkey_ss58, block) 
        uids = [self.get_uid_for_hotkey_on_subnet(hotkey_ss58, net) for net in netuids] 
        return [self.neuron_for_uid( uid, net ) for uid, net in list(zip(uids, netuids))]

    def neuron_has_validator_permit( self, uid: int, netuid: int, block: Optional[int] = None ) -> Optional[bool]:
        return self.query_subtensor( 'ValidatorPermit', block, [ netuid, uid ] ).value

    def neuron_for_wallet( self, wallet: 'bittensor.Wallet', netuid = int, block: Optional[int] = None ) -> Optional[NeuronInfo]: 
        return self.get_neuron_for_pubkey_and_subnet ( wallet.hotkey.ss58_address, netuid = netuid, block = block )

    def neuron_for_uid( self, uid: int, netuid: int, block: Optional[int] = None ) -> Optional[NeuronInfo]: 
        r""" Returns a list of neuron from the chain. 
        Args:
            uid ( int ):
                The uid of the neuron to query for.
            netuid ( int ):
                The uid of the network to query for.
            block ( int ):
                The neuron at a particular block
        Returns:
            neuron (Optional[NeuronInfo]):
                neuron metadata associated with uid or None if it does not exist.
        """
        if uid == None: return NeuronInfo._null_neuron()
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [netuid, uid]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="neuronInfo_getNeuron", # custom rpc method
                    params=params
                )
        json_body = make_substrate_call_with_retry()
        result = json_body['result']

        if result in (None, []):
            return NeuronInfo._null_neuron()
        
        return NeuronInfo.from_vec_u8( result ) 

    def neurons(self, netuid: int, block: Optional[int] = None ) -> List[NeuronInfo]: 
        r""" Returns a list of neuron from the chain. 
        Args:
            netuid ( int ):
                The netuid of the subnet to pull neurons from.
            block ( Optional[int] ):
                block to sync from.
        Returns:
            neuron (List[NeuronInfo]):
                List of neuron metadata objects.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [netuid]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="neuronInfo_getNeurons", # custom rpc method
                    params=params
                )
        
        json_body = make_substrate_call_with_retry()
        result = json_body['result']

        if result in (None, []):
            return []
        
        return NeuronInfo.list_from_vec_u8( result )
    
    def neuron_for_uid_lite( self, uid: int, netuid: int, block: Optional[int] = None ) -> Optional[NeuronInfoLite]: 
        r""" Returns a list of neuron lite from the chain. 
        Args:
            uid ( int ):
                The uid of the neuron to query for.
            netuid ( int ):
                The uid of the network to query for.
            block ( int ):
                The neuron at a particular block
        Returns:
            neuron (Optional[NeuronInfoLite]):
                neuron metadata associated with uid or None if it does not exist.
        """
        if uid == None: return NeuronInfoLite._null_neuron()
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [netuid, uid]
                if block_hash:
                    params = params + [block_hash] 
                return substrate.rpc_request(
                    method="neuronInfo_getNeuronLite", # custom rpc method
                    params=params
                )
        json_body = make_substrate_call_with_retry()
        result = json_body['result']

        if result in (None, []):
            return NeuronInfoLite._null_neuron()
        
        return NeuronInfoLite.from_vec_u8( result ) 

    def neurons_lite(self, netuid: int, block: Optional[int] = None ) -> List[NeuronInfoLite]: 
        r""" Returns a list of neuron lite from the chain. 
        Args:
            netuid ( int ):
                The netuid of the subnet to pull neurons from.
            block ( Optional[int] ):
                block to sync from.
        Returns:
            neuron (List[NeuronInfoLite]):
                List of neuron lite metadata objects.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [netuid]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="neuronInfo_getNeuronsLite", # custom rpc method
                    params=params
                )
        
        json_body = make_substrate_call_with_retry()
        result = json_body['result']

        if result in (None, []):
            return []
        
        return NeuronInfoLite.list_from_vec_u8( result )

    def metagraph( self, netuid: int, block: Optional[int] = None, lite: bool = True ) -> 'bittensor.Metagraph':
        r""" Returns the metagraph for the subnet.
        Args:
            netuid ( int ):
                The network uid of the subnet to query.
            block (Optional[int]):
                The block to create the metagraph for.
                Defaults to latest.
            lite (bool, default=True):
                If true, returns a metagraph using the lite sync (no weights, no bonds)
        Returns:
            metagraph ( `bittensor.Metagraph` ):
                The metagraph for the subnet at the block.
        """
        status: Optional['rich.console.Status'] = None
        if bittensor.__use_console__:
            status = bittensor.__console__.status("Synchronizing Metagraph...", spinner="earth")
            status.start()
        
        # Get neurons.
        if lite:
            neurons = self.neurons_lite( netuid = netuid, block = block )
        else:
            neurons = self.neurons( netuid = netuid, block = block )
        
        # Get subnet info.
        subnet_info: Optional[bittensor.SubnetInfo] = self.get_subnet_info( netuid = netuid, block = block )
        if subnet_info == None:
            status.stop() if status else ...
            raise ValueError('Could not find subnet info for netuid: {}'.format(netuid))

        status.stop() if status else ...

        # Create metagraph.
        block_number = self.block
        
        metagraph = bittensor.metagraph.from_neurons( network = self.network, netuid = netuid, info = subnet_info, neurons = neurons, block = block_number )
        print("Metagraph subtensor: ", self.network)
        return metagraph

    ################
    #### Transfer ##
    ################


    

    ################
    #### Legacy ####
    ################

    def get_balance(self, address: str, block: int = None) -> Balance:
        r""" Returns the token balance for the passed ss58_address address
        Args:
            address (Substrate address format, default = 42):
                ss58 chain address.
        Return:
            balance (bittensor.utils.balance.Balance):
                account balance
        """
        try:
            @retry(delay=2, tries=3, backoff=2, max_delay=4)
            def make_substrate_call_with_retry():
                with self.substrate as substrate:
                    return substrate.query(
                        module='System',
                        storage_function='Account',
                        params=[address],
                        block_hash = None if block == None else substrate.get_block_hash( block )
                    )
            result = make_substrate_call_with_retry()
        except scalecodec.exceptions.RemainingScaleBytesNotEmptyException:
            logger.critical("Your wallet it legacy formatted, you need to run btcli stake --ammount 0 to reformat it." )
            return Balance(1000)
        return Balance( result.value['data']['free'] )

    def get_current_block(self) -> int:
        r""" Returns the current block number on the chain.
        Returns:
            block_number (int):
                Current chain blocknumber.
        """        
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.get_block_number(None)
        return make_substrate_call_with_retry()

    def get_balances(self, block: int = None) -> Dict[str, Balance]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query_map(
                    module='System',
                    storage_function='Account',
                    block_hash = None if block == None else substrate.get_block_hash( block )
                )
        result = make_substrate_call_with_retry()
        return_dict = {}
        for r in result:
            bal = bittensor.Balance( int( r[1]['data']['free'].value ) )
            return_dict[r[0].value] = bal
        return return_dict

    @staticmethod
    def _null_neuron() -> NeuronInfo:
        neuron = NeuronInfo(
            uid = 0,
            netuid = 0,
            active =  0,
            stake = '0',
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
            hotkey = "000000000000000000000000000000000000000000000000"
        )
        return neuron
