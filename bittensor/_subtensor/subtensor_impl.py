# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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
import torch
from rich.prompt import Confirm
from typing import List, Dict, Union, Optional, Tuple

from dataclasses import dataclass

import bittensor
from tqdm import tqdm
import bittensor.utils.networking as net
import bittensor.utils.weight_utils as weight_utils
from retry import retry
from substrateinterface import SubstrateInterface
from bittensor.utils.balance import Balance
from bittensor.utils import is_valid_bittensor_address_or_public_key
import json

# Mocking imports
import os
import random
import scalecodec
import time
import subprocess
from sys import platform   

from loguru import logger
logger = logger.opt(colors=True)


RAOPERTAO = 1000000000
U16_MAX = 65535
U64_MAX = 18446744073709551615

@dataclass
class NeuronMetadata:
    r"""
    Dataclass for neuron metadata.
    """
    hotkey: str
    coldkey: str
    uid: int
    netuid: int
    active: int
    ip: str
    ip_type: int
    port: int
    stake: List[Tuple[str, Balance]]
    rank: float
    emission: float
    incentive: float
    consensus: float
    trust: float
    dividends: float
    last_update: int
    version: int
    weights: List[List[int]]
    bonds: List[List[int]]
    prometheus_info: 'PrometheusInfo'
    is_null: bool = False

    @staticmethod
    def __u8_key_to_ss58(u8_key: Dict) -> str:
        r""" Converts a u8 key to ss58.
        """
        # First byte is length, then 32 bytes of key.
        return scalecodec.ss58_encode( bytes(u8_key['id'][1:33]).hex(), bittensor.__ss58_format__)
        
    @classmethod
    def from_json(cls, json: Dict) -> 'NeuronMetadata':
        r""" Returns a NeuronMetadata object from a json dictionary.
        """
        return NeuronMetadata(
            hotkey = cls.__u8_key_to_ss58(json['hotkey']),
            coldkey = cls.__u8_key_to_ss58(json['coldkey']),
            uid = json['uid'],
            netuid = json['netuid'],
            active = int(json['active']), # 0 or 1
            ip = bittensor.utils.networking.int_to_ip(int(json['axon_info']['ip'])),
            ip_type = json['axon_info']['ip_type'],
            port = json['axon_info']['port'],
            stake = [(cls.__u8_key_to_ss58(coldkey), Balance.from_rao(stake) ) for coldkey, stake in json['stake']],
            rank = json['rank'] / U64_MAX,
            emission = json['emission'] / RAOPERTAO,
            incentive = json['incentive'] / U64_MAX,
            consensus = json['consensus'] / U64_MAX,
            trust = json['trust'] / U64_MAX,
            dividends = json['dividends'] / U64_MAX,
            last_update = json['last_update'],
            version = json['axon_info']['version'],
            weights = json['weights'],
            bonds = json['bonds'],
            prometheus_info = PrometheusInfo.from_json(json['prometheus_info']),
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

class ChainError(BaseException):
    r""" Base error for any chain related errors.
    """
    pass


class ChainConnectionError(ChainError):
    r""" Error for any chain connection related errors.
    """
    pass


class ChainTransactionError(ChainError):
    r""" Error for any chain transaction related errors.
    """
    pass


class ChainQueryError(ChainError):
    r""" Error for any chain query related errors.
    """
    pass


class StakeError(ChainTransactionError):
    r""" Error raised when a stake transaction fails.
    """
    pass


class UnstakeError(ChainTransactionError):
    r""" Error raised when an unstake transaction fails.
    """
    pass


class TransferError(ChainTransactionError):
    r""" Error raised when a transfer transaction fails.
    """
    pass


class RegistrationError(ChainTransactionError):
    r""" Error raised when a neuron registration transaction fails.
    """
    pass


class NotRegisteredError(ChainTransactionError):
    r""" Error raised when a neuron is not registered, and the transaction requires it to be.
    """
    pass


class NotDelegateError(StakeError):
    r""" Error raised when a hotkey you are trying to stake to is not a delegate.
    """
    pass


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
  
    def endpoint_for_network( 
            self,
            blacklist: List[str] = [] 
        ) -> str:
        r""" Returns a chain endpoint based on self.network.
            Returns None if there are no available endpoints.
        """

        # Chain endpoint overrides the --network flag.
        if self.chain_endpoint != None:
            if self.chain_endpoint in blacklist:
                return None
            else:
                return self.chain_endpoint

    def connect( self, timeout: int = 10, failure = True ) -> bool:
        attempted_endpoints = []
        while True:
            def connection_error_message():
                print(
                '\n'.join((
                    "Check that your internet connection is working and the chain endpoints are available: <blue>{}</blue>",
                    "The subtensor.network should likely be one of the following choices:",
                    "    -- local - (your locally running node)",
                    "    -- nobunaga - (staging)",
                    "    -- nakamoto - (main)",
                    "Or you may set the endpoint manually using the --subtensor.chain_endpoint flag",
                    "To run a local node (See: docs/running_a_validator.md)"
                ))
                .format( attempted_endpoints) )

            # ---- Get next endpoint ----
            ws_chain_endpoint = self.endpoint_for_network( blacklist = attempted_endpoints )
            if ws_chain_endpoint == None:
                logger.error("No more endpoints available for subtensor.network: <blue>{}</blue>, attempted: <blue>{}</blue>".format(self.network, attempted_endpoints))
                connection_error_message()
                if failure:
                    logger.critical('Unable to connect to network:<blue>{}</blue>.\nMake sure your internet connection is stable and the network is properly set.'.format(self.network))
                else:
                    return False
            attempted_endpoints.append(ws_chain_endpoint)

            # --- Attempt connection ----
            try:
                with self.substrate:
                    logger.success("Network:".ljust(20) + "<blue>{}</blue>", self.network)
                    logger.success("Endpoint:".ljust(20) + "<blue>{}</blue>", ws_chain_endpoint)
                    return True
            
            except Exception:
                logger.error( "Error while connecting to network:<blue>{}</blue> at endpoint: <blue>{}</blue>".format(self.network, ws_chain_endpoint))
                connection_error_message()
                if failure:
                    raise RuntimeError('Unable to connect to network:<blue>{}</blue>.\nMake sure your internet connection is stable and the network is properly set.'.format(self.network))
                else:
                    return False

    def rho (self, netuid: int) -> int:
        r""" Incentive mechanism rho parameter.
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            rho (int):
                Incentive mechanism rho parameter.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'Rho',
                    params = [netuid]
                ).value
        return make_substrate_call_with_retry()

    def kappa (self, netuid: int) -> int:
        r""" Incentive mechanism kappa parameter.
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            kappa (int):
                Incentive mechanism kappa parameter.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'Kappa',
                    params = [netuid]
                ).value
        return make_substrate_call_with_retry()

    def difficulty (self, netuid: int) -> int:
        r""" Returns registration difficulty from the chain.
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            difficulty (int):
                Registration difficulty.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'Difficulty',
                    params = [netuid]
                ).value
        return make_substrate_call_with_retry()

    @property
    def total_issuance (self) -> 'bittensor.Balance':
        r""" Returns the total token issuance.
        Returns:
            total_issuance (int):
                Total issuance as balance.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return bittensor.Balance.from_rao( substrate.query(  module='Paratensor', storage_function = 'TotalIssuance').value )
        return make_substrate_call_with_retry()

    def immunity_period (self, netuid: int) -> int:
        r""" Returns the chain registration immunity_period
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            immunity_period (int):
                Chain registration immunity_period
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'ImmunityPeriod',
                    params = [netuid]
                ).value
        return make_substrate_call_with_retry()

    def serving_rate_limit ( self, netuid: int ) -> int:
        r"""
        Returns the subnet serving rate limit.
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            serving_rate_limit (int):
                Subnet serving rate limit.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'ServingRateLimit',
                    params = [netuid]
                ).value
        
        return make_substrate_call_with_retry()

    def validator_batch_size (self, netuid: int) -> int:
        r""" Returns the subnet default validator batch size.
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            batch_size (int):
                subnet default validator batch size.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'ValidatorBatchSize',
                    params = [netuid]
                ).value
        return make_substrate_call_with_retry()

    def validator_sequence_length (self, netuid: int) -> int:
        r""" Returns the subnet default validator sequence length.
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            sequence_length (int):
                subnet default validator sequence length.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'ValidatorSequenceLength',
                    params = [netuid]
                ).value
        return make_substrate_call_with_retry()

    def validator_epochs_per_reset (self, netuid: int) -> int:
        r""" Epochs passed before the validator resets its weights.
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            validator_epochs_per_reset (int):
                Epochs passed before the validator resets its weights.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'ValidatorEpochsPerReset',
                    params = [netuid]
                ).value
        return make_substrate_call_with_retry()

    def validator_epoch_length (self, netuid: int) -> int:
        r""" Default subnet validator epoch length.
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            validator_epoch_length (int):
                Default validator epoch length. 
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'ValidatorEpochLen',
                    params = [netuid]
                ).value
        return make_substrate_call_with_retry()

    @property
    def total_stake (self) -> 'bittensor.Balance':
        r""" Returns total stake on the chain.
        Returns:
            total_stake (bittensor.Balance):
                Total stake as balance.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return bittensor.Balance.from_rao( substrate.query(  module='Paratensor', storage_function = 'TotalStake' ).value )
        return make_substrate_call_with_retry()

    def min_allowed_weights (self, netuid: int) -> int:
        r""" Returns min allowed number of weights.
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            min_allowed_weights (int):
                Min number of weights allowed to be set.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'MinAllowedWeights',
                    params = [netuid]
                ).value
        return make_substrate_call_with_retry()

    def max_weight_limit (self, netuid: int) -> int:
        r""" Returns MaxWeightLimit
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            max_weight (int):
                the max value for weights after normalizaiton
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                U32_MAX = 4294967295
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'MaxWeightLimit',
                    params = [netuid]
                ).value/U32_MAX
        return make_substrate_call_with_retry()

    def scaling_law_power (self, netuid: int) -> int:
        r""" Returns ScalingLawPower
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            ScalingLawPower (float):
                the power term attached to scaling law
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                MAX = 100
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'ScalingLawPower',
                    params = [netuid]
                ).value/MAX
        return make_substrate_call_with_retry()

    def synergy_scaling_law_power (self, netuid: int) -> int:
        r""" Returns SynergyScalingLawPower
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            SynergyScalingLawPower (float):
                the term attached to synergy calculation during shapley scores
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                MAX = 100
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'SynergyScalingLawPower',
                    params = [netuid]
                ).value/MAX
        return make_substrate_call_with_retry()

    def validator_exclude_quantile (self, netuid: int) -> int:
        r""" Returns ValidatorExcludeQuantile
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            ValidatorExcludeQuantile (float):
                the quantile that validators should exclude when setting their weights
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                MAX = 100
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'ValidatorExcludeQuantile',
                    params = [netuid]
                ).value/MAX
        return make_substrate_call_with_retry()

    def max_allowed_min_max_ratio(self, netuid: int) -> int:
        r""" Returns the chains max_allowed_min_max_ratio
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            max_allowed_min_max_ratio (int):
                The max ratio allowed between the min and max.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'MaxAllowedMaxMinRatio',
                    params = [netuid]
                ).value
        return make_substrate_call_with_retry()

    def n (self, netuid: int) -> int:
        r""" Returns total number of neurons on the chain for this subnetwork.
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            n (int):
                Total number of neurons on chain.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'N',
                    params = [netuid]
                ).value
        return make_substrate_call_with_retry()

    def max_n (self, netuid: int) -> int:
        r""" Returns maximum number of neuron uids allowed in this subnetwork.
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            max_n (int):
                Maximum number of neuron positions on the graph.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'MaxAllowedUids',
                    params = [netuid]
                ).value
        return make_substrate_call_with_retry()

    @property
    def block (self) -> int:
        r""" Returns current chain block.
        Returns:
            block (int):
                Current chain block.
        """
        return self.get_current_block()

    def blocks_since_epoch (self, netuid: int) -> int:
        r""" Returns blocks since last epoch.
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            blocks_since_epoch (int):
                blocks_since_epoch 
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'BlocksSinceLastStep',
                    params = [netuid]
                ).value
        return make_substrate_call_with_retry()

    def tempo (self, netuid: int) -> int:
        r""" Returns the tempo for this subnetwork.
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            blocks_since_epoch (int):
                blocks_since_epoch 
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'Tempo',
                    params = [netuid]
                ).value
        return make_substrate_call_with_retry()

    def blocks_per_epoch (self, netuid: int) -> int:
        r""" Returns blocks per chain epoch.
        Args:
            netuid (int):
                The netuid of the subnet to query.
        Returns:
            blocks_per_epoch (int):
                blocks_per_epoch 
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'BlocksPerStep',
                    params = [netuid]
                ).value
        return make_substrate_call_with_retry()


    def serve_axon (
        self,
        axon: 'bittensor.Axon',
        use_upnpc: bool = False,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False,
    ) -> bool:
        r""" Serves the axon to the network.
        Args:
            axon (bittensor.Axon):
                Axon to serve.
            use_upnpc (:type:bool, `optional`): 
                If true, the axon attempts port forward through your router before 
                subscribing.                
            wait_for_inclusion (bool):
                If set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                If set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """
        axon.wallet.hotkey
        axon.wallet.coldkeypub

        # ---- Setup UPNPC ----
        if use_upnpc:
            if prompt:
                if not Confirm.ask("Attempt port forwarding with upnpc?"):
                    return False
            try:
                external_port = net.upnpc_create_port_map( port = axon.port )
                bittensor.__console__.print(":white_heavy_check_mark: [green]Forwarded port: {}[/green]".format( axon.port ))
                bittensor.logging.success(prefix = 'Forwarded port', sufix = '<blue>{}</blue>'.format( axon.port ))
            except net.UPNPCException as upnpc_exception:
                raise RuntimeError('Failed to hole-punch with upnpc with exception {}'.format( upnpc_exception )) from upnpc_exception
        else:
            external_port = axon.external_port

        # ---- Get external ip ----
        if axon.external_ip == None:
            try:
                external_ip = net.get_external_ip()
                bittensor.__console__.print(":white_heavy_check_mark: [green]Found external ip: {}[/green]".format( external_ip ))
                bittensor.logging.success(prefix = 'External IP', sufix = '<blue>{}</blue>'.format( external_ip ))
            except Exception as E:
                raise RuntimeError('Unable to attain your external ip. Check your internet connection. error: {}'.format(E)) from E
        else:
            external_ip = axon.external_ip
        
        # ---- Subscribe to chain ----
        serve_success = self.serve(
                wallet = axon.wallet,
                ip = external_ip,
                port = external_port,
                netuid = axon.netuid,
                protocol = axon.protocol,
                wait_for_inclusion = wait_for_inclusion,
                wait_for_finalization = wait_for_finalization,
                prompt = prompt
        )
        return serve_success

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
        r""" Registers the wallet to chain.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            netuid (int):
                The netuid of the subnet to register on.
            wait_for_inclusion (bool):
                If set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                If set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
            max_allowed_attempts (int):
                Maximum number of attempts to register the wallet.
            cuda (bool):
                If true, the wallet should be registered using CUDA device(s).
            dev_id (Union[List[int], int]):
                The CUDA device id to use, or a list of device ids.
            TPB (int):
                The number of threads per block (CUDA).
            num_processes (int):
                The number of processes to use to register.
            update_interval (int):
                The number of nonces to solve between updates.
            log_verbose (bool):
                If true, the registration process will log more information.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """

        with bittensor.__console__.status(f":satellite: Checking Account on [bold]subnet:{netuid}[/bold]..."):
            neuron = self.neuron_for_pubkey( wallet.hotkey.ss58_address, netuid = netuid )
            if not neuron.is_null:
                bittensor.__console__.print(
                ":white_heavy_check_mark: [green]Already Registered[/green]:\n" + \
                "  uid: [bold white]{}[/bold white]\n" + \
                "  netuid: [bold white]{}[/bold white]\n"
                "  hotkey: [bold white]{}[/bold white]\n" + \
                "  coldkey: [bold white]{}[/bold white]"
                    .format(neuron.uid, neuron.netuid, neuron.hotkey, neuron.coldkey))
                return True

        if prompt:
            if not Confirm.ask("Continue Registration?\n  hotkey:     [bold white]{}[/bold white]\n  coldkey:    [bold white]{}[/bold white]\n  network:    [bold white]{}[/bold white]".format( wallet.hotkey.ss58_address, wallet.coldkeypub.ss58_address, self.network ) ):
                return False

        # Attempt rolling registration.
        attempts = 1
        while True:
            bittensor.__console__.print(":satellite: Registering...({}/{})".format(attempts, max_allowed_attempts))
            # Solve latest POW.
            if cuda:
                if not torch.cuda.is_available():
                    if prompt:
                        bittensor.__console__.error('CUDA is not available.')
                    return False
                pow_result = bittensor.utils.create_pow( self, wallet, netuid, output_in_place, cuda, dev_id, TPB, num_processes=num_processes, update_interval=update_interval, log_verbose=log_verbose )
            else:
                pow_result = bittensor.utils.create_pow( self, wallet, netuid, output_in_place, num_processes=num_processes, update_interval=update_interval, log_verbose=log_verbose )

            # pow failed
            if not pow_result:
                # might be registered already on this subnet
                if (wallet.is_registered( subtensor = self, netuid = netuid )):
                    bittensor.__console__.print(f":white_heavy_check_mark: [green]Already registered on netuid:{netuid}[/green]")
                    return True
                
            # pow successful, proceed to submit pow to chain for registration
            else:
                with bittensor.__console__.status(":satellite: Submitting POW..."):
                    # check if pow result is still valid
                    while bittensor.utils.POWNotStale(self, pow_result):
                        with self.substrate as substrate:
                            # create extrinsic call
                            call = substrate.compose_call( 
                                call_module='Paratensor',  
                                call_function='register', 
                                call_params={ 
                                    'block_number': pow_result['block_number'], 
                                    'nonce': pow_result['nonce'], 
                                    'work': bittensor.utils.hex_bytes_to_u8_list( pow_result['work'] ), 
                                    'hotkey': wallet.hotkey.ss58_address, 
                                    'coldkey': wallet.coldkeypub.ss58_address,
                                    'netuid': netuid,
                                } 
                            )
                            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.hotkey )
                            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization )
                            
                            # We only wait here if we expect finalization.
                            if not wait_for_finalization and not wait_for_inclusion:
                                bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                                return True
                            
                            # process if registration successful, try again if pow is still valid
                            response.process_events()
                            if not response.is_success:
                                if 'key is already registered' in response.error_message:
                                    # Error meant that the key is already registered.
                                    bittensor.__console__.print(f":white_heavy_check_mark: [green]Already Registered on [bold]subnet:{netuid}[/bold][/green]")
                                    return True

                                bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))
                                time.sleep(0.5)
                            
                            # Successful registration, final check for neuron and pubkey
                            else:
                                bittensor.__console__.print(":satellite: Checking Balance...")
                                is_registered = wallet.is_registered( subtensor = self, netuid = netuid )
                                if is_registered:
                                    bittensor.__console__.print(":white_heavy_check_mark: [green]Registered[/green]")
                                    return True
                                else:
                                    # neuron not found, try again
                                    bittensor.__console__.print(":cross_mark: [red]Unknown error. Neuron not found.[/red]")
                                    continue
                    else:
                        # Exited loop because pow is no longer valid.
                        bittensor.__console__.print( "[red]POW is stale.[/red]" )
                        # Try again.
                        continue
                        
            if attempts < max_allowed_attempts:
                #Failed registration, retry pow
                attempts += 1
                bittensor.__console__.print( ":satellite: Failed registration, retrying pow ...({}/{})".format(attempts, max_allowed_attempts))
            else:
                # Failed to register after max attempts.
                bittensor.__console__.print( "[red]No more attempts.[/red]" )
                return False 

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
        r""" Subscribes an bittensor endpoint to the substensor chain.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            ip (str):
                endpoint host port i.e. 192.122.31.4
            port (int):
                endpoint port number i.e. 9221
            protocol (int):
                int representation of the protocol 
            netuid (int):
                network uid to serve on.
            placeholder1 (int):
                placeholder for future use.
            placeholder2 (int):
                placeholder for future use.
            wait_for_inclusion (bool):
                if set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                if set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """

        # Decrypt hotkey
        wallet.hotkey

        params = {
            'version': bittensor.__version_as_int__,
            'ip': net.ip_to_int(ip),
            'port': port,
            'ip_type': net.ip_version(ip),
            'protocol': protocol,
            'placeholder1': placeholder1,
            'placeholder2': placeholder2,
            'netuid': netuid,
            'coldkey': wallet.coldkeypub.ss58_address,
        }

        with bittensor.__console__.status(":satellite: Checking Axon..."):
            neuron = self.neuron_for_pubkey( wallet.hotkey.ss58_address, netuid = netuid )
            neuron_up_to_date = not neuron.is_null and params == {
                'version': neuron.version,
                'ip': neuron.ip,
                'port': neuron.port,
                'ip_type': neuron.ip_type,
                'netuid': neuron.netuid,
                'coldkey': neuron.coldkey,
                'protocol': neuron.protocol,
                'placeholder1': neuron.placeholder1,
                'placeholder2': neuron.placeholder2,
            }
            if neuron_up_to_date:
                bittensor.__console__.print(":white_heavy_check_mark: [green]Already Served[/green]\n  [bold white]ip: {}\n  port: {}\n  modality: {}\n  hotkey: {}\n  coldkey: {}[/bold white]".format(ip, port, modality, wallet.hotkey.ss58_address, wallet.coldkeypub.ss58_address))
                return True

        if prompt:
            output = params.copy()
            output['coldkey'] = wallet.coldkeypub.ss58_address
            output['hotkey'] = wallet.hotkey.ss58_address
            if not Confirm.ask("Do you want to serve axon:\n  [bold white]{}[/bold white]".format(
                json.dumps(output, indent=4, sort_keys=True)
            )):
                return False
        
        with bittensor.__console__.status(":satellite: Serving axon on: [white]{}:{}[/white] ...".format(self.network, netuid)):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='Paratensor',
                    call_function='serve_axon',
                    call_params=params
                )
                extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.hotkey)
                response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                if wait_for_inclusion or wait_for_finalization:
                    response.process_events()
                    if response.is_success:
                        bittensor.__console__.print(':white_heavy_check_mark: [green]Served[/green]\n  [bold white]{}[/bold white]'.format(json.dumps(params, indent=4, sort_keys=True) ))
                        return True
                    else:
                        bittensor.__console__.print(':cross_mark: [green]Failed to Serve axon[/green] error: {}'.format(response.error_message))
                        return False
                else:
                    return True

    def get_total_stake_for_hotkey(
        self,
        ss58_address: str,
        block: Optional[int] = None,
    ) -> Optional['bittensor.Balance']:
        r"""
        Returns the stake of the specified account.
        Args:
            ss58_address (str):
                ss58 address of the account to check.
            block (Optional[int]):
                Block to query. If None, the latest finalized block is used.
        Returns:
            stake (Optional[bittensor.Balance]):
                Stake of the account or None if the account does not exist.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function='TotalHotkeyStake',
                    params = [ss58_address],
                    block_hash = None if block == None else substrate.get_block_hash( block )
                )
        result = make_substrate_call_with_retry()
        if result != None:
            return bittensor.Balance.from_rao(result.value)
        else:
            return None

    def get_total_stake_for_coldkey(
        self,
        ss58_address: str,
        block: Optional[int] = None,
    ) -> Optional['bittensor.Balance']:
        r"""
        Returns the stake of the specified account.
        Args:
            ss58_address (str):
                ss58 address of the coldkey account to check.
            block (Optional[int]):
                Block to query. If None, the latest finalized block is used.
        Returns:
            stake (Optional[bittensor.Balance]):
                Stake of the account or None if the account does not exist.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function='TotalColdkeyStake',
                    params = [ss58_address],
                    block_hash = None if block == None else substrate.get_block_hash( block )
                )
        result = make_substrate_call_with_retry()
        if result != None:
            return bittensor.Balance.from_rao(result.value)
        else:
            return None

    def get_stake_for_coldkey_and_hotkey(
        self,
        hotkey_ss58: str,
        coldkey_ss58: str,
        block: Optional[int] = None,
    ) -> Optional['bittensor.Balance']:
        r"""
        Returns the stake of the specified account.
        Args:
            hotkey_ss58 (str):
                ss58 address of the hotkey account to check.
            coldkey_ss58 (str):
                ss58 address of the hotkey account to check.
            block (Optional[int]):
                Block to query. If None, the latest finalized block is used.
        Returns:
            stake (Optional[bittensor.Balance]):
                Stake of the account or None if the account does not exist.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function='Stake',
                    params = [hotkey_ss58, coldkey_ss58],
                    block_hash = None if block == None else substrate.get_block_hash( block )
                )
        result = make_substrate_call_with_retry()
        if result != None:
            return bittensor.Balance.from_rao(result.value)
        else:
            return None

    def get_owner_for_hotkey(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
    ) -> str:
        """
        Returns the owner of the specified hotkey. (coldkey)
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function='Owner',
                    params = [hotkey_ss58],
                    block_hash = None if block == None else substrate.get_block_hash( block )
                )
        result = make_substrate_call_with_retry()
        if result != None:
            return result
        else:
            return None

    def get_delegate_take(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Returns the delegate take of the specified hotkey.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function='Delegates',
                    params = [hotkey_ss58],
                    block_hash = None if block == None else substrate.get_block_hash( block )
                )
        result = make_substrate_call_with_retry()
        if result != None:
            return result.value / U16_MAX
        else:
            return None
    
    def add_stake(
            self, 
            wallet: 'bittensor.wallet',
            hotkey_ss58: Optional[str] = None,
            amount: Union[Balance, float] = None, 
            wait_for_inclusion: bool = True,
            wait_for_finalization: bool = False,
            prompt: bool = False,
        ) -> bool:
        r""" Adds the specified amount of stake to passed hotkey uid.
        Args:
            wallet (bittensor.wallet):
                Bittensor wallet object.
            hotkey_ss58 (Optional[str]):
                ss58 address of the hotkey account to stake to
                defaults to the wallet's hotkey.
            amount (Union[Balance, float]):
                Amount to stake as bittensor balance, or float interpreted as Tao.
            wait_for_inclusion (bool):
                If set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                If set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.

        Raises:
            NotRegisteredError:
                If the wallet is not registered on the chain.
            NotDelegateError:
                If the hotkey is not a delegate on the chain.
        """
        # Decrypt keys,
        wallet.coldkey

        own_hotkey: bool = False # Flag to indicate if we are using the wallet's own hotkey.
        if hotkey_ss58 is None:
            hotkey_ss58 = wallet.hotkey.ss58_address # Default to wallet's own hotkey.
            own_hotkey = True

        with bittensor.__console__.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(self.network)):
            old_balance = self.get_balance( wallet.coldkey.ss58_address )

            if not self.is_hotkey_registered( hotkey_ss58 ): # Hotkey is not registered on the chain.
                raise NotRegisteredError("Hotkey: {} is not registered.".format(hotkey_ss58))

            if not own_hotkey:
                # This is not the wallet's own hotkey so we are delegating.
                if not self.is_hotkey_delegate( hotkey_ss58 ):
                    raise NotDelegateError("Hotkey: {} is not a delegate.".format(hotkey_ss58))
                
                # Get hotkey take
                hotkey_take = self.get_hotkey_take( hotkey_ss58 )
                # Get hotkey owner
                hotkey_owner = self.get_owner_for_hotkey( hotkey_ss58 )
            
            # Get current stake
            old_stake = self.get_stake_for_coldkey_and_hotkey( coldkey_ss58=wallet.coldkeypub.ss58_address, hotkey_ss58=hotkey_ss58 )

        # Convert to bittensor.Balance
        if amount == None:
            # Stake it all.
            staking_balance = bittensor.Balance.from_tao( old_balance.tao )
        elif not isinstance(amount, bittensor.Balance ):
            staking_balance = bittensor.Balance.from_tao( amount )
        else:
            staking_balance = amount

        # Remove existential balance to keep key alive.
        if staking_balance > bittensor.Balance.from_rao( 1000 ):
            staking_balance = staking_balance - bittensor.Balance.from_rao( 1000 )
        else:
            staking_balance = staking_balance

        # Estimate transfer fee.
        staking_fee = None # To be filled.
        with bittensor.__console__.status(":satellite: Estimating Staking Fees..."):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='Paratensor', 
                    call_function='add_stake',
                    call_params={
                        'hotkey': hotkey_ss58,
                        'amount_staked': staking_balance.rao
                    }
                )
                payment_info = substrate.get_payment_info(call = call, keypair = wallet.coldkey)
                if payment_info:
                    staking_fee = bittensor.Balance.from_rao(payment_info['partialFee'])
                    bittensor.__console__.print("[green]Estimated Fee: {}[/green]".format( staking_fee ))
                else:
                    staking_fee = bittensor.Balance.from_tao( 0.2 )
                    bittensor.__console__.print(":cross_mark: [red]Failed[/red]: could not estimate staking fee, assuming base fee of 0.2")

        # Check enough to stake.
        if staking_balance > old_balance + staking_fee:
            bittensor.__console__.print(":cross_mark: [red]Not enough stake[/red]:[bold white]\n  balance:{}\n  amount: {}\n  fee: {}\n  coldkey: {}[/bold white]".format(old_balance, staking_balance, staking_fee, wallet.name))
            return False
                
        # Ask before moving on.
        if prompt:
            if not own_hotkey:
                # We are delegating.
                if not Confirm.ask("Do you want to delegate:[bold white]\n  amount: {}\n  to: {}\n  fee: {}\n  take: {}\n  owner: {}[/bold white]".format( staking_balance, wallet.hotkey_str, staking_fee, hotkey_take, hotkey_owner) ):
                    return False
            else:
                if not Confirm.ask("Do you want to stake:[bold white]\n  amount: {}\n  to: {}\n  fee: {}[/bold white]".format( staking_balance, wallet.hotkey_str, staking_fee) ):
                    return False

        try:
            with bittensor.__console__.status(":satellite: Staking to: [bold white]{}[/bold white] ...".format(self.network)):
                staking_response: bool = self.__do_add_stake_single(
                    wallet = wallet,
                    hotkey_ss58 = hotkey_ss58,
                    amount = staking_balance,
                    wait_for_inclusion = wait_for_inclusion,
                    wait_for_finalization = wait_for_finalization,
                )

            if staking_response: # If we successfully staked.
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                    return True

                bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")
                with bittensor.__console__.status(":satellite: Checking Balance on: [white]{}[/white] ...".format(self.network)):
                    new_balance = self.get_balance( address = wallet.coldkey.ss58_address )
                    block = self.get_current_block()
                    new_stake = self.get_stake_for_coldkey_and_hotkey(
                        coldkey_ss58=wallet.coldkeypub.ss58_address,
                        hotkey_ss58= wallet.hotkey.ss58_address,
                        block=block
                    ) # Get current stake

                    bittensor.__console__.print("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_balance, new_balance ))
                    bittensor.__console__.print("Stake:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_stake, new_stake ))
                    return True
            else:
                bittensor.__console__.print(":cross_mark: [red]Failed[/red]: Error unknown.")
                return False

        except NotRegisteredError as e:
            bittensor.__console__.print(":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(wallet.hotkey_str))
            return False
        except StakeError as e:
            bittensor.__console__.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
            return False

    def __do_add_stake_single(
        self, 
        wallet: 'bittensor.wallet',
        hotkey_ss58: str,
        amount: 'bittensor.Balance', 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        r"""
        Executes a stake call to the chain using the wallet and amount specified.
        Args:
            wallet (bittensor.wallet):
                Bittensor wallet object.
            hotkey_ss58 (str):
                Hotkey to stake to.
            amount (bittensor.Balance):
                Amount to stake as bittensor balance object.
            wait_for_inclusion (bool):
                If set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                If set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        Raises:
            StakeError:
                If the extrinsic fails to be finalized or included in the block.
            NotDelegateError:
                If the hotkey is not a delegate.
            NotRegisteredError:
                If the hotkey is not registered in any subnets.

        """
        # Decrypt keys,
        wallet.coldkey
        wallet.hotkey

        if not self.is_registered( hotkey_ss58 = hotkey_ss58 ):
            raise NotRegisteredError("Hotkey: {} is not registered.".format(hotkey_ss58))

        if not wallet.hotkey.ss58_address == hotkey_ss58:
            # We are delegating.
            # Verify that the hotkey is a delegate.
            if not self.is_delegate( hotkey_ss58 = hotkey_ss58 ):
                raise NotDelegateError("Hotkey: {} is not a delegate.".format(hotkey_ss58))
    
        with self.substrate as substrate:
            call = substrate.compose_call(
            call_module='Paratensor', 
            call_function='add_stake',
            call_params={
                'hotkey': hotkey_ss58,
                'amount_staked': amount.rao
                }
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            response.process_events()
            if response.is_success:
                return True
            else:
                raise StakeError(response.error_message)

    def add_stake_multiple (
            self, 
            wallet: 'bittensor.wallet',
            hotkey_ss58s: List[str],
            amounts: List[Union[Balance, float]] = None, 
            wait_for_inclusion: bool = True,
            wait_for_finalization: bool = False,
            prompt: bool = False,
        ) -> bool:
        r""" Adds stake to each hotkey_ss58 in the list, using each amount, from a common coldkey.
        Args:
            wallet (bittensor.wallet):
                Bittensor wallet object for the coldkey.
            hotkey_ss58s (List[str]):
                List of hotkeys to stake to.
            amounts (List[Union[Balance, float]]):
                List of amounts to stake. If None, stake all to the first hotkey.
            wait_for_inclusion (bool):
                if set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                if set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or included in the block.
                flag is true if any wallet was staked.
                If we did not wait for finalization / inclusion, the response is true.
        """
        if not isinstance(hotkey_ss58s, list):
            raise TypeError("hotkey_ss58s must be a list of str")
        
        if len(hotkey_ss58s) == 0:
            return True

        if amounts is not None and len(amounts) != len(hotkey_ss58s):
            raise ValueError("amounts must be a list of the same length as hotkey_ss58s")

        if amounts is not None and not all(isinstance(amount, (Balance, float)) for amount in amounts):
            raise TypeError("amounts must be a [list of bittensor.Balance or float] or None")

        if amounts is None:
            amounts = [None] * len(hotkey_ss58s)
        else:
            # Convert to Balance
            amounts = [bittensor.Balance.from_tao(amount) if isinstance(amount, float) else amount for amount in amounts ]

            if sum(amount.tao for amount in amounts) == 0:
                # Staking 0 tao
                return True

        # Decrypt coldkey.
        wallet.coldkey

        old_stakes = []
        with bittensor.__console__.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(self.network)):
            old_balance = self.get_balance( wallet.coldkey.ss58_address )

            # Get the old stakes.
            for hotkey_ss58 in hotkey_ss58s:
                old_stakes.append( self.get_stake_for_coldkey_and_hotkey( coldkey_ss58 = wallet.coldkey.ss58_address, hotkey_ss58 = hotkey_ss58 ) )

        # Remove existential balance to keep key alive.
        ## Keys must maintain a balance of at least 1000 rao to stay alive.
        total_staking_rao = sum([amount.rao if amount is not None else 0 for amount in amounts])
        if total_staking_rao == 0:
            # Staking all to the first wallet.
            if old_balance.rao > 1000:
                old_balance -= bittensor.Balance.from_rao(1000)

        elif total_staking_rao < 1000:
            # Staking less than 1000 rao to the wallets.
            pass
        else:
            # Staking more than 1000 rao to the wallets.
            ## Reduce the amount to stake to each wallet to keep the balance above 1000 rao.
            percent_reduction = 1 - (1000 / total_staking_rao)
            amounts = [Balance.from_tao(amount.tao * percent_reduction) for amount in amounts]
        
        successful_stakes = 0
        for hotkey_ss58, amount, old_stake in zip(hotkey_ss58s, amounts, old_stakes):
            staking_all = False
            # Convert to bittensor.Balance
            if amount == None:
                # Stake it all.
                staking_balance = bittensor.Balance.from_tao( old_balance.tao )
                staking_all = True
            else:
                # Amounts are cast to balance earlier in the function
                assert isinstance(amount, bittensor.Balance)
                staking_balance = amount

            # Estimate staking fee.
            stake_fee = None # To be filled.
            with bittensor.__console__.status(":satellite: Estimating Staking Fees..."):
                with self.substrate as substrate:
                    call = substrate.compose_call(
                    call_module='Paratensor', 
                    call_function='add_stake',
                    call_params={
                        'hotkey': hotkey_ss58,
                        'amount_staked': staking_balance.rao
                        }
                    )
                    payment_info = substrate.get_payment_info(call = call, keypair = wallet.coldkey)
                    if payment_info:
                        stake_fee = bittensor.Balance.from_rao(payment_info['partialFee'])
                        bittensor.__console__.print("[green]Estimated Fee: {}[/green]".format( stake_fee ))
                    else:
                        stake_fee = bittensor.Balance.from_tao( 0.2 )
                        bittensor.__console__.print(":cross_mark: [red]Failed[/red]: could not estimate staking fee, assuming base fee of 0.2")

            # Check enough to stake
            if staking_all:
                staking_balance -= stake_fee
                max(staking_balance, bittensor.Balance.from_tao(0))

            if staking_balance > old_balance - stake_fee:
                bittensor.__console__.print(":cross_mark: [red]Not enough balance[/red]: [green]{}[/green] to stake: [blue]{}[/blue] from coldkey: [white]{}[/white]".format(old_balance, staking_balance, wallet.name))
                continue
                            
            # Ask before moving on.
            if prompt:
                if not Confirm.ask("Do you want to stake:\n[bold white]  amount: {}\n  hotkey: {}\n  fee: {}[/bold white ]?".format( staking_balance, wallet.hotkey_str, stake_fee) ):
                    continue

            try:
                staking_response: bool = self.__do_add_stake_single(
                    wallet = wallet,
                    hotkey_ss58 = hotkey_ss58,
                    amount = staking_balance,
                    wait_for_inclusion = wait_for_inclusion,
                    wait_for_finalization = wait_for_finalization,
                )

                if staking_response: # If we successfully staked.
                    # We only wait here if we expect finalization.
                    if not wait_for_finalization and not wait_for_inclusion:
                        bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                        old_balance -= staking_balance + stake_fee
                        successful_stakes += 1
                        if staking_all:
                            # If staked all, no need to continue
                            break

                        continue

                    bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")

                    block = self.get_current_block()
                    new_stake = self.get_stake_for_coldkey_and_hotkey( wallet.coldkey.ss58_address, hotkey_ss58, block = block )
                    new_balance = self.get_balance( wallet.coldkeypub.ss58_address, block = block )
                    bittensor.__console__.print("Stake ({}): [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( wallet.hotkey.ss58_address, old_stake, new_stake ))
                    old_balance = new_balance
                    successful_stakes += 1
                    if staking_all:
                        # If staked all, no need to continue
                        break

                else:
                    bittensor.__console__.print(":cross_mark: [red]Failed[/red]: Error unknown.")
                    continue

            except NotRegisteredError as e:
                bittensor.__console__.print(":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(wallet.hotkey_str))
                continue
            except StakeError as e:
                bittensor.__console__.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
                continue
                
        
        if successful_stakes != 0:
            with bittensor.__console__.status(":satellite: Checking Balance on: ([white]{}[/white] ...".format(self.network)):
                new_balance = self.get_balance( wallet.coldkeypub.ss58_address )
            bittensor.__console__.print("Balance: [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_balance, new_balance ))
            return True

        return False

    def transfer(
            self, 
            wallet: 'bittensor.wallet',
            dest: str, 
            amount: Union[Balance, float], 
            wait_for_inclusion: bool = True,
            wait_for_finalization: bool = False,
            prompt: bool = False,
        ) -> bool:
        r""" Transfers funds from this wallet to the destination public key address
        Args:
            wallet (bittensor.wallet):
                Bittensor wallet object to make transfer from.
            dest (str, ss58_address or ed25519):
                Destination public key address of reciever. 
            amount (Union[Balance, int]):
                Amount to stake as bittensor balance, or float interpreted as Tao.
            wait_for_inclusion (bool):
                If set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                If set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
        Returns:
            success (bool):
                Flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """
        # Validate destination address.
        if not is_valid_bittensor_address_or_public_key( dest ):
            bittensor.__console__.print(":cross_mark: [red]Invalid destination address[/red]:[bold white]\n  {}[/bold white]".format(dest))
            return False

        if isinstance( dest, bytes):
            # Convert bytes to hex string.
            dest = "0x" + dest.hex()

        # Unlock wallet coldkey.
        wallet.coldkey

        # Convert to bittensor.Balance
        if not isinstance(amount, bittensor.Balance ):
            transfer_balance = bittensor.Balance.from_tao( amount )
        else:
            transfer_balance = amount

        # Check balance.
        with bittensor.__console__.status(":satellite: Checking Balance..."):
            account_balance = self.get_balance( wallet.coldkey.ss58_address )

        # Estimate transfer fee.
        with bittensor.__console__.status(":satellite: Estimating Transfer Fees..."):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='Balances',
                    call_function='transfer',
                    call_params={
                        'dest': dest, 
                        'value': transfer_balance.rao
                    }
                )
                payment_info = substrate.get_payment_info(call = call, keypair = wallet.coldkey)
                transfer_fee = "N/A"
                if payment_info:
                    transfer_fee = bittensor.Balance.from_rao(payment_info['partialFee'])
                    bittensor.__console__.print("[green]Estimated Fee: {}[/green]".format( transfer_fee ))
                else:
                    bittensor.__console__.print(":cross_mark: [red]Failed[/red]: could not estimate transfer fee, assuming base fee of 0.2")
                    transfer_fee = bittensor.Balance.from_tao( 0.2 )

        if account_balance < transfer_balance + transfer_fee:
            bittensor.__console__.print(":cross_mark: [red]Not enough balance[/red]:[bold white]\n  balance: {}\n  amount: {} fee: {}[/bold white]".format( account_balance, transfer_balance, transfer_fee ))
            return False

        # Ask before moving on.
        if prompt:
            if not Confirm.ask("Do you want to transfer:[bold white]\n  amount: {}\n  from: {}:{}\n  to: {}\n  for fee: {}[/bold white]".format( transfer_balance, wallet.name, wallet.coldkey.ss58_address, dest, transfer_fee )):
                return False

        with bittensor.__console__.status(":satellite: Transferring..."):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='Balances',
                    call_function='transfer',
                    call_params={
                        'dest': dest, 
                        'value': transfer_balance.rao
                    }
                )
                extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
                response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                    return True

                # Otherwise continue with finalization.
                response.process_events()
                if response.is_success:
                    bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")
                    block_hash = response.block_hash
                    bittensor.__console__.print("[green]Block Hash: {}[/green]".format( block_hash ))
                    explorer_url = "https://explorer.nakamoto.opentensor.ai/#/explorer/query/{block_hash}".format( block_hash = block_hash )
                    bittensor.__console__.print("[green]Explorer Link: {}[/green]".format( explorer_url ))
                else:
                    bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))

        if response.is_success:
            with bittensor.__console__.status(":satellite: Checking Balance..."):
                new_balance = self.get_balance( wallet.coldkey.ss58_address )
                bittensor.__console__.print("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(account_balance, new_balance))
                return True
        
        return False

    def __do_remove_stake_single(
        self, 
        wallet: 'bittensor.wallet',
        hotkey_ss58: str,
        amount: 'bittensor.Balance', 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        r"""
        Executes an unstake call to the chain using the wallet and amount specified.
        Args:
            wallet (bittensor.wallet):
                Bittensor wallet object.
            hotkey_ss58 (str):
                Hotkey address to unstake from.
            amount (bittensor.Balance):
                Amount to unstake as bittensor balance object.
            wait_for_inclusion (bool):
                If set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                If set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        Raises:
            StakeError:
                If the extrinsic fails to be finalized or included in the block.
            NotRegisteredError:
                If the hotkey is not registered in any subnets.

        """
        # Decrypt keys,
        wallet.coldkey

        if not self.is_hotkey_registered( hotkey_ss58 ):
            # Hotkey is not registered in any subnets.
            raise NotRegisteredError("Hotkey: {} is not registered.".format(hotkey_ss58))

    
        with self.substrate as substrate:
            call = substrate.compose_call(
            call_module='Paratensor', 
            call_function='remove_stake',
            call_params={
                'hotkey': hotkey_ss58,
                'amount_unstaked': amount.rao
                }
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            response.process_events()
            if response.is_success:
                return True
            else:
                raise StakeError(response.error_message)

    def unstake (
            self, 
            wallet: 'bittensor.wallet',
            hotkey_ss58: Optional[str] = None,
            amount: Union[Balance, float] = None, 
            wait_for_inclusion:bool = True, 
            wait_for_finalization:bool = False,
            prompt: bool = False,
        ) -> bool:
        r""" Removes stake into the wallet coldkey from the specified hotkey uid.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            hotkey_ss58 (Optional[str]):
                ss58 address of the hotkey to unstake from.
                by default, the wallet hotkey is used.
            amount (Union[Balance, float]):
                Amount to stake as bittensor balance, or float interpreted as tao.
            wait_for_inclusion (bool):
                if set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                if set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """
        # Decrypt keys,
        wallet.coldkey

        own_hotkey: bool = False # Flag to indicate if we are using the wallet's own hotkey.
        if hotkey_ss58 is None:
            hotkey_ss58 = wallet.hotkey.ss58_address # Default to wallet's own hotkey.
            own_hotkey = True

        with bittensor.__console__.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(self.network)):
            old_balance = self.get_balance( wallet.coldkey.ss58_address )
            if not self.is_hotkey_registered( hotkey_ss58 ): # Hotkey is not registered on the chain.
                raise NotRegisteredError("Hotkey: {} is not registered.".format(hotkey_ss58))
            
            old_stake = self.get_stake_for_coldkey_and_hotkey( wallet.coldkey.ss58_address, hotkey_ss58 = hotkey_ss58 )

        # Convert to bittensor.Balance
        if amount == None:
            # Unstake it all.
            unstaking_balance = old_stake
        elif not isinstance(amount, bittensor.Balance ):
            unstaking_balance = bittensor.Balance.from_tao( amount )
        else:
            unstaking_balance = amount

        # Check enough to unstake.
        stake_on_uid = old_stake
        if unstaking_balance > stake_on_uid:
            bittensor.__console__.print(":cross_mark: [red]Not enough stake[/red]: [green]{}[/green] to unstake: [blue]{}[/blue] from hotkey: [white]{}[/white]".format(stake_on_uid, unstaking_balance, wallet.hotkey_str))
            return False

        # Estimate unstaking fee.
        unstake_fee = None # To be filled.
        with bittensor.__console__.status(":satellite: Estimating Staking Fees..."):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='Paratensor', 
                    call_function='remove_stake',
                    call_params={
                        'hotkey': hotkey_ss58,
                        'amount_unstaked': unstaking_balance.rao
                    }
                )
                payment_info = substrate.get_payment_info(call = call, keypair = wallet.coldkey)
                if payment_info:
                    unstake_fee = bittensor.Balance.from_rao(payment_info['partialFee'])
                    bittensor.__console__.print("[green]Estimated Fee: {}[/green]".format( unstake_fee ))
                else:
                    unstake_fee = bittensor.Balance.from_tao( 0.2 )
                    bittensor.__console__.print(":cross_mark: [red]Failed[/red]: could not estimate staking fee, assuming base fee of 0.2")
                        
        # Ask before moving on.
        if prompt:
            if not Confirm.ask("Do you want to unstake:\n[bold white]  amount: {}\n  hotkey: {}\n  fee: {}[/bold white ]?".format( unstaking_balance, wallet.hotkey_str, unstake_fee) ):
                return False

        
        try:
            with bittensor.__console__.status(":satellite: Unstaking from chain: [white]{}[/white] ...".format(self.network)):
                staking_response: bool = self.__do_remove_stake_single(
                    wallet = wallet,
                    hotkey_ss58 = hotkey_ss58,
                    amount = unstaking_balance,
                    wait_for_inclusion = wait_for_inclusion,
                    wait_for_finalization = wait_for_finalization,
                )

            if staking_response: # If we successfully unstaked.
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                    return True

                bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")
                with bittensor.__console__.status(":satellite: Checking Balance on: [white]{}[/white] ...".format(self.network)):
                    new_balance = self.get_balance( address = wallet.coldkey.ss58_address )
                    new_stake = self.get_stake_for_coldkey_and_hotkey( wallet.coldkey.ss58_address, hotkey_ss58 = hotkey_ss58 ) # Get stake on hotkey.
                    bittensor.__console__.print("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_balance, new_balance ))
                    bittensor.__console__.print("Stake:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_stake, new_stake ))
                    return True
            else:
                bittensor.__console__.print(":cross_mark: [red]Failed[/red]: Error unknown.")
                return False

        except NotRegisteredError as e:
            bittensor.__console__.print(":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(wallet.hotkey_str))
            return False
        except StakeError as e:
            bittensor.__console__.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
            return False
            

    def unstake_multiple (
            self, 
            wallet: 'bittensor.wallet',
            hotkey_ss58s: List[str],
            amounts: List[Union[Balance, float]] = None, 
            wait_for_inclusion: bool = True, 
            wait_for_finalization: bool = False,
            prompt: bool = False,
        ) -> bool:
        r""" Removes stake from each hotkey_ss58 in the list, using each amount, to a common coldkey.
        Args:
            wallet (bittensor.wallet):
                The wallet with the coldkey to unstake to.
            hotkey_ss58s (List[str]):
                List of hotkeys to unstake from.
            amounts (List[Union[Balance, float]]):
                List of amounts to unstake. If None, unstake all.
            wait_for_inclusion (bool):
                if set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                if set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or included in the block.
                flag is true if any wallet was unstaked.
                If we did not wait for finalization / inclusion, the response is true.
        """
        if not isinstance(hotkey_ss58s, list):
            raise TypeError("hotkey_ss58s must be a list of str")
        
        if len(hotkey_ss58s) == 0:
            return True

        if amounts is not None and len(amounts) != len(hotkey_ss58s):
            raise ValueError("amounts must be a list of the same length as hotkey_ss58s")

        if amounts is not None and not all(isinstance(amount, (Balance, float)) for amount in amounts):
            raise TypeError("amounts must be a [list of bittensor.Balance or float] or None")

        if amounts is None:
            amounts = [None] * len(hotkey_ss58s)
        else:
            # Convert to Balance
            amounts = [bittensor.Balance.from_tao(amount) if isinstance(amount, float) else amount for amount in amounts ]

            if sum(amount.tao for amount in amounts) == 0:
                # Staking 0 tao
                return True

        # Unlock coldkey.
        wallet.coldkey

        old_stakes = []
        with bittensor.__console__.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(self.network)):
            old_balance = self.get_balance( wallet.coldkey.ss58_address )

            for hotkey_ss58 in hotkey_ss58s:
                old_stake = self.get_stake_for_coldkey_and_hotkey( wallet.coldkey.ss58_address, hotkey_ss58 = hotkey_ss58 ) # Get stake on hotkey.
                old_stakes.append(old_stake) # None if not registered.

        successful_unstakes = 0
        for hotkey_ss58, amount, old_stake in zip(hotkey_ss58s, amounts, old_stakes):
            # Covert to bittensor.Balance
            if amount == None:
                # Unstake it all.
                unstaking_balance = old_stake
            elif not isinstance(amount, bittensor.Balance ):
                unstaking_balance = bittensor.Balance.from_tao( amount )
            else:
                unstaking_balance = amount

            # Check enough to unstake.
            stake_on_uid = old_stake
            if unstaking_balance > stake_on_uid:
                bittensor.__console__.print(":cross_mark: [red]Not enough stake[/red]: [green]{}[/green] to unstake: [blue]{}[/blue] from hotkey: [white]{}[/white]".format(stake_on_uid, unstaking_balance, wallet.hotkey_str))
                continue

            # Estimate unstaking fee.
            unstake_fee = None # To be filled.
            with bittensor.__console__.status(":satellite: Estimating Staking Fees..."):
                with self.substrate as substrate:
                    call = substrate.compose_call(
                        call_module='Paratensor', 
                        call_function='remove_stake',
                        call_params={
                            'hotkey': hotkey_ss58,
                            'amount_unstaked': unstaking_balance.rao
                        }
                    )
                    payment_info = substrate.get_payment_info(call = call, keypair = wallet.coldkey)
                    if payment_info:
                        unstake_fee = bittensor.Balance.from_rao(payment_info['partialFee'])
                        bittensor.__console__.print("[green]Estimated Fee: {}[/green]".format( unstake_fee ))
                    else:
                        unstake_fee = bittensor.Balance.from_tao( 0.2 )
                        bittensor.__console__.print(":cross_mark: [red]Failed[/red]: could not estimate staking fee, assuming base fee of 0.2")
                            
            # Ask before moving on.
            if prompt:
                if not Confirm.ask("Do you want to unstake:\n[bold white]  amount: {}\n  hotkey: {}\n  fee: {}[/bold white ]?".format( unstaking_balance, wallet.hotkey_str, unstake_fee) ):
                    continue
            
            try:
                with bittensor.__console__.status(":satellite: Unstaking from chain: [white]{}[/white] ...".format(self.network)):
                    staking_response: bool = self.__do_remove_stake_single(
                        wallet = wallet,
                        hotkey_ss58 = hotkey_ss58,
                        amount = unstaking_balance,
                        wait_for_inclusion = wait_for_inclusion,
                        wait_for_finalization = wait_for_finalization,
                    )

                if staking_response: # If we successfully unstaked.
                    # We only wait here if we expect finalization.
                    if not wait_for_finalization and not wait_for_inclusion:
                        bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                        successful_unstakes += 1
                        continue

                    bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")
                    with bittensor.__console__.status(":satellite: Checking Balance on: [white]{}[/white] ...".format(self.network)):
                        block = self.get_current_block()
                        new_stake = self.get_stake_for_coldkey_and_hotkey( wallet.coldkey.ss58_address, hotkey_ss58 = hotkey_ss58, block = block )
                        bittensor.__console__.print("Stake ({}): [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( wallet.hotkey.ss58_address, stake_on_uid, new_stake ))
                        successful_unstakes += 1
                else:
                    bittensor.__console__.print(":cross_mark: [red]Failed[/red]: Error unknown.")
                    continue

            except NotRegisteredError as e:
                bittensor.__console__.print(":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(wallet.hotkey_str))
                continue
            except StakeError as e:
                bittensor.__console__.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
                continue
                
        
        if successful_unstakes != 0:
            with bittensor.__console__.status(":satellite: Checking Balance on: ([white]{}[/white] ...".format(self.network)):
                new_balance = self.get_balance( wallet.coldkey.ss58_address )
            bittensor.__console__.print("Balance: [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_balance, new_balance ))
            return True

        return False
                
    def set_weights(
            self, 
            wallet: 'bittensor.wallet',
            netuid: int,
            uids: Union[torch.LongTensor, list],
            weights: Union[torch.FloatTensor, list],
            wait_for_inclusion:bool = False,
            wait_for_finalization:bool = False,
            prompt:bool = False
        ) -> bool:
        r""" Sets the given weights and values on chain for wallet hotkey account.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            netuid (int):
                netuid of the subent to set weights for.
            uids (Union[torch.LongTensor, list]):
                uint64 uids of destination neurons.
            weights ( Union[torch.FloatTensor, list]):
                weights to set which must floats and correspond to the passed uids.
            wait_for_inclusion (bool):
                if set, waits for the extrinsic to enter a block before returning true,
                or returns false if the extrinsic fails to enter the block within the timeout.
            wait_for_finalization (bool):
                if set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block.
                If we did not wait for finalization / inclusion, the response is true.
        """
        # First convert types.
        if isinstance( uids, list ):
            uids = torch.tensor( uids, dtype = torch.int64 )
        if isinstance( weights, list ):
            weights = torch.tensor( weights, dtype = torch.float32 )

        # Reformat and normalize.
        weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit( uids, weights )

        # Ask before moving on.
        if prompt:
            if not Confirm.ask("Do you want to set weights:\n[bold white]  weights: {}\n  uids: {}[/bold white ]?".format( [float(v/4294967295) for v in weight_vals], weight_uids) ):
                return False

        with bittensor.__console__.status(":satellite: Setting weights on [white]{}[/white] ...".format(self.network)):
            try:
                with self.substrate as substrate:
                    call = substrate.compose_call(
                        call_module='Paratensor',
                        call_function='set_weights',
                        call_params = {
                            'dests': weight_uids,
                            'weights': weight_vals,
                            'netuid': netuid,
                        }
                    )
                    extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.hotkey )
                    response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                    # We only wait here if we expect finalization.
                    if not wait_for_finalization and not wait_for_inclusion:
                        bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                        return True

                    response.process_events()
                    if response.is_success:
                        bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")
                        bittensor.logging.success(  prefix = 'Set weights', sufix = '<green>Finalized: </green>' + str(response.is_success) )
                    else:
                        bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))
                        bittensor.logging.warning(  prefix = 'Set weights', sufix = '<red>Failed: </red>' + str(response.error_message) )

            except Exception as e:
                bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(e))
                bittensor.logging.warning(  prefix = 'Set weights', sufix = '<red>Failed: </red>' + str(e) )
                return False

        if response.is_success:
            bittensor.__console__.print("Set weights:\n[bold white]  weights: {}\n  uids: {}[/bold white ]".format( [float(v/4294967295) for v in weight_vals], weight_uids ))
            message = '<green>Success: </green>' + f'Set {len(uids)} weights, top 5 weights' + str(list(zip(uids.tolist()[:5], [round (w,4) for w in weights.tolist()[:5]] )))
            logger.debug('Set weights:'.ljust(20) +  message)
            return True
        
        return False

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

    def neurons(self, netuid: int, block: Optional[int] = None ) -> List[NeuronMetadata]: 
        r""" Returns a list of neuron from the chain. 
        Args:
            netuid ( int ):
                The netuid of the subnet to pull neurons from.
            block ( Optional[int] ):
                block to sync from.
        Returns:
            neuron (List[NeuronMetadata]):
                List of neuron metadata objects.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [netuid]
                if block_hash:
                    params = [block_hash] + params
                return substrate.rpc_request(
                    method="neuronInfo_getNeurons", # custom rpc method
                    params=params
                )
        
        json_body = make_substrate_call_with_retry()
        result = json_body['result']
        
        return [ NeuronMetadata.from_json( neuron ) for neuron in result ]


    @staticmethod
    def _null_neuron() -> NeuronMetadata:
        neuron = NeuronMetadata(
            active = 0,
            stake = 0,
            rank = 0,
            trust = 0,
            consensus = 0,
            incentive = 0,
            dividends = 0,
            emission = 0,
            weights = [],
            bonds = [],
            version = 0,
            modality = 0,
            uid = 0,
            netuid = 0,
            port = 0,
            priority = 0,
            ip_type = 0,
            last_update = 0,
            ip = 0,
            is_null = True,
            coldkey = "000000000000000000000000000000000000000000000000",
            hotkey = "000000000000000000000000000000000000000000000000"
        )
        return neuron

    @staticmethod
    def _neuron_dict_to_namespace(neuron_dict) -> NeuronMetadata:
        if neuron_dict['hotkey'] == '5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM':
            return Subtensor._null_neuron()
        else:
            neuron = NeuronMetadata( **neuron_dict )
            neuron.stake = neuron.stake / RAOPERTAO
            neuron.rank = neuron.rank / U64_MAX
            neuron.trust = neuron.trust / U64_MAX
            neuron.consensus = neuron.consensus / U64_MAX
            neuron.incentive = neuron.incentive / U64_MAX
            neuron.dividends = neuron.dividends / U64_MAX
            neuron.emission = neuron.emission / RAOPERTAO
                
            return neuron

    def neuron_for_uid( self, uid: int, netuid: int, block: Optional[int] = None ) -> Optional[NeuronMetadata]: 
        r""" Returns a list of neuron from the chain. 
        Args:
            uid ( int ):
                The uid of the neuron to query for.
            netuid ( int ):
                The uid of the network to query for.
            block ( int ):
                The neuron at a particular block
        Returns:
            neuron (Optional[NeuronMetadata]):
                neuron metadata associated with uid or None if it does not exist.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [netuid, uid]
                if block_hash:
                    params = [block_hash] + params
                return substrate.rpc_request(
                    method="neuronInfo_getNeuron", # custom rpc method
                    params=params
                )
        
        json_body = make_substrate_call_with_retry()
        if json_body['result'] == None:
            return None

        result = json_body['result']
        return NeuronMetadata.from_json( result ) 

    def get_delegate_take( self, hotkey_ss58: str ) -> Optional[float]:
        r""" Returns the take for a delegate.
        Args:
            hotkey_ss58 ( str ):
                The hotkey to check.
        Returns:
            take (Optional[float]):
                The take for the delegate or None if the hotkey is not a delegate.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function='Delegates',
                    params = [ hotkey_ss58 ]
                ).value

        result = make_substrate_call_with_retry()
        if result:
            return result.value / U16_MAX
        else:
            return None

    def get_delegates( self, block: Optional[int] = None ) -> List[DelegateInfo]:
        r""" Returns a list of all delegates on the network.
        Args:
            block ( Optional[int] ):
                block to sync at.
        Returns:
            delegates (List[DelegateInfo]):
                List of delegate info objects.
                
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query_map(
                    module='Paratensor',
                    storage_function='Delegates',
                    block_hash=None if block == None else substrate.get_block_hash( block )
                )

        delegates = []
        result = make_substrate_call_with_retry()
        if result:
            for delegate_ss58, take in result.value:
                total_stake = self.get_total_stake_for_hotkey( delegate_ss58, block )
                nominators = self.get_nominators_for_hotkey( delegate_ss58, block )
                owner = self.get_owner_for_hotkey( delegate_ss58, block )
                info = DelegateInfo(
                    hotkey_ss58=delegate_ss58,
                    take = take / U16_MAX,
                    total_stake=total_stake,
                    nominators=len(nominators),
                    owner_ss58=owner
                )
                delegates.append( info )
            return delegates
        else:
            return []

    def get_nominators_for_hotkey( self, hotkey_ss58: str, block: Optional[int] = None ) -> List[Tuple[str, Balance]]:
        r""" Returns the nominators for a delegate hotkey.
        Args:
            hotkey_ss58 ( str ):
                The hotkey to check.
            block ( Optional[int] ):
                block to sync at.
        Returns:
            nominators (List[Tuple[str, Balance]]):
                List of nominator ss58 and stake.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query_map(
                    module='Paratensor',
                    storage_function='Stake',
                    params = [ hotkey_ss58 ],
                    block_hash=None if block == None else substrate.get_block_hash( block )
                )

        result = make_substrate_call_with_retry()
        if result:
            nominators = [(nominator_ss58, stake) for nominator_ss58, stake in result.value]
            return nominators

        else:
            return 0

    def is_hotkey_delegate( self, hotkey_ss58: str ) -> bool:
        r""" Returns true if the hotkey is a delegate.
        Args:
            hotkey_ss58 ( str ):
                The hotkey to check.
        Returns:
            is_delegate (bool):
                True if the hotkey is a delegate.
        """
        return self.get_delegate_take( hotkey_ss58 ) is not None

    def become_delegate( self, wallet: 'bittensor.Wallet', wait_for_finalization: bool = False, wait_for_inclusion: bool = True ) -> bool:
        r""" Becomes a delegate for the hotkey.
        Args:
            wallet ( bittensor.Wallet ):
                The wallet to become a delegate for.
        Returns:
            success (bool):
                True if the transaction was successful.
        """
        # Unlock the coldkey.
        wallet.coldkey
        wallet.hotkey

        # Check if the hotkey is already a delegate.
        if self.is_hotkey_delegate( wallet.hotkey.ss58_address ):
            logger.error('Hotkey {} is already a delegate.'.format(wallet.hotkey.ss58_address))
            return False

        with bittensor.__console__.status(":satellite: Sending become delegate call on [white]{}[/white] ...".format(self.network)):
            try:
                with self.substrate as substrate:
                    call = substrate.compose_call(
                        call_module='Paratensor',
                        call_function='become_delegate',
                        call_params = {
                            'hotkey': wallet.hotkey.ss58_address,
                            'take': int(0.5 * U16_MAX), # TODO: remove take from become delegate.
                        }
                    )
                    extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey ) # sign with coldkey
                    response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                    # We only wait here if we expect finalization.
                    if not wait_for_finalization and not wait_for_inclusion:
                        bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                        return True

                    response.process_events()
                    if response.is_success:
                        bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")
                        bittensor.logging.success(  prefix = 'Become Delegate', sufix = '<green>Finalized: </green>' + str(response.is_success) )
                    else:
                        bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))
                        bittensor.logging.warning(  prefix = 'Set weights', sufix = '<red>Failed: </red>' + str(response.error_message) )

            except Exception as e:
                bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(e))
                bittensor.logging.warning(  prefix = 'Set weights', sufix = '<red>Failed: </red>' + str(e) )
                return False

        if response.is_success:
            return True
        
        return False
        

    def get_netuids_for_hotkey( self, ss58_hotkey: str, block: Optional[int] = None) -> List[int]:
        r""" Returns a list of netuids for the subnets this hotkey is registered on.
        Args:
            ss58_hotkey ( str ):
                The hotkey to query for a neuron.
            block ( Optional[int] ):
                The block to query for.
        Returns:
            netuids (List[int]):
                The list of netuids this hotkey is registered on.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query (
                    module='Paratensor',
                    storage_function='Subnets',
                    params = [ ss58_hotkey ],
                    block_hash = None if block == None else substrate.get_block_hash( block )
                )

        result = make_substrate_call_with_retry()
        return result.value


    def get_uid_for_hotkey( self, ss58_hotkey: str, netuid: Optional[int] = None, block: Optional[int] = None) -> Union[Dict[str, int], int]:
        r""" Returns true if the passed hotkey is registered on the chain.
        Args:
            ss58_hotkey ( str ):
                The hotkey to query for a neuron.
            netuid ( Optional[int] ):
                The netuid of the network to query for, default is None.
            block ( Optional[int] ):
                The block to query for.
        Returns:
            uid (int):
                The uid of the neuron associated with the hotkey or -1 if it does not exist.
            uids (Dict[str(int), int]):
                The uids of the neurons associated with the hotkey, as a dict of netuid to uid.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry(netuid: int):
            with self.substrate as substrate:
                return substrate.query (
                    module='Paratensor',
                    storage_function='Uids',
                    params = [ netuid, ss58_hotkey ],
                    block_hash = None if block == None else substrate.get_block_hash( block )
                )
        
        # Get netuids for hotkey.
        netuids = []
        #netuids = self.get_netuids_for_hotkey( ss58_hotkey = ss58_hotkey, block = block)

        if False and netuids == []:
            return -1

        if netuid:
            if False and netuid not in netuids:
                return -1
            else:
                result = make_substrate_call_with_retry( netuid )
                val = result.value
                if val:
                    return val
                else:
                    return -1
        
        # No netuid passed, return all uids.
        uids = { str(netuid): -1 for netuid in netuids }
        # Get uids for hotkey.
        for netuid in netuids:
            result = make_substrate_call_with_retry(netuid)
            if result != None:
                uids[str(netuid)] = result.value
            else:
                # Shouldn't happen.
                del uids[str(netuid)] 

        return uids

    def is_hotkey_registered( self, ss58_hotkey: str, netuid: Optional[int] = None, block: Optional[int] = None) -> bool:
        r""" Returns true if the passed hotkey is registered on the chain.
        Args:
            ss58_hotkey ( str ):
                The hotkey to query for a neuron.
            netuid ( Optional[int] ):
                The network uid to query for a neuron.
                None will return true if the hotkey is registered on any network.
            block (Optional[int]):
                The block to query.
            
        Returns:
            is_registered ( bool ):
                True if the passed hotkey is registered on the chain.
        """
        uid_or_uid_map: Union[Dict[str, int], int] = self.get_uid_for_hotkey( ss58_hotkey = ss58_hotkey, netuid = netuid, block = block)
        # will return -1 if not registered. 
        # will return a dict of netuid to uid if registered on multiple networks.

        if isinstance(uid_or_uid_map, dict):
            # registered on multiple networks.
            return True
        else:
            assert isinstance(uid_or_uid_map, int)
            return uid_or_uid_map != -1 # -1 means not registered.

    def neuron_for_pubkey( self, ss58_hotkey: str, netuid: Optional[int] = None, block: Optional[int] = None ) -> Optional[Union[NeuronMetadata, List[NeuronMetadata]]]: 
        r""" Returns a list of neuron from the chain. 
        Args:
            ss58_hotkey ( str ):
                The hotkey to query for a neuron.
            netuid ( Optional[int] ):
                The network uid to query for a neuron, default is None.
                None will return the neuron for the hotkey on each subnet it is registered on.
            block (Optional[int]):
                The block to query.

        Returns:
            neuron ( Optional[NeuronMetadata] ):
                neuron metadata associated with uid or None if it does not exist.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry(netuid: int):
            with self.substrate as substrate:
                return substrate.query (
                    module='Paratensor',
                    storage_function='Uids',
                    params = [ netuid, ss58_hotkey ],
                    block_hash = None if block == None else substrate.get_block_hash( block )
                )

        if False and netuid:
            result = make_substrate_call_with_retry( netuid )
            if not result.is_some:
                return Subtensor._null_neuron()
            uid = int(result.value)
            neuron = self.neuron_for_uid( uid, netuid = netuid, block = block )
            return neuron
        else:
            # multiple networks. netuid is null
            netuid_map_or_result = self.get_uid_for_hotkey( ss58_hotkey = ss58_hotkey, block = block )
            if netuid_map_or_result == -1:
                return Subtensor._null_neuron()
            else:
                netuid_map = netuid_map_or_result
            # TODO (Cameron): Use is_some when chain updated
                if netuid:
                    if str(netuid) in netuid_map:
                        return self.neuron_for_uid( netuid_map[str(netuid)], netuid = netuid, block = block )
                    else:
                        return Subtensor._null_neuron()
            # end TODO

                neurons = []
                for netuid_str, uid in netuid_map.items():
                    neuron = self.neuron_for_uid( uid, netuid = int(netuid_str), block = block )
                    neurons.append( neuron )
                return neurons

    def get_n( self, netuid: int, block: Optional[int] = None ) -> int: 
        r""" Returns the number of neurons on the chain at block.
        Args:
            netuid ( int ):
                The network uid to query.
            block ( Optional[int] ):
                The block number to get the neuron count from.
                Default is None, which returns the current block.
        Returns:
            n ( int ):
                the number of neurons subscribed to the chain.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return int(substrate.query(
                    module='Paratensor',
                    storage_function = 'N',
                    params = [ netuid ],
                    block_hash = None if block == None else substrate.get_block_hash( block )
                ).value)
        return make_substrate_call_with_retry()

    def neuron_for_wallet( self, wallet: 'bittensor.Wallet', netuid = int, block: Optional[int] = None ) -> Optional[NeuronMetadata]: 
        r""" Returns a list of neuron from the chain. 
        Args:
            wallet ( `bittensor.Wallet` ):
                Checks to ensure that the passed wallet is subscribed.
            netuid ( int ):
                The network uid of the subnet to query.
            block (Optional[int]):
                The block to query.
        Returns:
            neuron (Optional[NeuronMetadata] ):
                neuron metadata associated with uid or None if it does not exist.
        """
        return self.neuron_for_pubkey ( wallet.hotkey.ss58_address, netuid = netuid, block = block )


    def subnet_exists( self, netuid: int ) -> bool:
        r""" Returns true if the subnet exists.
        Args:
            netuid ( int ):
                The network uid of the subnet to query.
        Returns:
            exists ( bool ):
                True if the subnet exists.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='Paratensor',
                    storage_function = 'NetworksAdded',
                    params = [ netuid ],
                    block_hash = None
                ).value
        return make_substrate_call_with_retry()
