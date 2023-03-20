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
from rich.prompt import Confirm, Prompt
from typing import List, Dict, Union, Optional
from multiprocessing import Process

import bittensor
from tqdm import tqdm
import bittensor.utils.networking as net
import bittensor.utils.weight_utils as weight_utils
from retry import retry
from substrateinterface import SubstrateInterface
from bittensor.utils.balance import Balance
from bittensor.utils import is_valid_bittensor_address_or_public_key
from bittensor.utils.registratrion_old import create_pow
from types import SimpleNamespace

# Mocking imports
import os
import random
import scalecodec
import time
import subprocess
from sys import platform   

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
                print('''
                        Check that your internet connection is working and the chain endpoints are available: <blue>{}</blue>
                        The subtensor.network should likely be one of the following choices:
                            -- local - (your locally running node)
                            -- nobunaga - (staging)
                            -- nakamoto - (main)
                        Or you may set the endpoint manually using the --subtensor.chain_endpoint flag 
                        To run a local node (See: docs/running_a_validator.md) \n
                    '''.format( attempted_endpoints) )

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

    @property
    def rho (self) -> int:
        r""" Incentive mechanism rho parameter.
        Returns:
            rho (int):
                Incentive mechanism rho parameter.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubtensorModule', storage_function = 'Rho' ).value
        return make_substrate_call_with_retry()

    @property
    def kappa (self) -> int:
        r""" Incentive mechanism kappa parameter.
        Returns:
            kappa (int):
                Incentive mechanism kappa parameter.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubtensorModule', storage_function = 'Kappa' ).value
        return make_substrate_call_with_retry()

    @property
    def difficulty (self) -> int:
        r""" Returns registration difficulty from the chain.
        Returns:
            difficulty (int):
                Registration difficulty.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubtensorModule', storage_function = 'Difficulty' ).value
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
                return bittensor.Balance.from_rao( substrate.query(  module='SubtensorModule', storage_function = 'TotalIssuance').value )
        return make_substrate_call_with_retry()

    @property
    def immunity_period (self) -> int:
        r""" Returns the chain registration immunity_period
        Returns:
            immunity_period (int):
                Chain registration immunity_period
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubtensorModule', storage_function = 'ImmunityPeriod' ).value
        return make_substrate_call_with_retry()

    @property
    def validator_batch_size (self) -> int:
        r""" Returns the chain default validator batch size.
        Returns:
            batch_size (int):
                Chain default validator batch size.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubtensorModule', storage_function = 'ValidatorBatchSize' ).value
        return make_substrate_call_with_retry()

    @property
    def validator_sequence_length (self) -> int:
        r""" Returns the chain default validator sequence length.
        Returns:
            sequence_length (int):
                Chain default validator sequence length.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubtensorModule', storage_function = 'ValidatorSequenceLength' ).value
        return make_substrate_call_with_retry()

    @property
    def validator_epochs_per_reset (self) -> int:
        r""" Epochs passed before the validator resets its weights.
        Returns:
            validator_epochs_per_reset (int):
                Epochs passed before the validator resets its weights.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubtensorModule', storage_function = 'ValidatorEpochsPerReset' ).value
        return make_substrate_call_with_retry()

    @property
    def validator_epoch_length (self) -> int:
        r""" Default validator epoch length.
        Returns:
            validator_epoch_length (int):
                Default validator epoch length. 
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubtensorModule', storage_function = 'ValidatorEpochLen' ).value
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
                return bittensor.Balance.from_rao( substrate.query(  module='SubtensorModule', storage_function = 'TotalStake' ).value )
        return make_substrate_call_with_retry()

    @property
    def min_allowed_weights (self) -> int:
        r""" Returns min allowed number of weights.
        Returns:
            min_allowed_weights (int):
                Min number of weights allowed to be set.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubtensorModule', storage_function = 'MinAllowedWeights' ).value
        return make_substrate_call_with_retry()

    @property
    def max_weight_limit (self) -> int:
        r""" Returns MaxWeightLimit
        Returns:
            max_weight (int):
                the max value for weights after normalizaiton
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                U32_MAX = 4294967295
                return substrate.query( module='SubtensorModule', storage_function = 'MaxWeightLimit' ).value/U32_MAX
        return make_substrate_call_with_retry()

    @property
    def scaling_law_power (self) -> int:
        r""" Returns ScalingLawPower
        Returns:
            ScalingLawPower (float):
                the power term attached to scaling law
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                MAX = 100
                return substrate.query( module='SubtensorModule', storage_function = 'ScalingLawPower' ).value/MAX
        return make_substrate_call_with_retry()

    @property
    def synergy_scaling_law_power (self) -> int:
        r""" Returns SynergyScalingLawPower
        Returns:
            SynergyScalingLawPower (float):
                the term attached to synergy calculation during shapley scores
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                MAX = 100
                return substrate.query( module='SubtensorModule', storage_function = 'SynergyScalingLawPower' ).value/MAX
        return make_substrate_call_with_retry()

    @property
    def validator_exclude_quantile (self) -> int:
        r""" Returns ValidatorExcludeQuantile
        Returns:
            ValidatorExcludeQuantile (float):
                the quantile that validators should exclude when setting their weights
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                MAX = 100
                return substrate.query( module='SubtensorModule', storage_function = 'ValidatorExcludeQuantile' ).value/MAX
        return make_substrate_call_with_retry()

    @property
    def max_allowed_min_max_ratio(self) -> int:
        r""" Returns the chains max_allowed_min_max_ratio
        Returns:
            max_allowed_min_max_ratio (int):
                The max ratio allowed between the min and max.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubtensorModule', storage_function = 'MaxAllowedMaxMinRatio' ).value
        return make_substrate_call_with_retry()

    @property
    def n (self) -> int:
        r""" Returns total number of neurons on the chain.
        Returns:
            n (int):
                Total number of neurons on chain.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubtensorModule', storage_function = 'N' ).value
        return make_substrate_call_with_retry()

    @property
    def max_n (self) -> int:
        r""" Returns maximum number of neuron positions on the graph.
        Returns:
            max_n (int):
                Maximum number of neuron positions on the graph.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubtensorModule', storage_function = 'MaxAllowedUids' ).value
        return make_substrate_call_with_retry()

    @property
    def block (self) -> int:
        r""" Returns current chain block.
        Returns:
            block (int):
                Current chain block.
        """
        return self.get_current_block()

    @property
    def blocks_since_epoch (self) -> int:
        r""" Returns blocks since last epoch.
        Returns:
            blocks_since_epoch (int):
                blocks_since_epoch 
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubtensorModule', storage_function = 'BlocksSinceLastStep' ).value
        return make_substrate_call_with_retry()

    @property
    def blocks_per_epoch (self) -> int:
        r""" Returns blocks per chain epoch.
        Returns:
            blocks_per_epoch (int):
                blocks_per_epoch 
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubtensorModule', storage_function = 'BlocksPerStep' ).value
        return make_substrate_call_with_retry()

    def get_n (self, block: int = None) -> int:
        r""" Returns total number of neurons on the chain.
        Returns:
            n (int):
                Total number of neurons on chain.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(  
                    module='SubtensorModule', 
                    storage_function = 'N',
                    block_hash = None if block == None else substrate.get_block_hash( block )
                ).value
        return make_substrate_call_with_retry()

    @property
    def validator_prune_len (self) -> int:
        r""" Returns PruneLen 
        Returns:
            prune_len (int):
                the number of pruned tokens from each requests 
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubtensorModule', storage_function = 'ValidatorPruneLen' ).value
        return make_substrate_call_with_retry()

    @property
    def validator_logits_divergence (self) -> int:
        r""" Returns logits_divergence
        Returns:
            logits_divergence (int):
                the divergence value for logit distances, a measure for anomaly detection 
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                U64MAX = 18446744073709551615
                return substrate.query( module='SubtensorModule', storage_function = 'ValidatorLogitsDivergence' ).value/U64MAX
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
                modality = 0,
                wait_for_inclusion = wait_for_inclusion,
                wait_for_finalization = wait_for_finalization,
                prompt = prompt
        )
        return serve_success

    def register (
        self,
        wallet: 'bittensor.Wallet',
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
        netuid: int = None,
    ) -> bool:
        r""" Registers the wallet to chain.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
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

        with bittensor.__console__.status(":satellite: Checking Account..."):
             neuron = self.neuron_for_pubkey( wallet.hotkey.ss58_address )
             if not neuron.is_null:
                 bittensor.__console__.print(":white_heavy_check_mark: [green]Already Registered[/green]:\n  uid: [bold white]{}[/bold white]\n  hotkey: [bold white]{}[/bold white]\n  coldkey: [bold white]{}[/bold white]".format(neuron.uid, neuron.hotkey, neuron.coldkey))
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
                pow_result = create_pow( self, wallet, output_in_place, cuda, dev_id, TPB, num_processes=num_processes, update_interval=update_interval, log_verbose=log_verbose )
            else:
                pow_result = create_pow( self, wallet, output_in_place, num_processes=num_processes, update_interval=update_interval, log_verbose=log_verbose )

            # pow failed
            if not pow_result:
                # might be registered already
                if (wallet.is_registered( self )):
                    bittensor.__console__.print(":white_heavy_check_mark: [green]Registered[/green]")
                    return True
                
            # pow successful, proceed to submit pow to chain for registration
            else:
                with bittensor.__console__.status(":satellite: Submitting POW..."):
                    # check if pow result is still valid
                    while bittensor.utils.POWNotStale(self, pow_result):
                        with self.substrate as substrate:
                            # create extrinsic call
                            call = substrate.compose_call( 
                                call_module='SubtensorModule',  
                                call_function='register', 
                                call_params={ 
                                    'block_number': pow_result['block_number'], 
                                    'nonce': pow_result['nonce'], 
                                    'work': bittensor.utils.hex_bytes_to_u8_list( pow_result['work'] ), 
                                    'hotkey': wallet.hotkey.ss58_address, 
                                    'coldkey': wallet.coldkeypub.ss58_address
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
                                    bittensor.__console__.print(":white_heavy_check_mark: [green]Already Registered[/green]")
                                    return True

                                bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))
                                time.sleep(0.5)
                            
                            # Successful registration, final check for neuron and pubkey
                            else:
                                bittensor.__console__.print(":satellite: Checking Balance...")
                                neuron = self.neuron_for_pubkey( wallet.hotkey.ss58_address )
                                if not neuron.is_null:
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
            modality: int, 
            wait_for_inclusion: bool = False,
            wait_for_finalization = True,
            prompt: bool = False,
            netuid: int = None,
        ) -> bool:
        r""" Subscribes an bittensor endpoint to the substensor chain.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            ip (str):
                endpoint host port i.e. 192.122.31.4
            port (int):
                endpoint port number i.e. 9221
            modality (int):
                int encoded endpoint modality i.e 0 for TEXT
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
            'modality': modality,
            'coldkey': wallet.coldkeypub.ss58_address,
        }

        with bittensor.__console__.status(":satellite: Checking Axon..."):
            neuron = self.neuron_for_pubkey( wallet.hotkey.ss58_address )
            neuron_up_to_date = not neuron.is_null and params == {
                'version': neuron.version,
                'ip': neuron.ip,
                'port': neuron.port,
                'ip_type': neuron.ip_type,
                'modality': neuron.modality,
                'coldkey': neuron.coldkey
            }
            if neuron_up_to_date:
                bittensor.__console__.print(":white_heavy_check_mark: [green]Already Served[/green]\n  [bold white]ip: {}\n  port: {}\n  modality: {}\n  hotkey: {}\n  coldkey: {}[/bold white]".format(ip, port, modality, wallet.hotkey.ss58_address, wallet.coldkeypub.ss58_address))
                return True

        if prompt:
            if not Confirm.ask("Do you want to serve axon:\n  [bold white]ip: {}\n  port: {}\n  modality: {}\n  hotkey: {}\n  coldkey: {}[/bold white]".format(ip, port, modality, wallet.hotkey.ss58_address, wallet.coldkeypub.ss58_address)):
                return False
        
        with bittensor.__console__.status(":satellite: Serving axon on: [white]{}[/white] ...".format(self.network)):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='SubtensorModule',
                    call_function='serve_axon',
                    call_params=params
                )
                extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.hotkey)
                response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                if wait_for_inclusion or wait_for_finalization:
                    response.process_events()
                    if response.is_success:
                        bittensor.__console__.print(':white_heavy_check_mark: [green]Served[/green]\n  [bold white]ip: {}\n  port: {}\n  modality: {}\n  hotkey: {}\n  coldkey: {}[/bold white]'.format(ip, port, modality, wallet.hotkey.ss58_address, wallet.coldkeypub.ss58_address ))
                        return True
                    else:
                        bittensor.__console__.print(':cross_mark: [green]Failed to Subscribe[/green] error: {}'.format(response.error_message))
                        return False
                else:
                    return True

    def add_stake(
            self, 
            wallet: 'bittensor.wallet',
            amount: Union[Balance, float] = None, 
            wait_for_inclusion: bool = True,
            wait_for_finalization: bool = False,
            prompt: bool = False,
        ) -> bool:
        r""" Adds the specified amount of stake to passed hotkey uid.
        Args:
            wallet (bittensor.wallet):
                Bittensor wallet object.
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
        """
        # Decrypt keys,
        wallet.coldkey
        wallet.hotkey

        with bittensor.__console__.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(self.network)):
            old_balance = self.get_balance( wallet.coldkey.ss58_address )
            neuron = self.neuron_for_pubkey( ss58_hotkey = wallet.hotkey.ss58_address )
        if neuron.is_null:
            bittensor.__console__.print(":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(wallet.hotkey_str))
            return False

        # Covert to bittensor.Balance
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
                    call_module='SubtensorModule', 
                    call_function='add_stake',
                    call_params={
                        'hotkey': wallet.hotkey.ss58_address,
                        'ammount_staked': staking_balance.rao
                    }
                )
                payment_info = substrate.get_payment_info(call = call, keypair = wallet.coldkey)
                if payment_info:
                    staking_fee = bittensor.Balance.from_rao(payment_info['partialFee'])
                    bittensor.__console__.print("[green]Estimated Fee: {}[/green]".format( staking_fee ))
                else:
                    staking_fee = bittensor.Balance.from_tao( 0.2 )
                    bittensor.__console__.print(":cross_mark: [red]Failed[/red]: could not estimate staking fee, assuming base fee of 0.2")

        # Check enough to unstake.
        if staking_balance > old_balance + staking_fee:
            bittensor.__console__.print(":cross_mark: [red]Not enough stake[/red]:[bold white]\n  balance:{}\n  amount: {}\n  fee: {}\n  coldkey: {}[/bold white]".format(old_balance, staking_balance, staking_fee, wallet.name))
            return False
                
        # Ask before moving on.
        if prompt:
            if not Confirm.ask("Do you want to stake:[bold white]\n  amount: {}\n  to: {}\n  fee: {}[/bold white]".format( staking_balance, wallet.hotkey_str, staking_fee) ):
                return False

        with bittensor.__console__.status(":satellite: Staking to: [bold white]{}[/bold white] ...".format(self.network)):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='SubtensorModule', 
                    call_function='add_stake',
                    call_params={
                        'hotkey': wallet.hotkey.ss58_address,
                        'ammount_staked': staking_balance.rao
                    }
                )
                extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
                response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                    return True

                if response.is_success:
                    bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")
                else:
                    bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))

        if response.is_success:
            with bittensor.__console__.status(":satellite: Checking Balance on: [white]{}[/white] ...".format(self.network)):
                new_balance = self.get_balance( wallet.coldkey.ss58_address )
                old_stake = bittensor.Balance.from_tao( neuron.stake )
                new_stake = bittensor.Balance.from_tao( self.neuron_for_pubkey( ss58_hotkey = wallet.hotkey.ss58_address ).stake)
                bittensor.__console__.print("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_balance, new_balance ))
                bittensor.__console__.print("Stake:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_stake, new_stake ))
                return True
        
        return False

    def add_stake_multiple (
            self, 
            wallets: List['bittensor.wallet'],
            amounts: List[Union[Balance, float]] = None, 
            wait_for_inclusion: bool = True,
            wait_for_finalization: bool = False,
            prompt: bool = False,
        ) -> bool:
        r""" Adds stake to each wallet hotkey in the list, using each amount, from the common coldkey.
        Args:
            wallets (List[bittensor.wallet]):
                List of wallets to stake.
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
        if not isinstance(wallets, list):
            raise TypeError("wallets must be a list of bittensor.wallet")
        
        if len(wallets) == 0:
            return True
            

        if amounts is not None and len(amounts) != len(wallets):
            raise ValueError("amounts must be a list of the same length as wallets")

        if amounts is not None and not all(isinstance(amount, (Balance, float)) for amount in amounts):
            raise TypeError("amounts must be a [list of bittensor.Balance or float] or None")

        if amounts is None:
            amounts = [None] * len(wallets)
        else:
            # Convert to Balance
            amounts = [bittensor.Balance.from_tao(amount) if isinstance(amount, float) else amount for amount in amounts ]

            if sum(amount.tao for amount in amounts) == 0:
                # Staking 0 tao
                return True

        wallet_0: 'bittensor.wallet' = wallets[0]
        # Decrypt coldkey for all wallet(s) to use
        wallet_0.coldkey

        neurons = []
        with bittensor.__console__.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(self.network)):
            old_balance = self.get_balance( wallet_0.coldkey.ss58_address )

            for wallet in wallets:
                neuron = self.neuron_for_pubkey( ss58_hotkey = wallet.hotkey.ss58_address )

                if neuron.is_null:
                    neurons.append( None )
                    continue

                neurons.append( neuron )

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
        for wallet, amount, neuron in zip(wallets, amounts, neurons):
            if neuron is None:
                bittensor.__console__.print(":cross_mark: [red]Hotkey: {} is not registered. Skipping ...[/red]".format( wallet.hotkey_str ))
                continue

            if wallet.coldkeypub.ss58_address != wallet_0.coldkeypub.ss58_address:
                bittensor.__console__.print(":cross_mark: [red]Hotkey: {} is not under the same coldkey. Skipping ...[/red]".format( wallet.hotkey_str ))
                continue

            # Assign decrypted coldkey from wallet_0
            #  so we don't have to decrypt again
            wallet._coldkey = wallet_0.coldkey
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
                    call_module='SubtensorModule', 
                    call_function='add_stake',
                    call_params={
                        'hotkey': wallet.hotkey.ss58_address,
                        'ammount_staked': staking_balance.rao
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

            with bittensor.__console__.status(":satellite: Staking to chain: [white]{}[/white] ...".format(self.network)):
                with self.substrate as substrate:
                    call = substrate.compose_call(
                    call_module='SubtensorModule', 
                    call_function='add_stake',
                    call_params={
                        'hotkey': wallet.hotkey.ss58_address,
                        'ammount_staked': staking_balance.rao
                        }
                    )
                    extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
                    response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                    # We only wait here if we expect finalization.
                    if not wait_for_finalization and not wait_for_inclusion:
                        bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                        old_balance -= staking_balance + stake_fee
                        successful_stakes += 1
                        if staking_all:
                            # If staked all, no need to continue
                            break

                        continue

                    response.process_events()
                    if response.is_success:
                        bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")
                    else:
                        bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))

            if response.is_success:
                block = self.get_current_block()
                new_stake = bittensor.Balance.from_tao( self.neuron_for_uid( uid = neuron.uid, block = block ).stake)
                new_balance = self.get_balance( wallet.coldkey.ss58_address )
                bittensor.__console__.print("Stake ({}): [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( neuron.uid, neuron.stake, new_stake ))
                old_balance = new_balance
                successful_stakes += 1
                if staking_all:
                    # If staked all, no need to continue
                    break
        
        if successful_stakes != 0:
            with bittensor.__console__.status(":satellite: Checking Balance on: ([white]{}[/white] ...".format(self.network)):
                new_balance = self.get_balance( wallet.coldkey.ss58_address )
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

    def unstake (
            self, 
            wallet: 'bittensor.wallet',
            amount: Union[Balance, float] = None, 
            wait_for_inclusion:bool = True, 
            wait_for_finalization:bool = False,
            prompt: bool = False,
        ) -> bool:
        r""" Removes stake into the wallet coldkey from the specified hotkey uid.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
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
        wallet.hotkey

        with bittensor.__console__.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(self.network)):
            old_balance = self.get_balance( wallet.coldkey.ss58_address )
            neuron = self.neuron_for_pubkey( ss58_hotkey = wallet.hotkey.ss58_address )
        if neuron.is_null:
            bittensor.__console__.print(":cross_mark: [red]Hotkey: {} is not registered.[/red]".format( wallet.hotkey_str ))
            return False

        # Covert to bittensor.Balance
        if amount == None:
            # Unstake it all.
            unstaking_balance = bittensor.Balance.from_tao( neuron.stake )
        elif not isinstance(amount, bittensor.Balance ):
            unstaking_balance = bittensor.Balance.from_tao( amount )
        else:
            unstaking_balance = amount

        # Check enough to unstake.
        stake_on_uid = bittensor.Balance.from_tao( neuron.stake )
        if unstaking_balance > stake_on_uid:
            bittensor.__console__.print(":cross_mark: [red]Not enough stake[/red]: [green]{}[/green] to unstake: [blue]{}[/blue] from hotkey: [white]{}[/white]".format(stake_on_uid, unstaking_balance, wallet.hotkey_str))
            return False

        # Estimate unstaking fee.
        unstake_fee = None # To be filled.
        with bittensor.__console__.status(":satellite: Estimating Staking Fees..."):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='SubtensorModule', 
                    call_function='remove_stake',
                    call_params={
                        'hotkey': wallet.hotkey.ss58_address,
                        'ammount_unstaked': unstaking_balance.rao
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

        with bittensor.__console__.status(":satellite: Unstaking from chain: [white]{}[/white] ...".format(self.network)):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='SubtensorModule', 
                    call_function='remove_stake',
                    call_params={
                        'hotkey': wallet.hotkey.ss58_address,
                        'ammount_unstaked': unstaking_balance.rao
                    }
                )
                extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
                response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                    return True

                response.process_events()
                if response.is_success:
                    bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")
                else:
                    bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))

        if response.is_success:
            with bittensor.__console__.status(":satellite: Checking Balance on: ([white]{}[/white] ...".format(self.network)):
                new_balance = self.get_balance( wallet.coldkey.ss58_address )
                block = self.get_current_block()
                new_stake = bittensor.Balance.from_tao( self.neuron_for_uid( uid = neuron.uid, block = block ).stake)
                bittensor.__console__.print("Balance: [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_balance, new_balance ))
                bittensor.__console__.print("Stake: [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( stake_on_uid, new_stake ))
                return True
        
        return False

    def unstake_multiple (
            self, 
            wallets: List['bittensor.wallet'],
            amounts: List[Union[Balance, float]] = None, 
            wait_for_inclusion: bool = True, 
            wait_for_finalization: bool = False,
            prompt: bool = False,
        ) -> bool:
        r""" Removes stake from each wallet hotkey in the list, using each amount, to their common coldkey.
        Args:
            wallets (List[bittensor.wallet]):
                List of wallets to unstake.
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
        if not isinstance(wallets, list):
            raise TypeError("wallets must be a list of bittensor.wallet")
        
        if len(wallets) == 0:
            return True

        if amounts is not None and len(amounts) != len(wallets):
            raise ValueError("amounts must be a list of the same length as wallets")

        if amounts is not None and not all(isinstance(amount, (Balance, float)) for amount in amounts):
            raise TypeError("amounts must be a [list of bittensor.Balance or float] or None")

        if amounts is None:
            amounts = [None] * len(wallets)
        else:
            # Convert to Balance
            amounts = [bittensor.Balance.from_tao(amount) if isinstance(amount, float) else amount for amount in amounts ]

            if sum(amount.tao for amount in amounts) == 0:
                # Staking 0 tao
                return True


        wallet_0: 'bittensor.wallet' = wallets[0]
        # Decrypt coldkey for all wallet(s) to use
        wallet_0.coldkey

        neurons = []
        with bittensor.__console__.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(self.network)):
            old_balance = self.get_balance( wallet_0.coldkey.ss58_address )

            for wallet in wallets:
                neuron = self.neuron_for_pubkey( ss58_hotkey = wallet.hotkey.ss58_address )

                if neuron.is_null:
                    neurons.append( None )
                    continue

                neurons.append( neuron )

        successful_unstakes = 0
        for wallet, amount, neuron in zip(wallets, amounts, neurons):
            if neuron is None:
                bittensor.__console__.print(":cross_mark: [red]Hotkey: {} is not registered. Skipping ...[/red]".format( wallet.hotkey_str ))
                continue

            if wallet.coldkeypub.ss58_address != wallet_0.coldkeypub.ss58_address:
                bittensor.__console__.print(":cross_mark: [red]Hotkey: {} is not under the same coldkey. Skipping ...[/red]".format( wallet.hotkey_str ))
                continue

            # Assign decrypted coldkey from wallet_0
            #  so we don't have to decrypt again
            wallet._coldkey = wallet_0._coldkey

            # Covert to bittensor.Balance
            if amount == None:
                # Unstake it all.
                unstaking_balance = bittensor.Balance.from_tao( neuron.stake )
            elif not isinstance(amount, bittensor.Balance ):
                unstaking_balance = bittensor.Balance.from_tao( amount )
            else:
                unstaking_balance = amount

            # Check enough to unstake.
            stake_on_uid = bittensor.Balance.from_tao( neuron.stake )
            if unstaking_balance > stake_on_uid:
                bittensor.__console__.print(":cross_mark: [red]Not enough stake[/red]: [green]{}[/green] to unstake: [blue]{}[/blue] from hotkey: [white]{}[/white]".format(stake_on_uid, unstaking_balance, wallet.hotkey_str))
                continue

            # Estimate unstaking fee.
            unstake_fee = None # To be filled.
            with bittensor.__console__.status(":satellite: Estimating Staking Fees..."):
                with self.substrate as substrate:
                    call = substrate.compose_call(
                        call_module='SubtensorModule', 
                        call_function='remove_stake',
                        call_params={
                            'hotkey': wallet.hotkey.ss58_address,
                            'ammount_unstaked': unstaking_balance.rao
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

            with bittensor.__console__.status(":satellite: Unstaking from chain: [white]{}[/white] ...".format(self.network)):
                with self.substrate as substrate:
                    call = substrate.compose_call(
                        call_module='SubtensorModule', 
                        call_function='remove_stake',
                        call_params={
                            'hotkey': wallet.hotkey.ss58_address,
                            'ammount_unstaked': unstaking_balance.rao
                        }
                    )
                    extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
                    response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                    # We only wait here if we expect finalization.
                    if not wait_for_finalization and not wait_for_inclusion:
                        bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                        successful_unstakes += 1
                        continue

                    response.process_events()
                    if response.is_success:
                        bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")
                    else:
                        bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))

            if response.is_success:
                block = self.get_current_block()
                new_stake = bittensor.Balance.from_tao( self.neuron_for_uid( uid = neuron.uid, block = block ).stake)
                bittensor.__console__.print("Stake ({}): [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( neuron.uid, stake_on_uid, new_stake ))
                successful_unstakes += 1
        
        if successful_unstakes != 0:
            with bittensor.__console__.status(":satellite: Checking Balance on: ([white]{}[/white] ...".format(self.network)):
                new_balance = self.get_balance( wallet.coldkey.ss58_address )
            bittensor.__console__.print("Balance: [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_balance, new_balance ))
            return True

        return False
                
    def set_weights(
            self, 
            wallet: 'bittensor.wallet',
            uids: Union[torch.LongTensor, list],
            weights: Union[torch.FloatTensor, list],
            wait_for_inclusion:bool = False,
            wait_for_finalization:bool = False,
            prompt:bool = False,
            netuid:int = None
        ) -> bool:
        r""" Sets the given weights and values on chain for wallet hotkey account.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
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
                        call_module='SubtensorModule',
                        call_function='set_weights',
                        call_params = {'dests': weight_uids, 'weights': weight_vals}
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

    def neurons(self, block: int = None ) -> List[SimpleNamespace]: 
        r""" Returns a list of neuron from the chain. 
        Args:
            block (int):
                block to sync from.
        Returns:
            neuron (List[SimpleNamespace]):
                List of neuron objects.
        """
        neurons = []
        for id in tqdm(range(self.get_n( block ))): 
            try:
                neuron = self.neuron_for_uid(id, block)
                neurons.append( neuron )
            except Exception as e:
                logger.error('Exception encountered when pulling neuron {}: {}'.format(id, e))
                break
        return neurons

    @staticmethod
    def _null_neuron() -> SimpleNamespace:
        neuron = SimpleNamespace()
        neuron.active = 0   
        neuron.stake = 0
        neuron.rank = 0
        neuron.trust = 0
        neuron.consensus = 0
        neuron.incentive = 0
        neuron.dividends = 0
        neuron.emission = 0
        neuron.weights = []
        neuron.bonds = []
        neuron.version = 0
        neuron.modality = 0
        neuron.uid = 0
        neuron.port = 0
        neuron.priority = 0
        neuron.ip_type = 0
        neuron.last_update = 0
        neuron.ip = 0
        neuron.is_null = True
        neuron.coldkey = "000000000000000000000000000000000000000000000000"
        neuron.hotkey = "000000000000000000000000000000000000000000000000"
        return neuron

    @staticmethod
    def _neuron_dict_to_namespace(neuron_dict) -> SimpleNamespace:
        if neuron_dict['hotkey'] == '5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM':
            return Subtensor._null_neuron()
        else:
            RAOPERTAO = 1000000000
            U64MAX = 18446744073709551615
            neuron = SimpleNamespace( **neuron_dict )
            neuron.stake = neuron.stake / RAOPERTAO
            neuron.rank = neuron.rank / U64MAX
            neuron.trust = neuron.trust / U64MAX
            neuron.consensus = neuron.consensus / U64MAX
            neuron.incentive = neuron.incentive / U64MAX
            neuron.dividends = neuron.dividends / U64MAX
            neuron.emission = neuron.emission / RAOPERTAO
            neuron.is_null = False
            return neuron

    def neuron_for_uid( self, uid: int, block: int = None, netuid: int = None ) -> Union[ dict, None ]: 
        r""" Returns a list of neuron from the chain. 
        Args:
            uid ( int ):
                The uid of the neuron to query for.
            block ( int ):
                The neuron at a particular block
        Returns:
            neuron (dict(NeuronMetadata)):
                neuron object associated with uid or None if it does not exist.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                result = dict( substrate.query( 
                    module='SubtensorModule',  
                    storage_function='Neurons', 
                    params = [ uid ], 
                    block_hash = None if block == None else substrate.get_block_hash( block )
                ).value )
            return result
        result = make_substrate_call_with_retry()
        neuron = Subtensor._neuron_dict_to_namespace( result )
        return neuron

    def get_uid_for_hotkey( self, ss58_hotkey: str, block: int = None, netuid: int = None ) -> int:
        r""" Returns true if the passed hotkey is registered on the chain.
        Args:
            ss58_hotkey ( str ):
                The hotkey to query for a neuron.
        Returns:
            uid ( int ):
                UID of passed hotkey or -1 if it is non-existent.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query (
                    module='SubtensorModule',
                    storage_function='Hotkeys',
                    params = [ ss58_hotkey ],
                    block_hash = None if block == None else substrate.get_block_hash( block )
                )
        result = make_substrate_call_with_retry()
        # Process the result.
        uid = int(result.value)
        
        neuron = self.neuron_for_uid( uid, block )
        if neuron.hotkey != ss58_hotkey:
            return -1
        else:
            return uid


    def is_hotkey_registered( self, ss58_hotkey: str, block: int = None) -> bool:
        r""" Returns true if the passed hotkey is registered on the chain.
        Args:
            ss58_hotkey ( str ):
                The hotkey to query for a neuron.
        Returns:
            is_registered ( bool):
                True if the passed hotkey is registered on the chain.
        """
        uid = self.get_uid_for_hotkey( ss58_hotkey = ss58_hotkey, block = block)
        if uid == -1:
            return False
        else:
            return True

    def neuron_for_pubkey( self, ss58_hotkey: str, block: int = None ) -> SimpleNamespace: 
        r""" Returns a list of neuron from the chain. 
        Args:
            ss58_hotkey ( str ):
                The hotkey to query for a neuron.
        Returns:
            neuron ( dict(NeuronMetadata) ):
                neuron object associated with uid or None if it does not exist.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query (
                    module='SubtensorModule',
                    storage_function='Hotkeys',
                    params = [ ss58_hotkey ],
                    block_hash = None if block == None else substrate.get_block_hash( block )
                )
        result = make_substrate_call_with_retry()
        # Get response uid. This will be zero if it doesn't exist.
        uid = int(result.value)
        neuron = self.neuron_for_uid( uid, block )
        if neuron.hotkey != ss58_hotkey:
            return Subtensor._null_neuron()
        else:
            return neuron

    def get_n( self, block: int = None ) -> int: 
        r""" Returns the number of neurons on the chain at block.
        Args:
            block ( int ):
                The block number to get the neuron count from.
        Returns:
            n ( int ):
                the number of neurons subscribed to the chain.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return int(substrate.query(  module='SubtensorModule', storage_function = 'N', block_hash = None if block == None else substrate.get_block_hash( block ) ).value)
        return make_substrate_call_with_retry()

    def neuron_for_wallet( self, wallet: 'bittensor.Wallet', block: int = None ) -> SimpleNamespace: 
        r""" Returns a list of neuron from the chain. 
        Args:
            wallet ( `bittensor.Wallet` ):
                Checks to ensure that the passed wallet is subscribed.
        Returns:
            neuron ( dict(NeuronMetadata) ):
                neuron object associated with uid or None if it does not exist.
        """
        return self.neuron_for_pubkey ( wallet.hotkey.ss58_address, block = block )
