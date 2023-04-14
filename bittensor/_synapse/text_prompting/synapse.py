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

import os
import json
import grpc
import time
import copy
import torch
import argparse
import bittensor

from rich import print
from typing import List, Dict, Union
from datetime import datetime
from abc import ABC, abstractmethod

class TextPromptingSynapse( ABC, bittensor.grpc.TextPromptingServicer ):

    ##############
    #### INIT ####
    ##############
    def __init__(
        self,
        config: "bittensor.Config" = None
    ):
        config = config if config != None else self.config()
        self.config = copy.deepcopy( config )
        self.super_check_config( self.config )
        self.config.to_defaults()
        bittensor.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        self.subtensor = bittensor.subtensor( self.config )
        self.wallet = bittensor.wallet( self.config )
        self.metagraph = self.subtensor.metagraph( self.config.netuid )
        self.axon = bittensor.axon( 
            wallet = self.wallet,
            metagraph = self.metagraph,
            config = self.config,
        )
        bittensor.grpc.add_TextPromptingServicer_to_server( self, self.axon.server )
        self.axon.start()

    ##############
    #### Args ####
    ##############
    @classmethod
    @abstractmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        ...

    @classmethod
    def add_super_args( cls, parser: argparse.ArgumentParser ):
        cls.add_args(parser)
        parser.add_argument(
            '--netuid', 
            type = int, 
            help = 'Subnet netuid', 
            default = 1
        )
        parser.add_argument(
            '--neuron.name', 
            type = str,
            help = 'Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ',
            default = 'openai_prompting_miner'
        )
        parser.add_argument(
            '--neuron.blocks_per_epoch', 
            type = str, 
            help = 'Blocks until the miner sets weights on chain',
            default = 100
        )
        parser.add_argument(
            '--neuron.no_set_weights', 
            action = 'store_true', 
            help = 'If True, the model does not set weights.',
            default = False
        )
        parser.add_argument(
            '--neuron.max_batch_size', 
            type = int, 
            help = 'The maximum batch size for forward requests.',
            default = -1
        )
        parser.add_argument(
            '--neuron.max_sequence_len', 
            type = int, 
            help = 'The maximum sequence length for forward requests.',
            default = -1
        )
        parser.add_argument(
            '--neuron.blacklist.hotkeys', 
            type = str, 
            required = False, 
            nargs = '*', 
            action = 'store',
            help = 'To blacklist certain hotkeys', default=[]
        )
        parser.add_argument(
            '--neuron.blacklist.allow_non_registered',
            action = 'store_true',
            help = 'If True, the miner will allow non-registered hotkeys to mine.',
            default = True
        )
        parser.add_argument(
            '--neuron.blacklist.default_stake',
            type = float,
            help = 'Set default stake for miners.',
            default = 0.0
        )
        parser.add_argument(
            '--neuron.default_priority',
            type = float,
            help = 'Set default priority for miners.',
            default = 0.0
        )
        bittensor.wallet.add_args( parser )
        bittensor.axon.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.logging.add_args( parser )
        bittensor.metagraph.add_args( parser )

    ################
    #### Config ####
    ################
    @classmethod
    @abstractmethod
    def check_config( cls, config: 'bittensor.Config' ):
        ...

    @classmethod
    def config( cls ) -> "bittensor.Config":
        parser = argparse.ArgumentParser()
        cls.add_super_args( parser )
        return bittensor.config( parser )

    @classmethod
    def super_check_config( cls, config: "bittensor.Config" ):
        cls.check_config( config )
        bittensor.axon.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.logging.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        full_path = os.path.expanduser(
            '{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.get('name', bittensor.defaults.wallet.name),
                                  config.wallet.get('hotkey', bittensor.defaults.wallet.hotkey), config.neuron.name ) )
        config.neuron.full_path = os.path.expanduser( full_path )
        if not os.path.exists( config.neuron.full_path ):
            os.makedirs( config.neuron.full_path )

    @classmethod
    def help( cls ):
        parser = argparse.ArgumentParser()
        cls.add_super_args( parser )
        cls.add_args(parser)
        print( cls.__new__.__doc__ )
        parser.print_help()

    ######################
    #### Blacklisting ####
    #####################
    def priority( self, request: Union[ bittensor.ForwardTextPromptingRequest, bittensor.BackwardTextPromptingRequest ] ) -> float:
        if self.metagraph is not None:
            uid = self.metagraph.hotkeys.index( request.hotkey )
            return self.metagraph.S[uid].item()
        else:
            return self.config.neuron.default_priority

    def blacklist( self, request: Union[ bittensor.ForwardTextPromptingRequest, bittensor.BackwardTextPromptingRequest ] ) -> bool:
        # Check for registration
        def registration_check():
            is_registered = request.hotkey in self.metagraph.hotkeys
            if not is_registered:
                if self.config.neuron.blacklist.allow_non_registered:
                    return False
                bittensor.logging.debug( "Blacklisted. Not registered.")
                raise Exception("Registration blacklist") 
        
        # Blacklist based on stake.
        def stake_check() -> bool:
            default_stake = self.config.neuron.blacklist.default_stake
            if default_stake <= 0.0:
                return False
            uid = self.metagraph.hotkeys.index( request.hotkey )
            if self.metagraph.S[uid].item() < default_stake:
                bittensor.logging.debug( "Blacklisted. Stake too low.")
                raise Exception("Stake blacklist")
            return False

        # Optionally blacklist based on checks.
        try:
            registration_check()
            stake_check()
            return False
        except Exception as e:
            bittensor.logging.warning( "Blacklisted. Error in `registration_check` or `stake_check()" )
            return True
    

    #################
    #### Forward ####
    #################
    @abstractmethod
    def forward( self, messages: List[Dict[str, str]] ) -> str:
        ...

    def apply_forward_request( self, request: bittensor.ForwardTextPromptingRequest ) -> bittensor.ForwardTextPromptingResponse:
        formatted_messages = [ json.loads( message ) for message in request.messages ]
        completion = self.forward( messages = formatted_messages )
        return bittensor.ForwardTextPromptingResponse(
            version = bittensor.__version_as_int__,
            hotkey = self.axon.wallet.hotkey.ss58_address,
            response = completion,
            return_code = bittensor.proto.ReturnCode.Success,
        )
    
    def Forward( self, request: bittensor.ForwardTextPromptingRequest, context: grpc.ServicerContext ) -> bittensor.ForwardTextPromptingResponse:
        r"""Forward TextPrompting
        ----------------------------
        Args:
            request (bittensor.ForwardRequest):
                text prompting forward request
            context (grpc.ServicerContext):
                grpc tcp context.
        Returns:
            response (bittensor.ForwardResponse):
                text prompting forward response
        """
        try: 
            uid = self.metagraph.hotkeys.index( request.hotkey )
        except:
            uid = None
        start_time = time.time()
        if self.blacklist( request ):
            raise Exception("Blacklisted")
        priority = self.priority( request )
        forward_future = self.axon.priority_threadpool.submit(
            self.apply_forward_request,
            request = request,
            priority = priority,
        )
        bittensor.logging.rpc_log(
            axon = True,
            forward = True,
            is_response = True,
            code = bittensor.proto.ReturnCode.Success,
            call_time = time.time() - start_time,
            pubkey = request.hotkey,
            uid = uid,
            inputs = torch.Size( [len(message) for message in request.messages] ),
            outputs = None,
            message = "Success",
            synapse = "TextPrompting",
        )
        response = forward_future.result( timeout = request.timeout )
        bittensor.logging.rpc_log(
            axon = True,
            forward = True,
            is_response = False,
            code = bittensor.proto.ReturnCode.Success,
            call_time = time.time() - start_time,
            pubkey = request.hotkey,
            uid = uid,
            inputs = torch.Size( [len(message) for message in request.messages] ),
            outputs = torch.Size([len(response.response)]),
            message = "Success",
            synapse = "TextPrompting",
        )
        return response
    
    ##################
    #### Backward ####
    ##################
    def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ):
        pass

    def apply_backward_request( self, request: bittensor.BackwardTextPromptingRequest ):
        formatted_messages = [ json.loads(message) for message in request.messages ]
        formatted_rewards = torch.tensor( [ request.rewards ], dtype = torch.float32 )
        self.backward(
            messages = formatted_messages,
            response = request.response,
            rewards = formatted_rewards
        )    

    def Backward( self, request: bittensor.BackwardTextPromptingRequest, context: grpc.ServicerContext) -> bittensor.BackwardTextPromptingResponse:
        """BackwardTextPrompting
        ----------------------------
         Args:
            request ( bittensor.BackwardTextPromptingRequest ):
                text prompting backward request
            context ( grpc.ServicerContext ):
                grpc tcp context.
        """
        try: 
            uid = self.metagraph.hotkeys.index( request.hotkey )
        except:
            uid = None
        
        try:
            start_time = time.time()
            if self.blacklist( request ):
                raise Exception("Blacklisted")
            priority = self.priority( request )
            self.axon.priority_threadpool.submit(
                self.apply_backward_request,
                request = request,
                priority = priority,
            )
            bittensor.logging.rpc_log(
                axon = True,
                forward = False,
                is_response = True,
                code = bittensor.proto.ReturnCode.Success,
                call_time = time.time() - start_time,
                pubkey = request.hotkey,
                uid = uid,
                inputs = torch.Size( [ len(request.rewards) ] ),
                outputs = None,
                message = 'Success',
                synapse = 'Text Prompting',
            )
            code = bittensor.proto.ReturnCode.Success
            message = "Success"
            
        except Exception as e:
            code = bittensor.proto.ReturnCode.UnknownException
            message = str(e)
        finally:
            bittensor.logging.rpc_log(
                axon = True,
                forward = False,
                is_response = False,
                code = code,
                call_time = time.time() - start_time,
                pubkey = request.hotkey,
                uid = uid,
                inputs = torch.Size( [ len(request.rewards) ] ),
                outputs = None,
                message = message,
                synapse = 'Text Prompting',
            )
            return bittensor.ForwardTextPromptingResponse(
                version = bittensor.__version_as_int__,
                hotkey = self.axon.wallet.hotkey.ss58_address,
                return_code = code
            )


    ##################
    #### RUN ####
    ##################
    def run( self ):

        # --- Start the miner.
        self.wallet.reregister( netuid = self.config.netuid, subtensor = self.subtensor )
        self.axon.netuid = self.config.netuid
        self.axon.protocol = 4
        self.subtensor.serve_axon( self.axon )

        # --- Run Forever.
        last_update = self.subtensor.get_current_block()
        while True:

            # --- Wait until next epoch.
            current_block = self.subtensor.get_current_block()
            while (current_block - last_update) < self.config.neuron.blocks_per_epoch:
                time.sleep( 0.1 ) #bittensor.__blocktime__
                current_block = self.subtensor.get_current_block()
            last_update = self.subtensor.get_current_block()

            # --- Update the metagraph with the latest network state.
            self.metagraph.sync( netuid = self.config.netuid, subtensor = self.subtensor )
            uid = self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )

            # --- Log performance.
            print(
                f"[white not bold]{datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
                f"{f'UID [bright_cyan]{uid}[/bright_cyan]'.center(16 + len('[bright_cyan][/bright_cyan]'))} | "
                f'[dim white not bold] [green]{str(self.metagraph.S[uid].item()):.4}[/green] Stake [/dim white not bold]'
                f'[dim white not bold]| [yellow]{str(self.metagraph.trust[uid].item()) :.3}[/yellow] Trust [/dim white not bold]'
                f'[dim white not bold]| [green]{str(self.metagraph.incentive[uid].item()):.3}[/green] Incentive [/dim white not bold]')

            # --- Set weights.
            if not self.config.neuron.no_set_weights:
                try:
                    # --- query the chain for the most current number of peers on the network
                    chain_weights = torch.zeros( self.subtensor.subnetwork_n( netuid = self.config.netuid ))
                    chain_weights[uid] = 1
                    did_set = self.subtensor.set_weights(
                        uids=torch.arange(0, len(chain_weights)),
                        netuid=self.config.netuid,
                        weights=chain_weights,
                        wait_for_inclusion=False,
                        wallet=self.wallet,
                        version_key=1
                    )
                except:
                    pass
