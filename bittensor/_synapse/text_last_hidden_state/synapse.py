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
import grpc
import torch
import bittensor
import argparse

class TextLastHiddenStateSynapse( bittensor.Synapse ):
    """ TextLastHiddenStateSynapse: A class for servicing text_last_hidden_state requests."""

    def __init__(
            self, 
            config: 'bittensor.Config' = None, 
            metagraph: 'bittensor.metagraph.Metagraph' = None
        ):
        if config == None: config = bittensor.config()
        TextLastHiddenStateSynapse.check_config( config )
        super().__init__( config, metagraph )
        self.config = config
        self.metagraph = metagraph

    @classmethod
    def config(cls) -> 'bittensor.Config':
        """ Returns the config for this synapse."""
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None ):
        """ Accept specific arguments from parser
        """
        prefix_str = '' if prefix == None else prefix + '.'
        super().add_args( parser = parser, prefix = prefix )
        try:
            parser.add_argument('--' + prefix_str + 'synapse.text_last_hidden_state.blacklist.stake', type=float, help='The amount of stake (tao) required to make a call.', default=10)
            parser.add_argument('--' + prefix_str + 'synapse.text_last_hidden_state.blacklist.allow_non_registered', action='store_true', help='''If true, allow non-registered peers''', default=True)
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod   
    def help(cls):
        """ Print help to stdout """
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        print (cls.__new__.__doc__)
        parser.print_help()

    @classmethod   
    def add_defaults(cls, defaults):
        """ Add default values to defaults object"""
        defaults.synapse = bittensor.Config()
        defaults.synapse.text_last_hidden_state.blacklist.stake = os.getenv('BT_SYNAPSE_TEXT_LAST_HIDDEN_STATE_BLACKLIST_STAKE') if os.getenv('BT_SYNAPSE_TEXT_LAST_HIDDEN_STATE_BLACKLIST_STAKE') != None else 10
        defaults.synapse.text_last_hidden_state.blacklist.allow_non_registered = os.getenv('BT_SYNAPSE_TEXT_LAST_HIDDEN_STATE_BLACKLIST_ALLOW_NON_REGISTERED') if os.getenv('BT_SYNAPSE_TEXT_LAST_HIDDEN_STATE_BLACKLIST_ALLOW_NON_REGISTERED') != None else True

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    def __str__(self):
        return 'TextLastHiddenState'
    
    def priority( self, forward_call: 'bittensor.TextSeq2SeqBittensorCall' ) -> float:
        """ priority: Returns the priority of the synapse for the given hotkey and text_inputs."""
        raise NotImplementedError('Must implement priority() in subclass.')
    
    def _priority( self, forward_call: 'bittensor.TextSeq2SeqBittensorCall' ) -> float:
        """ _priority: Returns the priority of the forward call.
            Args:
                forward_call (:obj:`bittensor.BittensorCall`, `required`):
                    forward_call to check.
            Returns:
                float: priority of the forward call.
        """
        return self.priority( forward_call)

    def blacklist( self, forward_call: 'bittensor.TextSeq2SeqBittensorCall'  ) -> bool:
        """ blacklist: Returns True if the synapse should not be called for the given hotkey and text_inputs."""
        raise NotImplementedError('Must implement blacklist() in subclass.')
    
    def _blacklist( self, forward_call: bittensor.BittensorCall ) -> bool:
        """ __blacklist: Checks if the forward call is blacklisted.
            Args:
                forward_call (:obj:`bittensor.BittensorCall`, `required`):
                    forward_call to check.
            Returns:
                bool: True if blacklisted, False otherwise.
        """

        # Call subclass blacklist and optionaly return if metagraph is None.
        try:
            subclass_blacklist = self.blacklist( forward_call )
        except:
            # subclass_blacklist not implemented.
            subclass_blacklist = False
        if subclass_blacklist:
            return subclass_blacklist
        elif self.metagraph == None:
            return subclass_blacklist

        # Check for registration
        def registration_check():
            is_registered = forward_call.hotkey in self.metagraph.hotkeys
            if not is_registered:
                if self.config.synapse.text_seq2seq.blacklist.allow_non_registered:
                    return False
                raise Exception('Registration blacklist')

        # Blacklist based on stake.
        def stake_check() -> bool:
            uid = self.metagraph.hotkeys.index( forward_call.hotkey )
            if self.metagraph.S[uid].item() < self.config.synapse.text_seq2seq.blacklist.stake:
                raise Exception('Stake blacklist')
            return False

        # Optionally blacklist based on checks.
        try:
            registration_check()
            stake_check()            
            return False
        except Exception as e:
            return True

    def forward( self, forward_call: 'bittensor.TextLastHiddenStateBittensorCall' ) -> bittensor.TextLastHiddenStateBittensorCall:
        """ fills in the hidden_states of the forward call.
            Args:
                forward_call (:obj:`bittensor.TextLastHiddenStateBittensorCall`, `required`):
                    bittensor forward call dataclass to fill.
            Returns:
                forward_call (:obj:`bittensor.TextLastHiddenStateBittensorCall`, `required`):
                    filled bittensor forward call dataclass.
        """
        raise NotImplementedError('Must implement forward() in subclass.')
    
    def pre_process_request_proto_to_forward_call( 
            self, 
            request_proto: bittensor.ForwardTextLastHiddenStateRequest 
        ) -> 'bittensor.TextLastHiddenStateBittensorCall':
        """ pre_process_request_proto_to_forward_call
            ------------------------------------------
            Args:
                request_proto (bittensor.ForwardTextLastHiddenStateRequest):
                    bittensor forward request proto.
            Returns:
                bittensor.TextLastHiddenStateBittensorCall (:obj:`bittensor.TextLastHiddenStateBittensorCall`, `required`):
                    bittensor forward call dataclass.
        """
        # Deserialize text inputs.
        text_deserializer = bittensor.serializer( serializer_type = request_proto.text_inputs_serializer_type )
        text_inputs = text_deserializer.deserialize( request_proto.serialized_text_inputs )
        # Optionally deserialize mask.
        if len( request_proto.serialized_mask.shape ) > 0:
            mask_deserializer = bittensor.serializer( serializer_type = request_proto.mask_serializer_type )
            mask = mask_deserializer.deserialize( request_proto.serialized_mask )
        else:
            mask = None

        return bittensor.TextLastHiddenStateBittensorCall(
            text_inputs = text_inputs,
            mask = mask,
            timeout = request_proto.timeout,
            mask_serializer_type = request_proto.mask_serializer_type,
            text_inputs_serializer_type = request_proto.text_inputs_serializer_type,
            hidden_states_serializer_type = request_proto.hidden_states_serializer_type,
        )
    
    def post_process_forward_call_to_response_proto( 
            self, 
            forward_call: 'bittensor.TextLastHiddenStateBittensorCall' 
        ) -> bittensor.ForwardTextLastHiddenStateResponse:
        """ post_process_forward_call_to_response_proto
            --------------------------------------------
            Args:
                forward_call (bittensor.TextLastHiddenStateBittensorCall):
                    forward_call.text_inputs (torch.FloatTensor): text inputs.
                    forward_call.timeout (float): timeout for the request.
                    forward_call.text_inputs_serializer_type (bittensor.proto.SerializerType): text inputs serializer type.
                    forward_call.hidden_states_serializer_type (bittensor.proto.SerializerType): hidden states serializer type.
                    forward_call.hidden_states (torch.FloatTensor): hidden states.
            Returns:    
                response (bittensor.ForwardTextLastHiddenStateResponse):
                    response.serialized_hidden_states (string): serialized hidden states.
        """
        # Optionally apply mask.
        if forward_call.mask != None:
            # Apply mask.
            hidden_states = forward_call.hidden_states.reshape( -1, bittensor.__network_dim__ )

            # Filter hidden states.
            hidden_states = hidden_states[ forward_call.mask.reshape(-1) ]

        # Else return the raw hidden states.
        else:
            hidden_states = forward_call.hidden_states

        # Serialize hidden states.
        hidden_state_serializer = bittensor.serializer( serializer_type = forward_call.hidden_states_serializer_type )
        serialized_hidden_states = hidden_state_serializer.serialize( hidden_states )

        # Return the forward response proto.
        return bittensor.ForwardTextLastHiddenStateResponse(
            serialized_hidden_states = serialized_hidden_states
        )
    
    def ForwardTextLastHiddenState( 
            self, 
            request: bittensor.ForwardTextLastHiddenStateRequest, 
            context: grpc.ServicerContext 
        ) -> bittensor.ForwardTextLastHiddenStateResponse:
        """ ForwardTextLastHiddenState
            ----------------------------
            Args:
                request (bittensor.ForwardTextLastHiddenStateRequest): 
                    request.version (int): version of the caller.
                    request.hotkey (string): hotkey of the neuron.
                    request.timeout (float): timeout for the request.
                    request.text_inputs_serializer_type (bittensor.proto.SerializerType): text inputs serializer type.
                    request.serialized_text_inputs (string): serialized text inputs.
                context (grpc.ServicerContext):
                    grpc tcp context.
            Returns:
                response (bittensor.ForwardTextLastHiddenStateResponse): 
                    response.serialized_hidden_states (string): serialized hidden states.
        """
        return self._Forward( request_proto = request )
    