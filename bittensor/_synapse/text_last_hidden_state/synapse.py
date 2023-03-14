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
import copy
import torch
import bittensor
import argparse

class TextLastHiddenStateSynapse( bittensor.Synapse, bittensor.grpc.TextLastHiddenStateServicer ):
    """ TextLastHiddenStateSynapse: A class for servicing text_last_hidden_state requests."""

    # Synapse name.
    name: str = 'text_last_hidden_state'

    def __init__(
            self, 
            config: 'bittensor.Config' = None, 
            metagraph: 'bittensor.metagraph.Metagraph' = None,
            wallet: 'bittensor.wallet' = None
        ):
        if config == None: config = TextLastHiddenStateSynapse.config()
        TextLastHiddenStateSynapse.check_config( config )
        super().__init__( config, metagraph )
        self.config = copy.deepcopy( config )
        self.metagraph = metagraph
        self.wallet = wallet
        self.priority_threadpool = bittensor.prioritythreadpool( config = config.synapse.text_last_hidden_state )

    def __str__(self):
        return 'TextLastHiddenState'

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
        bittensor.prioritythreadpool.add_args( parser, prefix = prefix_str + 'synapse.text_last_hidden_state' )
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
     
    def _attach( self, axon: 'bittensor.axon' ):
        """ _attach: Attaches the synapse to the axon."""
        bittensor.grpc.add_TextLastHiddenStateServicer_to_server( self, axon.server )

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
    
    def forward( self, forward_call: 'bittensor.TextLastHiddenStateForwardCall' ) -> bittensor.TextLastHiddenStateForwardCall:
        """ fills in the hidden_states of the forward call.
            Args:
                forward_call (:obj:`bittensor.TextLastHiddenStateForwardCall`, `required`):
                    bittensor forward call dataclass to fill.
            Returns:
                forward_call (:obj:`bittensor.TextLastHiddenStateForwardCall`, `required`):
                    filled bittensor forward call dataclass.
        """
        raise NotImplementedError('Must implement forward() in subclass.')
    
    def pre_process_request_proto_to_forward_call( 
            self, 
            request_proto: bittensor.ForwardTextLastHiddenStateRequest 
        ) -> 'bittensor.TextLastHiddenStateForwardCall':
        """ pre_process_request_proto_to_forward_call
            ------------------------------------------
            Args:
                request_proto (bittensor.ForwardTextLastHiddenStateRequest):
                    bittensor forward request proto.
            Returns:
                bittensor.TextLastHiddenStateForwardCall (:obj:`bittensor.TextLastHiddenStateForwardCall`, `required`):
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

        return bittensor.TextLastHiddenStateForwardCall(
            text_inputs = text_inputs,
            mask = mask,
            timeout = request_proto.timeout,
            mask_serializer_type = request_proto.mask_serializer_type,
            text_inputs_serializer_type = request_proto.text_inputs_serializer_type,
            hidden_states_serializer_type = request_proto.hidden_states_serializer_type,
        )
    
    def post_process_forward_call_to_response_proto( 
            self, 
            forward_call: 'bittensor.TextLastHiddenStateForwardCall' 
        ) -> bittensor.ForwardTextLastHiddenStateResponse:
        """ post_process_forward_call_to_response_proto
            --------------------------------------------
            Args:
                forward_call (bittensor.TextLastHiddenStateForwardCall):
                    forward_call.text_inputs (torch.FloatTensor): text inputs.
                    forward_call.timeout (float): timeout for the request.
                    forward_call.text_inputs_serializer_type (bittensor.proto.SerializerType): text inputs serializer type.
                    forward_call.hidden_states_serializer_type (bittensor.proto.SerializerType): hidden states serializer type.
                    forward_call.hidden_states (torch.FloatTensor): hidden states.
            Returns:    
                response (bittensor.ForwardTextLastHiddenStateResponse):
                    response.serialized_hidden_states (string): serialized hidden states.
        """
        # Serialize hidden states.
        hidden_state_serializer = bittensor.serializer( serializer_type = forward_call.hidden_states_serializer_type )

        # Check if response is sucessful
        if (forward_call.request_code != bittensor.proto.ReturnCode.Success) or \
            (forward_call.response_code != bittensor.proto.ReturnCode.Success):
            serialized_hidden_states = None

        else:
            # Optionally apply mask.
            if forward_call.mask != None:
                # Apply mask.
                hidden_states = forward_call.hidden_states.reshape( -1, bittensor.__network_dim__ )

                # Filter hidden states.
                hidden_states = hidden_states[ forward_call.mask.reshape(-1) ]

            # Else return the raw hidden states.
            else:
                hidden_states = forward_call.hidden_states
            serialized_hidden_states = hidden_state_serializer.serialize( hidden_states )
            
        # Return the forward response proto.
        return bittensor.ForwardTextLastHiddenStateResponse(
            version = bittensor.__version_as_int__, 
            serialized_hidden_states = serialized_hidden_states,
            hotkey = self.wallet.hotkey.ss58_address, 
            return_code = forward_call.request_code,
            message = forward_call.request_message
        )
    
    def pre_process_request_proto_to_backward_call( 
        self, 
        request_proto: 'bittensor.BackwardRequest' 
    ) -> 'bittensor.BittensorCall':
        """ pre_process_request_proto_to_backward_call
            ------------------------------------------
            Args:
                request_proto (bittensor.BackwardRequest):
                    request_proto to process in to a backward call.
            Returns:
                bittensor.BittensorCall (:obj:`bittensor.BittensorCall`, `required`):
                    backward call processed from the request proto.
        """
        text_deserializer = bittensor.serializer( serializer_type = request_proto.text_inputs_serializer_type )
        text_inputs = text_deserializer.deserialize( request_proto.serialized_text_inputs )

        hidden_states_deserializer = bittensor.serializer( serializer_type = request_proto.hidden_states_serializer_type )
        hidden_states = hidden_states_deserializer.deserialize( request_proto.serialized_hidden_states )

        hidden_states_grads_deserializer = bittensor.serializer( serializer_type = request_proto.hidden_states_grads_serializer_type )
        hidden_states_grads = hidden_states_grads_deserializer.deserialize( request_proto.serialized_hidden_states_grads )

        # Optionally deserialize mask.
        try:
            mask_serializer = bittensor.serializer( serializer_type = request_proto.mask_serializer_type )
            mask = mask_serializer.serialize( request_proto.serialized_mask )
        except:
            mask = None

        # If the mask is not none, we need to expand the hidden states to the proper size.
        if mask != None:
            # From the encode_forward_response function the forward_response_tensor is [ len(mask), net_dim ]
            # a set of rows from the stacked_forward_response_tensor = [ bs * seq, net_dim ]
            # We will load these rows into a destination tensor = [bs, seq, net_dim]
            hidden_states_destination = torch.zeros( [ mask.size(0) * mask.size(1), bittensor.__network_dim__ ])
            hidden_states_grads_destination = torch.zeros( [ mask.size(0) * mask.size(1), bittensor.__network_dim__ ])

            # Iterate through the mask and fill the destination tensor 
            # with the hidden states from the forward call.
            counter = 0
            for i, not_masked in enumerate(mask.reshape(-1)):
                if not_masked:
                    hidden_states_destination[i, :] = hidden_states[counter, :]
                    hidden_states_grads_destination[i, :] = hidden_states_grads[counter, :]
                    counter += 1
            
            # Reshape the destination tensor to the proper expanded size.
            hidden_states = hidden_states_destination.reshape( ( mask.size(0), mask.size(1), bittensor.__network_dim__) )
            hidden_states_grads = hidden_states_grads_destination.reshape( (mask.size(0), mask.size(1), bittensor.__network_dim__) )

        # Return backward call.
        return bittensor.TextLastHiddenStateForwardCall(
            mask = mask,
            text_inputs = text_inputs,
            hidden_states = hidden_states,
            hidden_states_grads = hidden_states_grads,

            mask_serializer_type = request_proto.mask_serializer_type,
            text_inputs_serializer_type = request_proto.text_inputs_serializer_type,
            hidden_states_serializer_type = request_proto.hidden_states_serializer_type,
            hidden_states_grads_serializer_type = request_proto.hidden_states_grads_serializer_type,
        )
    