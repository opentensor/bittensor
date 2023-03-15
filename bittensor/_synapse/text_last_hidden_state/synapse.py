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
import time
import grpc
import torch
import argparse
import bittensor

from loguru import logger

class TextLastHiddenStateSynapse( bittensor.grpc.TextLastHiddenStateServicer ):
    """ TextLastHiddenStateSynapse: A class for servicing text_last_hidden_state requests."""

    # Synapse name.
    name: str = 'text_last_hidden_state'

    def __init__(
            self, 
            wallet: 'bittensor.wallet',
            metagraph: 'bittensor.metagraph.Metagraph' = None,
            config: 'bittensor.Config' = None, 
        ):
        if config == None: config = TextLastHiddenStateSynapse.config()
        TextLastHiddenStateSynapse.check_config( config )
        self.config = copy.deepcopy( config )

        self.metagraph = metagraph
        self.wallet = wallet
        self.priority_threadpool = bittensor.prioritythreadpool( config = config.text_last_hidden_state )

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
        bittensor.prioritythreadpool.add_args( parser, prefix = prefix_str + 'text_last_hidden_state' )
        try:
            parser.add_argument('--' + prefix_str + 'text_last_hidden_state.blacklist.stake', type=float, help='The amount of stake (tao) required to make a call.', default=10)
            parser.add_argument('--' + prefix_str + 'text_last_hidden_state.blacklist.allow_non_registered', action='store_true', help='''If true, allow non-registered peers''', default=True)
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
        defaults.text_last_hidden_state = bittensor.Config()
        defaults.text_last_hidden_state.blacklist.stake = os.getenv('BT_TEXT_LAST_HIDDEN_STATE_BLACKLIST_STAKE') if os.getenv('BT_TEXT_LAST_HIDDEN_STATE_BLACKLIST_STAKE') != None else 10
        defaults.text_last_hidden_state.blacklist.allow_non_registered = os.getenv('BT_TEXT_LAST_HIDDEN_STATE_BLACKLIST_ALLOW_NON_REGISTERED') if os.getenv('BT_TEXT_LAST_HIDDEN_STATE_BLACKLIST_ALLOW_NON_REGISTERED') != None else True

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass
     
    def _attach( self, axon: 'bittensor.axon' ):
        """ _attach: Attaches the synapse to the axon."""
        bittensor.grpc.add_TextLastHiddenStateServicer_to_server( self, axon.server )

    def forward( 
            self, 
            text_inputs: torch.LongTensor,
            hotkey: str,
        ) -> torch.FloatTensor:
        """ fills in the hidden_states of the forward call.
            Args:
                text_inputs (:obj:`torch.LongTensor`, `required`):
                    tokenized text inputs.
                hotkey (:obj:`str`, `required`):
                    hotkey of the calling neuron
            Returns:
                hidden_states (:obj:`torch.FloatTensor`, `required`):
                    hidden states of the last layer of the model.
        """
        raise NotImplementedError('Must implement forward() in subclass.')
    
    def _forward( 
            self, 
            forward_call: 'bittensor.TextLastHiddenStateForwardCall' 
        ) -> bittensor.TextLastHiddenStateForwardCall:
        """ fills in the hidden_states of the forward call.
            Args:
                forward_call (:obj:`bittensor.TextLastHiddenStateForwardCall`, `required`):
                    bittensor forward call dataclass to fill.
            Returns:
                forward_call (:obj:`bittensor.TextLastHiddenStateForwardCall`, `required`):
                    filled bittensor forward call dataclass.
        """
        forward_call.hidden_states = self.forward( 
            text_inputs = forward_call.text_inputs,
            hotkey = forward_call.hotkey
        )
        return forward_call
    
    def backward( 
            self, 
            text_inputs: torch.LongTensor,
            hidden_states: torch.FloatTensor,
            hidden_states_grads: torch.FloatTensor,
        ):
        """ Accepts the backward call and updates the model.
            Args:
                text_inputs (:obj:`torch.LongTensor`, `required`):
                    tokenized text inputs.
                hidden_states (:obj:`torch.FloatTensor`, `required`):
                    hidden states of the last layer of the model from forward call.
                hidden_states_grads (:obj:`torch.FloatTensor`, `required`):
                    hidden states gradients of the last layer of the model from backward call.
        """
        pass

    def _backward( self, backward_call: 'bittensor.TextLastHiddenStateBackwardCall' ):
        """ Accepts the backward call and updates the model.
            Args:
                backward_call (:obj:`bittensor.TextLastHiddenStateBackwardCall`, `required`):
                    bittensor backward call dataclass to fill.
        """
        self.backward( 
            text_inputs = backward_call.text_inputs, 
            hidden_states = backward_call.hidden_states, 
            hidden_states_grads = backward_call.hidden_states_grads 
        )

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
        # Call subclass priority, if not implemented use the 
        # metagraph priority based on stake.
        try:
            return float( self.priority( forward_call ) )
        except:
            if self.metagraph != None:
                uid = self.metagraph.hotkeys.index( forward_call.hotkey )
                return float( self.metagraph.S[uid].item() )
            else:
                return 0.0 
            
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
            instance_blacklist = self.blacklist( forward_call )
        except:
            instance_blacklist = False
        if self.metagraph == None: return instance_blacklist

        # Check for registration
        def registration_check():
            is_registered = forward_call.hotkey in self.metagraph.hotkeys
            if not is_registered:
                if self.config.text_last_hidden_state.blacklist.allow_non_registered:
                    return False
                raise Exception('Registration blacklist')

        # Blacklist based on stake.
        def stake_check() -> bool:
            uid = self.metagraph.hotkeys.index( forward_call.hotkey )
            if self.metagraph.S[uid].item() < self.config.text_last_hidden_state.blacklist.stake:
                raise Exception('Stake blacklist')
            return False

        # Optionally blacklist based on checks.
        try:
            registration_check()
            stake_check()            
            return instance_blacklist
        except Exception as e:
            return True
    
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

        forward_call =  bittensor.TextLastHiddenStateForwardCall(
            text_inputs = text_inputs,
            mask = mask,
            mask_serializer_type = request_proto.mask_serializer_type,
            text_inputs_serializer_type = request_proto.text_inputs_serializer_type,
            hidden_states_serializer_type = request_proto.hidden_states_serializer_type,
        )
        return forward_call

    
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
            serialized_hidden_states = serialized_hidden_states,
            return_code = forward_call.request_code,
            message = forward_call.request_message
        )
    
    def post_process_backward_call_to_response_proto( 
            self, 
            forward_call: 'bittensor.TextLastHiddenStateBackwardCall' 
        ) -> bittensor.BackwardTextLastHiddenStateResponse:
        """ post_process_forward_call_to_response_proto
            --------------------------------------------
            Args:
                backward_call (bittensor.TextLastHiddenStateBackwardCall):
                    backward call to post process into a response proto.
            Returns:    
                response (bittensor.BackwardTextLastHiddenStateResponse):
                    serialized backward call response.
        """
        # Return the forward response proto empty. (just a pong)
        return bittensor.BackwardTextLastHiddenStateResponse()
    
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
        return bittensor.TextLastHiddenStateBackwardCall(
            mask = mask,
            text_inputs = text_inputs,
            hidden_states = hidden_states,
            hidden_states_grads = hidden_states_grads,

            mask_serializer_type = request_proto.mask_serializer_type,
            text_inputs_serializer_type = request_proto.text_inputs_serializer_type,
            hidden_states_serializer_type = request_proto.hidden_states_serializer_type,
            hidden_states_grads_serializer_type = request_proto.hidden_states_grads_serializer_type,
        )
    
    def Forward( 
            self, 
            request: 'bittensor.ForwardRequest', 
            context: grpc.ServicerContext 
        ) -> 'bittensor.ForwardResponse':
        """ ForwardTextLastHiddenState
            ----------------------------
            Args:
                request (bittensor.ForwardRequest): 
                    request.version (int): version of the caller.
                    request.hotkey (string): hotkey of the neuron.
                    request.timeout (float): timeout for the request.
                context (grpc.ServicerContext):
                    grpc tcp context.
            Returns:
                response (bittensor.ForwardResponse): 
                    response.serialized_hidden_states (string): serialized hidden states.
        """
        try:
            # Build forward call.
            forward_call = self.pre_process_request_proto_to_forward_call( request_proto = request )
            forward_call.hotkey = request.hotkey
            forward_call.timeout = request.timeout
            forward_call.start_time = time.time()
            forward_call.version = request.version

            # Check blacklist.
            if self._blacklist( forward_call ): raise Exception('Blacklisted')
            # Get priority.
            priority = self._priority( forward_call )
            # Queue the forward call.
            future = self.priority_threadpool.submit(
                self._forward,
                forward_call = forward_call,
                priority = priority,
            )
        except Exception as e:
            forward_call.request_code = bittensor.proto.ReturnCode.UnknownException
            forward_call.request_message = str(e)

        finally:
            # Log request.
            bittensor.logging.rpc_log ( 
                axon = True, 
                forward = True, 
                is_response = False, 
                code = forward_call.request_code, 
                call_time = time.time() - forward_call.start_time, 
                pubkey = forward_call.hotkey, 
                uid = None, 
                inputs = forward_call.get_inputs_shape() if forward_call.request_code == bittensor.proto.ReturnCode.Success else None,
                outputs = None,
                message = forward_call.request_message,
                synapse = self.__str__()
            )
            if forward_call.request_code != bittensor.proto.ReturnCode.Success:
                response = self.post_process_forward_call_to_response_proto( forward_call )
                response.hotkey = self.wallet.hotkey.ss58_address
                response.version = bittensor.__version_as_int__
                return response

        # Do forward.
        try:
            # Get the result.
            future.result( timeout = forward_call.timeout )

        except Exception as e:
            forward_call.response_code = bittensor.proto.ReturnCode.UnknownException
            forward_call.response_message = str(e)
        finally:
            # Log response
            bittensor.logging.rpc_log ( 
                axon = True, 
                forward = True, 
                is_response = True, 
                code = forward_call.response_code, 
                call_time = time.time() - forward_call.start_time, 
                pubkey = forward_call.hotkey, 
                uid = None, 
                inputs = list( forward_call.get_inputs_shape() ) if forward_call.response_code == bittensor.proto.ReturnCode.Success else None,
                outputs = list( forward_call.get_outputs_shape() ) if forward_call.response_code == bittensor.proto.ReturnCode.Success else None,
                message = forward_call.response_message,
                synapse = self.__str__()
            )
            response = self.post_process_forward_call_to_response_proto( forward_call )
            response.hotkey = self.wallet.hotkey.ss58_address
            response.version = bittensor.__version_as_int__
            return response
        
    def Backward( 
            self, 
            request: 'bittensor.BackwardRequest', 
            context: grpc.ServicerContext 
        ) -> 'bittensor.ForwardResponse':
        """ ForwardTextLastHiddenState
            ----------------------------
            Args:
                request (bittensor.BackwardRequest): 
                    request.version (int): version of the caller.
                    request.hotkey (string): hotkey of the neuron.
                context (grpc.ServicerContext):
                    grpc tcp context.
            Returns:
                response (bittensor.BackwardResponse): 
                    response from the backward call.

        """
        try:
            # Build backward call.
            backward_call = self.pre_process_request_proto_to_backward_call( request_proto = request )
            backward_call.hotkey = request.hotkey
            backward_call.start_time = time.time()
            backward_call.version = request.version

            # Check blacklist.
            if self._blacklist( backward_call ): raise Exception('Blacklisted')
            # Get priority.
            priority = self._priority( backward_call )
            # Queue the backward call.
            future = self.priority_threadpool.submit(
                self._backward,
                backward_call = backward_call,
                priority = priority,
            )
        except Exception as e:
            backward_call.request_code = bittensor.proto.ReturnCode.UnknownException
            backward_call.request_message = str(e)
        finally:
            # Log request.
            bittensor.logging.rpc_log ( 
                axon = True, 
                forward = False, 
                is_response = False, 
                code = backward_call.request_code, 
                call_time = time.time() - backward_call.start_time, 
                pubkey = backward_call.hotkey, 
                uid = None, 
                inputs = backward_call.get_inputs_shape() if backward_call.request_code == bittensor.proto.ReturnCode.Success else None,
                outputs = None,
                message = backward_call.request_message,
                synapse = self.__str__()
            )
            if backward_call.request_code != bittensor.proto.ReturnCode.Success:
                response_proto = self.post_process_backward_call_to_response_proto( backward_call )
                response_proto.hotkey = self.wallet.hotkey.ss58_address
                response_proto.version = bittensor.__version_as_int__
                response_proto.return_code = backward_call.request_code
                response_proto.message = backward_call.request_message
                return response_proto

        # Do backward.
        try:
            # Get the result.
            future.result( timeout = bittensor.__blocktime__ )

        except Exception as e:
            backward_call.response_code = bittensor.proto.ReturnCode.UnknownException
            backward_call.response_message = str(e)
        finally:
            # Log response
            bittensor.logging.rpc_log ( 
                axon = True, 
                forward = False, 
                is_response = True, 
                code = backward_call.response_code, 
                call_time = time.time() - backward_call.start_time, 
                pubkey = backward_call.hotkey, 
                uid = None, 
                inputs = list( backward_call.get_inputs_shape() ) if backward_call.response_code == bittensor.proto.ReturnCode.Success else None,
                outputs = list( backward_call.get_outputs_shape() ) if backward_call.response_code == bittensor.proto.ReturnCode.Success else None,
                message = backward_call.response_message,
                synapse = self.__str__()
            )
            response_proto = self.post_process_backward_call_to_response_proto( backward_call )
            response_proto.hotkey = self.wallet.hotkey.ss58_address
            response_proto.version = bittensor.__version_as_int__
            response_proto.return_code = backward_call.request_code
            response_proto.message = backward_call.request_message
            return response_proto

