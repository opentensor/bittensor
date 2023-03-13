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

import time
import torch
import bittensor

class Synapse( bittensor.grpc.BittensorServicer ):

    def __init__( self ):
        r""" Initializes a new Synapse."""
        self.priority_threadpool = bittensor.prioritythreadpool()

    def __str__(self):
        return "synapse"
    
    def _attach( self, axon: 'bittensor.axon.Axon' ):
        """ _attach: Attaches the synapse to the axon."""
        bittensor.grpc.add_BittensorServicer_to_server( self, axon.server )

    def priority( self, forward_call: call.ForwardCall ) -> float:
        raise NotImplementedError('Must implement priority() in subclass.')

    def blacklist( self, forward_call: call.ForwardCall ) -> torch.FloatTensor:
        raise NotImplementedError('Must implement blacklist() in subclass.')

    def forward(self, forward_call: call.ForwardCall ):
        raise NotImplementedError('Must implement forward() in subclass.')
    
    def pre_process_request_proto_to_forward_call( 
            self, 
            request_proto: 'bittensor.ForwardRequest' 
        ) -> 'bittensor.ForwardCall':
        """ pre_process_request_proto_to_forward_call
            ------------------------------------------
            Args:
                request_proto (bittensor.ForwardRequest):
                    request_proto to process in to a forward call.
            Returns:
                bittensor.ForwardCall (:obj:`bittensor.ForwardCall`, `required`):
                    forward call processed from the request proto.
            """
        raise NotImplementedError('Must implement pre_process_request_proto_to_forward_call() in subclass.')
    
    def post_process_forward_call_to_response_proto( 
            self, 
            forward_call: 'bittensor.ForwardCall' 
        ) -> 'bittensor.ForwardResponse':
        """ post_process_forward_call_to_response_proto
            --------------------------------------------
            Args:
                forward_call (bittensor.ForwardCall):
                    forward_call to process in to a response proto.
            Returns:    
                response (bittensor.ForwardResponse):
                    response proto processed from the forward call.
        """
        raise NotImplementedError('Must implement post_process_forward_call_to_response_proto() in subclass.')
        
    def _Forward( self, request_proto: 'bittensor.ForwardRequest' ) -> 'call.ForwardCall':
        forward_call = self.pre_process_request_proto_to_forward_call( request_proto = request_proto )
        try:
            # Check blacklist.
            if self.blacklist( forward_call ): raise Exception('Blacklisted')
            # Get priority.
            priority = self.priority( forward_call )
            # Queue the forward call.
            future = self.priority_threadpool.submit(
                self.forward,
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
                return self.post_process_forward_call_to_response_proto( forward_call )

        # Do forward.
        try:
            # Get the result.
            future.result( timeout = forward_call.timeout )

        except Exception as e:
            print ('failed forward')
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
            return self.post_process_forward_call_to_response_proto( forward_call )
