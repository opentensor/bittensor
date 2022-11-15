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

import bittensor
import torch
from typing import Union, List, Tuple, Optional

from .synapse_impl import Synapse

class TextLastHiddenState (Synapse):
    """ TastHiddenState Synapse type for getting last hidden layer embeddings from languge models.
    """
    synapse_type: bittensor.proto.Synapse.SynapseType = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE

    def interpret_mask_for_request_tensor( mask: Union[ int, List[int] ], request_tensor ) -> List[int]:
        bs, _, _ = request_tensor.shape

        # Converts an integer mask to a real index.
        def translate_int_mask( int_mask:int ) -> int:
            if int_mask < -bs or int_mask > bs:
                raise ValueError("Mask element {} cannot be interpreted for tensor of shape {}".format(int_mask, request_tensor.shape))
            # Translate mask to real index -1 --> (bs-1)
            translated_mask = list( range( bs ) )[ mask ]
            return translated_mask

        # Interpret mask from solo int.
        if isinstance( mask, int ):
            int_mask = translate_int_mask(mask)
            # Interprets the mask as refering the to an index from each example
            # in the batch.
            return [ int_mask * (i+1) for i in range(bs) ]

        # Interpret mask as list of ints
        elif isinstance( mask, list ):
            # Translate mask to real index -1 --> (bs-1)
            translated_masks = [ translate_int_mask(mi) for mi in mask ]
            shifted_mask = []
            for bi in range(bs):
                for mi in translated_masks:
                    shifted_mask.append( mi * (bi + 1) )
            return shifted_mask

        # Interpret mask as a double list of ints
        else:
            raise ValueError("Mask element {} must be of type Union[ int, List[int] ], got {}".format(mask, type(mask) ))


    def __init__( 
        self,
        mask: List[int] = [],
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ) -> 'TextLastHiddenState':
        """ TextLastHiddenState Synapse initializer.
            Args:
                mask (:obj:`List[int]` of shape :obj:`(n)`, `optional`, :default: `[]`):
                    An optional response mask over the returned embeddings.               
                forward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                forward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward response.
                backward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                backward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serialzer used to pack torch tensors on backward response.
            Returns:
                TextLastHiddenState (:obj:`TextLastHiddenState`, `required`):
                    TextLastHiddenState instance adapter class.
        """
        super().__init__ (
            forward_request_serializer_type,
            forward_response_serializer_type,
            backward_request_serializer_type,
            backward_response_serializer_type
        )
        self.mask = mask
        self.synapse_type = TextLastHiddenState.synapse_type

    def __repr__(self) -> str: return self.__str__()
    def __str__(self) -> str: return "TextLastHiddenState"

    @staticmethod
    def deserialize_from_wire_proto ( wire_proto: bittensor.proto.Synapse ) -> 'Synapse':
        """ Deserialzied the wire proto to an instance class.
        """
        instance_proto = bittensor.proto.Synapse.TextLastHiddenState()
        instance_proto.ParseFromString( wire_proto.synapse_data )
        return TextLastHiddenState.deserialize_from_instance_proto( instance_proto )

    @staticmethod
    def deserialize_from_instance_proto ( instance_proto: bittensor.proto.Synapse ) -> 'Synapse':
        """ Deserialzied the instance proto to an instance class.
            Args:
                isntance_proto (:obj:`bittensor.proto.Synapse` of shape :obj:`(1)`, `required`):
                    Synapse instance proto to be deserialized.
            Returns:
                synapse_instance_clasee (:obj:`torch.Tensor`, `required`):
                    Deserialized instance class.
        """
        return TextLastHiddenState (
            mask = instance_proto.forward_request_serializer_type,
            forward_request_serializer_type = instance_proto.forward_request_serializer_type,
            forward_response_serializer_type = instance_proto.forward_response_serializer_type,
            backward_request_serializer_type = instance_proto.backward_request_serializer_type,
            backward_response_serializer_type = instance_proto.backward_response_serializer_type,
        )

    def serialize_to_instance_proto( self ) -> 'bittensor.proto.Synapse.TextLastHiddenState':
        """ Serializes the class instance to a Synapse instance proto.
        """
        return bittensor.proto.Synapse.TextLastHiddenState ( 
            mask = self.mask,
            forward_request_serializer_type = self.forward_request_serializer_type,
            forward_response_serializer_type = self.forward_response_serializer_type,
            backward_request_serializer_type = self.backward_request_serializer_type,
            backward_response_serializer_type = self.backward_response_serializer_type,
        )

    def serialize_to_wire_proto( self, code: 'bittensor.proto.ReturnCode' = 0, message: str = '' ) -> 'bittensor.proto.Synapse':
        """ Serializes the class instance to a Synapse wire proto.
        """
        return bittensor.proto.Synapse (
                synapse_data = self.serialize_to_instance_proto().SerializeToString(),
                synapse_type = TextLastHiddenState.synapse_type,
                return_code = code,
                message = message
            )

    def check_forward_request_tensor     ( self, forward_request_tensor ): 
        if len( forward_request_tensor.shape ) != 2 or forward_request_tensor.shape[0] == 0 or forward_request_tensor.shape[1] == 0:
            raise ValueError( "forward_request_tensor.shape must be in [-1, -1], got: {} for synapse: {}".format( list(forward_request_tensor.shape), self ) ) 

    def check_forward_response_tensor    ( self, forward_request_tensor, forward_response_tensor ):
        if forward_response_tensor == None:
            raise ValueError('Empty Response')
            
        if ( 
             len( forward_response_tensor.shape ) != 3 or
             forward_response_tensor.size(0) != forward_request_tensor.size(0) or
             forward_response_tensor.size(1) != forward_request_tensor.size(1) - len(self.mask) or
             forward_response_tensor.size(2) != bittensor.__network_dim__
            ):
            raise ValueError( "forward_response_tensor.shape must be in [{}, {}, {}], got: {} for synapse: {}".format( forward_request_tensor.size(0) , forward_request_tensor.size(1), bittensor.__network_dim__, list(forward_response_tensor.shape), self ) ) 
   
    def check_backward_request_gradient  ( self, forward_request_tensor, backward_request_gradient ): 
        if ( 
             len( backward_request_gradient.shape ) != 3 or
             backward_request_gradient.size(0) != forward_request_tensor.size(0) or
             backward_request_gradient.size(1) != forward_request_tensor.size(1) - len(self.mask) or
             backward_request_gradient.size(2) != bittensor.__network_dim__
            ):
            raise ValueError( "backward_request_gradient.shape must be in [{}, {}, {}], got: {} for synapse: {}".format( forward_request_tensor.size(0) , forward_request_tensor.size(1), bittensor.__network_dim__, list(backward_request_gradient.shape), self ) ) 

    def encode_forward_request_tensor    ( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor: return forward_request_tensor
    def decode_forward_request_tensor    ( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor: return forward_request_tensor
    def encode_forward_response_tensor   ( self, forward_response_tensor: torch.Tensor ) -> torch.Tensor: 
        # Reshape the forward_response_tensor to a stack of representations
        # stacked_forward_response_tensor [ bs * seq, net_dim ]
        stacked_forward_response_tensor = forward_response_tensor.reshape( -1, bittensor.__network_dim__ )
        
        # The self.mask is a list of indices which refer to distinct rows in the  
        # stacked stacked_forward_response_tensor [ bs * seq, net_dim ]. We pull only these 
        # representations for the encoding so the respons has shape [ len(mask), net_dim ]
        return stacked_forward_response_tensor[ self.mask, : ]

    def decode_forward_response_tensor   ( self, forward_request_tensor: torch.Tensor, forward_response_tensor: torch.Tensor ) -> torch.Tensor: 
        # From the encode_forward_response function the forward_response_tensor is [ len(mask), net_dim ]
        # a set of rows from the stacked_forward_response_tensor = [ bs * seq, net_dim ]
        # We will load these rows into a destination tensor = [bs, seq, net_dim]
        destination = torch.zeros( [ forward_request_tensor.size(0) * forward_request_tensor.size(1), bittensor.__network_dim__ ])

        # Iterate through the mask and the rows of the forward_response_tensor
        # replacing each row in the destination with the row from the response_tensor.
        for i, j in list(zip(self.mask, len(self.mask))):
            destination[i,:] = forward_response_tensor[j,:]
        
        # Reshape the destination tensor to the proper expanded size.
        destination.reshape( (forward_request_tensor.size(0), forward_request_tensor.size(1), bittensor.__network_dim__) )

        # Destination has shape [ bs, seq, net_dim ]
        return destination

    def encode_backward_request_gradient ( self, backward_request_gradient: torch.Tensor ) -> torch.Tensor: return backward_request_gradient
    def decode_backward_request_gradient ( self, backward_request_gradient: torch.Tensor ) -> torch.Tensor: return backward_request_gradient


    def nill_forward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite forward call when the call fails.
        """
        try:
            return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), bittensor.__network_dim__ ), dtype=torch.float32)
        except:
            return torch.tensor([])
    def nill_backward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite backward call when the call fails.
        """
        try:
            return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), bittensor.__network_dim__ ), dtype=torch.float32)
        except:
            return torch.tensor([])
    