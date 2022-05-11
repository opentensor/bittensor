
from multiprocessing.sharedctypes import Value
from yaml import serialize_all
import bittensor
import torch
from typing import Union, List, Tuple, Optional

from bittensor._serializer import serializer


class Synapse:
    """ Proto serializable class which specifies the function to be called on a recieving neuron
        as well as the method of serialization and packing of forward/backward request/responses.
    """

    # Unique proto enum.
    synapse_type: bittensor.proto.Synapse.SynapseType = None

    def __init__(
        self, 
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ) -> 'Synapse':  
        """ Synapse super class initializer.
            Args:
                forward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                forward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward response.
                backward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                backward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serialzer used to pack torch tensors on backward response.
            Returns:
                Synapse (:obj:`Synapse`, `required`):
                    Synapse super class.
        """
        self.forward_request_serializer_type = forward_request_serializer_type
        self.forward_response_serializer_type = forward_response_serializer_type
        self.backward_request_serializer_type = backward_request_serializer_type
        self.backward_response_serializer_type = backward_response_serializer_type

    @staticmethod
    def deserialize_from_instance_proto ( isntance_proto: bittensor.proto.Synapse ) -> 'Synapse':
        """ Deserialzied the instance proto to an instance class.
            Args:
                isntance_proto (:obj:`bittensor.proto.Synapse` of shape :obj:`(1)`, `required`):
                    Synapse instance proto to be deserialized.
            Returns:
                synapse_instance_clasee (:obj:`torch.Tensor`, `required`):
                    Deserialized instance class.
        """
        raise NotImplementedError("deserialize_from_instance_proto should be implemented by the subclass.")

    @staticmethod
    def deserialize_from_wire_proto ( wire_proto: bittensor.proto.Synapse ) -> 'Synapse':
        """ Deserialzied the wire proto to an instance class.
            Args:
                wire_proto (:obj:`bittensor.proto.Synapse` of shape :obj:`(1)`, `required`):
                    Synapse wire proto to be deserialized.
            Returns:
                synapse_instance_clasee (:obj:`torch.Tensor`, `required`):
                    Deserialized instance class.
        """
        raise NotImplementedError("deserialize_from_wire_proto should be implemented by the subclass.")

    def serialize_to_instance_proto( self ) -> 'bittensor.proto.Synapse':
        """ Serializes the class instance to a Synapse instance proto.
            Returns:
                serialized_synapse_as_instance_proto (:obj:`torch.Tensor`, `required`):
                    Instance class serialized to a instance proto.
        """
        raise NotImplementedError("serialize_to_instance_proto should be implemented by the subclass.")

    def serialize_to_wire_proto( self ) -> 'bittensor.proto.Synapse':
        """ Serializes the class instance to a Synapse wire proto.
            Returns:
                serialized_synapse_as_wire_proto (:obj:`torch.Tensor`, `required`):
                    Instance class serialized to a wire proto.
        """
        raise NotImplementedError("serialize_to_wire_proto should be implemented by the subclass.")

    def nill_forward_response_tensor ( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite forward call when the call fails.
            Args:
                forward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Tensor being sent as forward request.
            Returns:
                nill_forward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Zeroed forward response tensor.
        """
        raise NotImplementedError("nill_forward_response_tensor should be implemented by the subclass.")

    def nill_backward_response_gradient ( self, forward_request_tensor : torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite backward call when the call fails.
            Args:
                forward_request_tensor  (:obj:`torch.Tensor`, `required`):
                    Tensor being sent as forward request.
            Returns:
                nill_backward_response_gradient  (:obj:`torch.Tensor`, `required`):
                    Zeroed backward response gradient.
        """
        raise NotImplementedError("nill_backward_response_tensor should be implemented by the subclass.")

    def check_forward_request_tensor     ( self, foward_request_tensor ): pass
    def check_forward_response_tensor    ( self, foward_request_tensor, forward_response_tensor ): pass
    def check_backward_request_gradient  ( self, foward_request_tensor, backward_request_gradient ): pass
    def check_backward_response_gradient  ( self, foward_request_tensor, backward_request_gradient ): pass
    def encode_forward_request_tensor    ( self, foward_request_tensor: torch.Tensor ) -> torch.Tensor: return foward_request_tensor
    def decode_forward_request_tensor    ( self, foward_request_tensor: torch.Tensor ) -> torch.Tensor: return foward_request_tensor
    def encode_forward_response_tensor   ( self, forward_response_tensor: torch.Tensor ) -> torch.Tensor: return forward_response_tensor
    def decode_forward_response_tensor   ( self, forward_response_tensor: torch.Tensor ) -> torch.Tensor: return forward_response_tensor
    def encode_backward_request_gradient ( self, backward_request_gradient: torch.Tensor ) -> torch.Tensor: return backward_request_gradient
    def decode_backward_request_gradient ( self, backward_request_gradient: torch.Tensor ) -> torch.Tensor: return backward_request_gradient
    def encode_backward_response_gradient ( self, backward_response_gradient: torch.Tensor ) -> torch.Tensor: return backward_response_gradient
    def decode_backward_response_gradient ( self, backward_response_gradient: torch.Tensor ) -> torch.Tensor: return backward_response_gradient

    def serialize_forward_request_tensor( self, forward_request_tensor: torch.Tensor ) -> Tuple[ 'bittensor.proto.Tensor', 'bittensor.proto.ReturnCode',  str ]:        
        self.check_forward_request_tensor ( forward_request_tensor )
        forward_request_tensor = self.encode_forward_request_tensor ( forward_request_tensor )
        tensor_serialzier = bittensor.serializer( serializer_type = self.forward_request_serializer_type )
        return tensor_serialzier.serialize( tensor_obj = forward_request_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def deserialize_forward_request_tensor( self, forward_request_proto: bittensor.proto.Tensor ) -> Tuple[ 'torch.Tensor', 'bittensor.proto.ReturnCode',  str ]:
        """ Returns a torch.Tensor from wire proto.Tensor after relevant deserialization has been applied. """
        tensor_deserialzier = bittensor.serializer( serializer_type = self.forward_request_serializer_type )
        forward_request_tensor = tensor_deserialzier.deserialize( tensor_obj = forward_request_proto, to_type = bittensor.proto.TensorType.TORCH )
        forward_request_tensor = self.decode_forward_request_tensor ( forward_request_tensor )
        self.check_forward_request_tensor ( forward_request_tensor )
        return forward_request_tensor

    def serialize_forward_response_tensor( self, foward_request_tensor: torch.Tensor, forward_response_tensor: torch.Tensor ) -> Tuple[ 'bittensor.proto.Tensor', 'bittensor.proto.ReturnCode',  str ]:
        """ Returns a bittensor.proto.Tensor to be sent on the wire after relevant serialization applied. """        
        self.check_forward_response_tensor ( foward_request_tensor, forward_response_tensor )
        encoded_tensor = self.encode_forward_response_tensor ( forward_response_tensor )
        tensor_serialzier = bittensor.serializer( serializer_type = self.forward_response_serializer_type )
        return tensor_serialzier.serialize( tensor_obj = encoded_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def deserialize_forward_response_proto( self, foward_request_tensor: torch.Tensor, forward_response_proto: bittensor.proto.Tensor ) -> Tuple[ 'torch.Tensor', 'bittensor.proto.ReturnCode',  str ]:
        """ Returns a torch.Tensor from wire proto.Tensor after relevant deserialization has been applied. """
        tensor_deserialzier = bittensor.serializer( serializer_type = self.forward_response_serializer_type )
        forward_response_tensor = tensor_deserialzier.deserialize( tensor_obj = forward_response_proto, to_type = bittensor.proto.TensorType.TORCH )
        forward_response_tensor = self.decode_forward_response_tensor ( forward_response_tensor )
        self.check_forward_response_tensor ( forward_response_tensor )
        return forward_response_tensor

    def serialize_backward_request_gradient( self, backward_request_gradient: torch.Tensor ) -> Tuple[ 'bittensor.proto.Tensor', 'bittensor.proto.ReturnCode',  str ]:
        """ Returns a bittensor.proto.Tensor gradient to be sent on the wire after relevant serialization applied. """
        self.check_backward_request_gradient ( backward_request_gradient )
        encoded_tensor = self.encode_backward_request_gradient ( backward_request_gradient )
        tensor_serialzier = bittensor.serializer( serializer_type = self.forward_request_serializer_type )
        return tensor_serialzier.serialize( tensor_obj = encoded_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def deserialize_backward_request_gradient( self, backward_request_proto: bittensor.proto.Tensor ) -> Tuple[ 'torch.Tensor', 'bittensor.proto.ReturnCode',  str ]:
        tensor_deserialzier = bittensor.serializer( serializer_type = self.backward_request_serializer_type )
        backward_request_gradient = tensor_deserialzier.deserialize( tensor_obj = backward_request_proto, to_type = bittensor.proto.TensorType.TORCH )
        backward_request_gradient = self.decode_backward_request_gradient ( backward_request_gradient )
        self.check_backward_request_gradient ( backward_request_gradient )
        return backward_request_gradient

class TextCausalLM (Synapse):
    """ CausalLM Synape type for training NTP    
    """

    synapse_type: bittensor.proto.Synapse.SynapseType = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM
    def __init__( 
        self, 
        topk: int = 512,
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ):  
        super().__init__ (
            forward_request_serializer_type,
            forward_response_serializer_type,
            backward_request_serializer_type,
            backward_response_serializer_type
        )
        self.topk = topk
        self.synapse_type = TextCausalLM.synapse_type

    @staticmethod
    def deserialize_from_instance_proto ( instance_proto: bittensor.proto.Synapse ) -> 'TextCausalLM':
        return TextCausalLM ( 
            topk = instance_proto.topk, 
            forward_request_serializer_type = instance_proto.forward_request_serializer_type,
            forward_response_serializer_type = instance_proto.forward_response_serializer_type,
            backward_request_serializer_type = instance_proto.backward_request_serializer_type,
            backward_response_serializer_type = instance_proto.backward_response_serializer_type,
        )

    @staticmethod
    def deserialize_from_wire_proto ( wire_proto: bittensor.proto.Synapse ) -> 'TextCausalLM':
        instance_proto = bittensor.proto.Synapse.TextCausalLM()
        instance_proto.ParseFromString( wire_proto.synapse_data )
        return TextCausalLM.deserialize_from_instance_proto( instance_proto )

    def serialize_to_instance_proto( self ) -> 'bittensor.proto.Synapse.TextCausalLM':
        return bittensor.proto.Synapse.TextCausalLM ( 
            topk = self.topk,
            forward_request_serializer_type = self.forward_request_serializer_type,
            forward_response_serializer_type = self.forward_response_serializer_type,
            backward_request_serializer_type = self.backward_request_serializer_type,
            backward_response_serializer_type = self.backward_response_serializer_type,
        )

    def serialize_to_wire_proto ( self, code: 'bittensor.proto.ReturnCode' = 0, message: str = '' ) -> bittensor.proto.Synapse:
        return bittensor.proto.Synapse (
                synapse_data = self.serialize_to_instance_proto().SerializeToString(),
                synapse_type = TextCausalLM.synapse_type,
                return_code = code,
                message = message
            )

    def check_forward_request_tensor     ( self, foward_request_tensor ): pass
    def check_forward_response_tensor    ( self, foward_request_tensor, forward_response_tensor ): pass
    def check_backward_request_gradient  ( self, foward_request_tensor, backward_request_gradient ): pass
    def encode_forward_request_tensor    ( self, foward_request_tensor: torch.Tensor ) -> torch.Tensor: return foward_request_tensor
    def decode_forward_request_tensor    ( self, foward_request_tensor: torch.Tensor ) -> torch.Tensor: return foward_request_tensor
    def encode_forward_response_tensor   ( self, forward_response_tensor: torch.Tensor ) -> torch.Tensor: return forward_response_tensor
    def decode_forward_response_tensor   ( self, forward_response_tensor: torch.Tensor ) -> torch.Tensor: return forward_response_tensor
    def encode_backward_request_gradient ( self, backward_request_gradient: torch.Tensor ) -> torch.Tensor: return backward_request_gradient
    def decode_backward_request_gradient ( self, backward_request_gradient: torch.Tensor ) -> torch.Tensor: return backward_request_gradient

    def nill_forward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), bittensor.__vocab_size__ ), dtype=torch.float32)

    def nill_backward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), forward_request_tensor.size(2) ), dtype=torch.float32)



class TextSeq2Seq (Synapse):
    """ Seq2Seq Synape type for generating text sequences. 
    """
    synapse_type: bittensor.proto.Synapse.SynapseType = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ
    def __init__( 
        self, 
        topk:int = 512, 
        num_to_generate: int = 512,
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ) -> 'TextSeq2Seq':  
        super().__init__ (
            forward_request_serializer_type,
            forward_response_serializer_type,
            backward_request_serializer_type,
            backward_response_serializer_type
        )
        self.topk = topk
        self.num_to_generate = num_to_generate
        self.synapse_type = TextSeq2Seq.synapse_type

    @staticmethod
    def deserialize_from_instance_proto ( instance_proto: bittensor.proto.Synapse ) -> 'Synapse':
        """ Deserialzied the instance proto to an instance class."""
        return TextSeq2Seq (
            topk = instance_proto.topk,
            num_to_generate = instance_proto.num_to_generate,
            forward_request_serializer_type = instance_proto.forward_request_serializer_type,
            forward_response_serializer_type = instance_proto.forward_response_serializer_type,
            backward_request_serializer_type = instance_proto.backward_request_serializer_type,
            backward_response_serializer_type = instance_proto.backward_response_serializer_type,
        )

    @staticmethod
    def deserialize_from_wire_proto ( wire_proto: bittensor.proto.Synapse ) -> 'Synapse':
        """ Deserialzied the wire proto to an instance class. """
        instance_proto = bittensor.proto.Synapse.TestSeq2Seq()
        instance_proto.ParseFromString( wire_proto.synapse_data )
        return TextSeq2Seq.deserialize_from_instance_proto( instance_proto )

    def serialize_to_instance_proto( self ) -> 'bittensor.proto.Synapse.TextSeq2Seq':
        """ Serializes the class instance to a Synapse instance proto."""
        return bittensor.proto.Synapse.TestSeq2Seq ( 
            topk = self.topk, 
            num_to_generate = self.num_to_generate,
            forward_request_serializer_type = self.forward_request_serializer_type,
            forward_response_serializer_type = self.forward_response_serializer_type,
            backward_request_serializer_type = self.backward_request_serializer_type,
            backward_response_serializer_type = self.backward_response_serializer_type,
        )

    def serialize_to_wire_proto( self, code: 'bittensor.proto.ReturnCode' = 0, message: str = ''  ) -> bittensor.proto.Synapse:
        """ Serializes the class instance to a Synapse wire proto. """
        return bittensor.proto.Synapse (
                synapse_data = self.serialize_to_instance_proto().SerializeToString(),
                synapse_type = TextSeq2Seq.synapse_type,
                return_code = code,
                message = message
            )

    def check_forward_request_tensor     ( self, foward_request_tensor ):
        if len( foward_request_tensor.shape ) != 2:
            raise ValueError( "foward_request_tensor.shape must be in [-1, -1], got: {} for synapse: {}".format( list(foward_request_tensor.shape), self ) ) 

    def check_forward_response_tensor    ( self, foward_request_tensor, forward_response_tensor ):
        if ( len( forward_response_tensor.shape ) != 3 or
             forward_response_tensor.size(0) != foward_request_tensor.size(0) or
             forward_response_tensor.size(1) != self.num_to_generate or
             forward_response_tensor.size(2) != self.topk
            ):
            raise ValueError( "forward_response_tensor.shape must be in [{}, {}, {}], got: {} for synapse: {}".format( foward_request_tensor.size(0) , self.num_to_generate, self.topk, list(forward_response_tensor.shape), self ) ) 

    def check_backward_request_gradient  ( self, foward_request_tensor, backward_request_gradient ):
        if list( backward_request_gradient.shape ) != list( foward_request_tensor.shape ):
            raise ValueError( "backward_request_gradient.shape: {} must be equivalent to foward_request_tensor.shape: {} for synapse: {}".format( list( backward_request_gradient.shape ), list(foward_request_tensor.shape), self ) ) 

    def encode_forward_request_tensor    ( self, foward_request_tensor: torch.Tensor ) -> torch.Tensor: 
        return foward_request_tensor
    def decode_forward_request_tensor    ( self, foward_request_tensor: torch.Tensor ) -> torch.Tensor: 
        return foward_request_tensor
    def encode_forward_response_tensor   ( self, forward_response_tensor: torch.Tensor ) -> torch.Tensor: 
        # Apply topk logit encoding.
        return forward_response_tensor
    def decode_forward_response_tensor   ( self, forward_response_tensor: torch.Tensor ) -> torch.Tensor: 
        # Decode topk logit encoding.
        return forward_response_tensor
    def encode_backward_request_gradient ( self, backward_request_gradient: torch.Tensor ) -> torch.Tensor: 
        # Apply topk logit encoding for gradients.
        return backward_request_gradient
    def decode_backward_request_gradient ( self, backward_request_gradient: torch.Tensor ) -> torch.Tensor:
        # Decode topk logit encoding for gradients.
        return backward_request_gradient

    def nill_forward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite forward call when the call fails."""
        return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), bittensor.__vocab_size__ ), dtype=torch.float32)

    def nill_backward_response_gradient( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite backward call when the call fails."""
        return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), forward_request_tensor.size(2) ), dtype=torch.float32)


class TextLastHiddenState (Synapse):
    """ TastHiddenState Synapse type for getting last hidden layer embeddings from languge models.
    """

    synapse_type: bittensor.proto.Synapse.SynapseType = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE
    def __init__( 
        self,
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ) -> 'TextLastHiddenState':
        """ TextLastHiddenState Synapse initializer.
            Args:
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
        self.synapse_type = TextLastHiddenState.synapse_type

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
            forward_request_serializer_type = instance_proto.forward_request_serializer_type,
            forward_response_serializer_type = instance_proto.forward_response_serializer_type,
            backward_request_serializer_type = instance_proto.backward_request_serializer_type,
            backward_response_serializer_type = instance_proto.backward_response_serializer_type,
        )

    def serialize_to_instance_proto( self ) -> 'bittensor.proto.Synapse.TextLastHiddenState':
        """ Serializes the class instance to a Synapse instance proto.
        """
        return bittensor.proto.Synapse.TextLastHiddenState ( 
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

    def check_forward_request_tensor     ( self, foward_request_tensor ): pass
    def check_forward_response_tensor    ( self, foward_request_tensor, forward_response_tensor ): pass
    def check_backward_request_gradient  ( self, foward_request_tensor, backward_request_gradient ): pass
    def encode_forward_request_tensor    ( self, foward_request_tensor: torch.Tensor ) -> torch.Tensor: return foward_request_tensor
    def decode_forward_request_tensor    ( self, foward_request_tensor: torch.Tensor ) -> torch.Tensor: return foward_request_tensor
    def encode_forward_response_tensor   ( self, forward_response_tensor: torch.Tensor ) -> torch.Tensor: return forward_response_tensor
    def decode_forward_response_tensor   ( self, forward_response_tensor: torch.Tensor ) -> torch.Tensor: return forward_response_tensor
    def encode_backward_request_gradient ( self, backward_request_gradient: torch.Tensor ) -> torch.Tensor: return backward_request_gradient
    def decode_backward_request_gradient ( self, backward_request_gradient: torch.Tensor ) -> torch.Tensor: return backward_request_gradient


    def nill_forward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite forward call when the call fails.
        """
        return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), bittensor.__network_dim__ ), dtype=torch.float32)

    def nill_backward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite backward call when the call fails.
        """
        return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), forward_request_tensor.size(2) ), dtype=torch.float32)

    

class synapse:
    
    @staticmethod
    def TextLastHiddenState (
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ) -> TextLastHiddenState:
        """ Factory function which returns a TextLastHiddenState synapse adapter given arguments.
            Args:
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
        return TextLastHiddenState (
            forward_request_serializer_type = forward_request_serializer_type,
            forward_response_serializer_type = forward_response_serializer_type,
            backward_request_serializer_type = backward_request_serializer_type,
            backward_response_serializer_type = backward_response_serializer_type,
        )

    @staticmethod
    def TextCausalLM ( 
        topk:int = 512,
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ) -> TextCausalLM:
        """ Factory function which returns a TextCausalLM synapse adapter given arguments.
            Args:
                forward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                forward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward response.
                backward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                backward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serialzer used to pack torch tensors on backward response.
            Returns:
                TextCausalLM (:obj:`TextCausalLM`, `required`):
                    TextCausalLM instance adapter class.
        """
        return TextCausalLM ( 
            topk = topk,
            forward_request_serializer_type = forward_request_serializer_type,
            forward_response_serializer_type = forward_response_serializer_type,
            backward_request_serializer_type = backward_request_serializer_type,
            backward_response_serializer_type = backward_response_serializer_type,
        )

    @staticmethod
    def TextSeq2Seq ( 
        topk:int = 512, 
        num_to_generate: int = 512,
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ) -> TextSeq2Seq:
        """ Factory function which returns a TextSeq2Seq synapse adapter given arguments.
            Args:
                topk (:obj:`int` of shape :obj:`(1)`, `optional`, :default: `512`):
                    Number of topk logits to return per item in the sequence.
                num_to_generate (:obj:`int` of shape :obj:`(1)`, `optional`, :default: `512`):
                    Number of topk logits to generate per example in the batch.
                forward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                forward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward response.
                backward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                backward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serialzer used to pack torch tensors on backward response.
            Returns:
                TextSeq2Seq (:obj:`TextSeq2Seq`, `required`):
                    TextSeq2Seq instance adapter class.
        """
        return TextSeq2Seq ( 
            topk = topk, 
            num_to_generate = num_to_generate,
            forward_request_serializer_type = forward_request_serializer_type,
            forward_response_serializer_type = forward_response_serializer_type,
            backward_request_serializer_type = backward_request_serializer_type,
            backward_response_serializer_type = backward_response_serializer_type,
        )

    @staticmethod
    def deserialize( synapse_wire_proto: bittensor.proto.Synapse ) -> Synapse:
        if synapse_wire_proto.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE:
            return TextLastHiddenState.deserialize_from_wire_proto ( synapse_wire_proto )
        elif synapse_wire_proto.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM:
            return TextCausalLM.deserialize_from_wire_proto( synapse_wire_proto )
        elif synapse_wire_proto.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ:
            return TextSeq2Seq.deserialize_from_wire_proto( synapse_wire_proto )
        else:
            raise ValueError("Synapse type is unknown.")