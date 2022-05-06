
from yaml import serialize_all
import bittensor
import torch
from typing import Union, List, Tuple, Optional


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

    def nill_forward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite forward call when the call fails.
            Args:
                forward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Tensor being sent as forward request.
            Returns:
                nill_forward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Zeroed forward response tensor.
        """
        raise NotImplementedError("nill_forward_response_tensor should be implemented by the subclass.")

    def nill_backward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite backward call when the call fails.
            Args:
                forward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Tensor being sent as forward request.
            Returns:
                nill_backward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Zeroed backward response tensor.
        """
        raise NotImplementedError("nill_backward_response_tensor should be implemented by the subclass.")

    def check_forward_request( self, foward_request_tensor ) -> Tuple[ bool, bittensor.proto.ReturnCode,  str ]:
        """ Checks that the forward request tensor being sent by the dendrite is well formed.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Forward input tensor to be sent on the wire.
            Returns:
                is_success (:obj:`bool`, `required`):
                    Did the forward_response_tensor meet requirements.
                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Check return code. Success of ResponseShapeException.
                message (:obj:`str`, `required`):
                    Message associated with check.
        """
        raise NotImplementedError("check_forward_request_shape should be implemented by the subclass.")

    def check_forward_response( self, foward_request_tensor, forward_response_tensor ) -> Tuple[ bool, bittensor.proto.ReturnCode,  str ]:
        """ Checks that the forward response tensor being sent by the axon is well formed.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Forward inputs sent on the wire.
                forward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Forward outputs received on the wire.
            Returns:
                is_success (:obj:`bool`, `required`):
                    Did the forward_response_tensor meet requirements.
                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Check return code. Success of ResponseShapeException.
                message (:obj:`str`, `required`):
                    Message associated with check.
        """
        raise NotImplementedError("check_forward_response_shape should be implemented by the subclass.")

    def serialize_forward_request_tensor( self, foward_request_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the dendrite side to serialize the synapse inputs.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_foward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.forward_request_serializer_type )
        return serializer.serialize( foward_request_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def serialize_forward_response_tensor( self, foward_response_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the axon side to serialize the synapse outputs.
            Args:
                foward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_foward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.forward_response_serializer_type )
        return serializer.serialize( foward_response_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def deserialize_backward_request_tensor( self, backward_request_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the dendrite side to serialize the synapse gradients.
            Args:
                backward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_backward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.backward_request_serializer_type )
        return serializer.serialize( backward_request_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def serialize_backward_response_tensor( self, backward_response_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the axon side to serialize the synapse output gradients.
            Args:
                backward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_backward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.backward_response_serializer_type )
        return serializer.serialize( backward_response_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def encode_forward_request_tensor (self, foward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the dendrite side before sending it over the wire. 
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Single torch tensor which should be encoded by the synapse before sending 
                    over the wire.
            Returns:
                encoded_foward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Encoded forward request tensor in format ready to be sent over the wire.
        """
        raise NotImplementedError("encode_forward_request_tensor should be implemented by the subclass.")

    def decode_forward_request_tensor (self, foward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the axon side before sending the tensor to the synapse function.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Torch tensor which should should be decoded via the inverse of the function 
                    of encode_forward_request_tensor.
            Returns:
                decoded_foward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Synapse decoded forward request tensor.

        """
        raise NotImplementedError("decode_forward_request_tensor should be implemented by the subclass.")

    def encode_forward_response_tensor (self, forward_response_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the axon side before sending the response over the wire. 
            Args:
                forward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be encoded before sending over the wire to the dendrite.
            Returns:
                encoded_forward_response_tensor(:obj:`torch.Tensor`, `required`):
                    Synapse encoded forward response tensor.
        """
        raise NotImplementedError("encode_forward_response_tensor should be implemented by the subclass.")

    def decode_forward_response_tensor (self, output_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the dendrite to decode the tensor recieved from the axon on the wire.
            Args:
                forward_response_tensor  (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be decoded via the inverse of the function encode forward response 
                    tensor.

             Returns:
                decoded_forward_response_tensor  (:obj:`torch.Tensor`, `required`):
                    Synapse decoded forward response tensor.
        """
        raise NotImplementedError("decode_forward_response_tensor should be implemented by the subclass.")


class TextCasualLM (Synapse):
    """ CasualLM Synape type for training NTP    
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
        """ TextCasualLM Synapse initializer.
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
                TextCasualLM (:obj:`TextCasualLM`, `required`):
                    TextCasualLM instance adapter class.
        """
        super(TextCasualLM).__init__ (
            forward_request_serializer_type,
            forward_response_serializer_type,
            backward_request_serializer_type,
            backward_response_serializer_type
        )
        self.topk = topk

    @staticmethod
    def deserialize_from_instance_proto ( instance_proto: bittensor.proto.Synapse ) -> 'TextCasualLM':
        """ Deserialzied the instance proto to an instance class.
            Args:
                instance_proto (:obj:`bittensor.proto.Synapse` of shape :obj:`(1)`, `required`):
                    Synapse instance proto to be deserialized.
            Returns:
                synapse_instance_clasee (:obj:`torch.Tensor`, `required`):
                    Deserialized instance class.
        """
        return TextCasualLM ( 
            topk = instance_proto.topk, 
            forward_request_serializer_type = instance_proto.forward_request_serializer_type,
            forward_response_serializer_type = instance_proto.forward_response_serializer_type,
            backward_request_serializer_type = instance_proto.backward_request_serializer_type,
            backward_response_serializer_type = instance_proto.backward_response_serializer_type,
        )

    @staticmethod
    def deserialize_from_wire_proto ( wire_proto: bittensor.proto.Synapse ) -> 'TextCasualLM':
        """ Deserialzied the wire proto to an instance class.
            Args:
                wire_proto (:obj:`bittensor.proto.Synapse` of shape :obj:`(1)`, `required`):
                    Synapse wire proto to be deserialized.
            Returns:
                synapse_instance_clasee (:obj:`torch.Tensor`, `required`):
                    Deserialized instance class.
        """
        instance_proto = bittensor.proto.Synapse.TextCasualLM.ParseFromString( wire_proto.synapse_data )
        return TextCasualLM.deserialize_from_instance_proto( instance_proto )

    def serialize_to_instance_proto( self ) -> 'bittensor.proto.Synapse.TextCasualLM':
        """ Serializes the class instance to a Synapse instance proto.
            Returns:
                serialized_synapse_as_instance_proto (:obj:`torch.Tensor`, `required`):
                    Instance class serialized to a instance proto.
        """
        return bittensor.proto.Synapse.TextCausalLM ( 
            topk = self.topk,
            forward_request_serializer_type = self.forward_request_serializer_type,
            forward_response_serializer_type = self.forward_response_serializer_type,
            backward_request_serializer_type = self.backward_request_serializer_type,
            backward_response_serializer_type = self.backward_response_serializer_type,
        )

    def serialize_to_wire_proto( self ) -> 'bittensor.proto.Synapse':
        """ Serializes the class instance to a Synapse wire proto.
            Returns:
                serialized_synapse_as_wire_proto (:obj:`torch.Tensor`, `required`):
                    Instance class serialized to a wire proto.
        """
        return bittensor.proto.Synapse (
                synapse_data = self.to_proto().SerializeToString(),
                synapse_type = TextCasualLM.synapse_type,
            )

    def nill_forward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite forward call when the call fails.
            Args:
                forward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Tensor being sent as forward request.
            Returns:
                nill_forward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Zeroed forward response tensor.
        """
        return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), bittensor.__vocab_size__ ), dtype=torch.float32)

    def nill_backward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite backward call when the call fails.
            Args:
                forward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Tensor being sent as forward request.
            Returns:
                nill_backward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Zeroed backward response tensor.
        """
        return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), forward_request_tensor.size(2) ), dtype=torch.float32)

    def check_forward_request ( self, foward_request_tensor ) -> Tuple[ bool, bittensor.proto.ReturnCode,  str ]:
        """ Checks that the forward request tensor being sent by the dendrite is well formed.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Forward input tensor to be sent on the wire.
            Returns:
                is_success (:obj:`bool`, `required`):
                    Did the forward_response_tensor meet requirements.
                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Check return code. Success of ResponseShapeException.
                message (:obj:`str`, `required`):
                    Message associated with check.
        """
        if  ( len(foward_request_tensor.shape) != 2 ):
            return False, bittensor.proto.ReturnCode.RequestShapeException, "foward_request_tensor.shape:{} is not correct for synapse: {}".format( foward_request_tensor.shape, self )
        else:
            return True, bittensor.proto.ReturnCode.Success, "Success"

    def check_forward_response_shape( self, foward_request_tensor, forward_response_tensor ) -> Tuple[ bool, bittensor.proto.ReturnCode,  str ]:
        """ Checks that the forward response tensor being sent by the axon is well formed.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Forward inputs sent on the wire.
                forward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Forward outputs received on the wire.
            Returns:
                is_success (:obj:`bool`, `required`):
                    Did the forward_response_tensor meet requirements.
                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Check return code. Success of ResponseShapeException.
                message (:obj:`str`, `required`):
                    Message associated with check.
        """
        if  ( 
                forward_response_tensor.size(0) != foward_request_tensor.size(0) or 
                forward_response_tensor.size(1) != foward_request_tensor.size(1) or 
                forward_response_tensor.size(2) != self.topk
            ):
            return False, bittensor.proto.ReturnCode.ResponseShapeException, "output.shape:{} does not match inputs:{} for synapse: {}".format( forward_response_tensor.shape, foward_request_tensor.shape, self )
        else:
            return True, bittensor.proto.ReturnCode.Success, "Success"

    def serialize_forward_request_tensor( self, foward_request_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the dendrite side to serialize the synapse inputs.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_foward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.forward_request_serializer_type )
        return serializer.serialize( foward_request_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def serialize_forward_response_tensor( self, foward_response_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the axon side to serialize the synapse outputs.
            Args:
                foward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_foward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.forward_response_serializer_type )
        return serializer.serialize( foward_response_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def deserialize_backward_request_tensor( self, backward_request_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the dendrite side to serialize the synapse gradients.
            Args:
                backward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_backward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.backward_request_serializer_type )
        return serializer.serialize( backward_request_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def serialize_backward_response_tensor( self, backward_response_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the axon side to serialize the synapse output gradients.
            Args:
                backward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_backward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.backward_response_serializer_type )
        return serializer.serialize( backward_response_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def encode_forward_request_tensor (self, foward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the dendrite side before sending it over the wire. 
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Single torch tensor which should be encoded by the synapse before sending 
                    over the wire.
            Returns:
                encoded_foward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Encoded forward request tensor in format ready to be sent over the wire.
        """
        return torch.where( torch.isnan(foward_request_tensor), torch.zeros_like(foward_request_tensor), foward_request_tensor).detach() 

    def decode_forward_request_tensor (self, foward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the axon side before sending the tensor to the synapse function.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Torch tensor which should should be decoded via the inverse of the function 
                    of encode_forward_request_tensor.
            Returns:
                decoded_foward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Synapse decoded forward request tensor.

        """
        return torch.where( torch.isnan(foward_request_tensor), torch.zeros_like(foward_request_tensor), foward_request_tensor).detach() 

    def encode_forward_response_tensor (self, forward_response_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the axon side before sending the response over the wire. 
            Args:
                forward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be encoded before sending over the wire to the dendrite.
            Returns:
                encoded_forward_response_tensor(:obj:`torch.Tensor`, `required`):
                    Synapse encoded forward response tensor.
        """
        return torch.where( torch.isnan(forward_response_tensor), torch.zeros_like(forward_response_tensor), forward_response_tensor).detach() 

    def decode_forward_response_tensor (self, forward_response_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the dendrite to decode the tensor recieved from the axon on the wire.
            Args:
                forward_response_tensor  (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be decoded via the inverse of the function encode forward response 
                    tensor.

             Returns:
                decoded_forward_response_tensor  (:obj:`torch.Tensor`, `required`):
                    Synapse decoded forward response tensor.
        """
        return torch.where( torch.isnan(forward_response_tensor), torch.zeros_like(forward_response_tensor), forward_response_tensor).detach() 


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
        """ TextSeq2Seq Synapse initializer.
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
        super(TextSeq2Seq).__init__ (
            forward_request_serializer_type,
            forward_response_serializer_type,
            backward_request_serializer_type,
            backward_response_serializer_type
        )
        self.topk = topk
        self.num_to_generate = num_to_generate

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
        """ Deserialzied the wire proto to an instance class.
            Args:
                wire_proto (:obj:`bittensor.proto.Synapse` of shape :obj:`(1)`, `required`):
                    Synapse wire proto to be deserialized.
            Returns:
                synapse_instance_clasee (:obj:`torch.Tensor`, `required`):
                    Deserialized instance class.
        """
        instance_proto = bittensor.proto.Synapse.TestSeq2Seq.ParseFromString( wire_proto.synapse_data )
        return TextSeq2Seq.deserialize_from_instance_proto( instance_proto )

    def serialize_to_instance_proto( self ) -> 'bittensor.proto.Synapse.TextSeq2Seq':
        """ Serializes the class instance to a Synapse instance proto.
            Returns:
                serialized_synapse_as_instance_proto (:obj:`torch.Tensor`, `required`):
                    Instance class serialized to a instance proto.
        """
        return bittensor.proto.Synapse.TestSeq2Seq ( 
            topk = self.topk, 
            num_to_generate = self.num_to_generate,
            forward_request_serializer_type = self.forward_request_serializer_type,
            forward_response_serializer_type = self.forward_response_serializer_type,
            backward_request_serializer_type = self.backward_request_serializer_type,
            backward_response_serializer_type = self.backward_response_serializer_type,
        )

    def serialize_to_wire_proto( self ) -> 'bittensor.proto.Synapse':
        """ Serializes the class instance to a Synapse wire proto.
            Returns:
                serialized_synapse_as_wire_proto (:obj:`torch.Tensor`, `required`):
                    Instance class serialized to a wire proto.
        """
        return bittensor.proto.Synapse (
                synapse_data = self.serialize_to_instance_proto().SerializeToString(),
                synapse_type = TextSeq2Seq.synapse_type,
            )

    def nill_forward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite forward call when the call fails.
            Args:
                forward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Tensor being sent as forward request.
            Returns:
                nill_forward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Zeroed forward response tensor.
        """
        return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), bittensor.__vocab_size__ ), dtype=torch.float32)

    def nill_backward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite backward call when the call fails.
            Args:
                forward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Tensor being sent as forward request.
            Returns:
                nill_backward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Zeroed backward response tensor.
        """
        return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), forward_request_tensor.size(2) ), dtype=torch.float32)

    def check_forward_request ( self, foward_request_tensor ) -> Tuple[ bool, bittensor.proto.ReturnCode,  str ]:
        """ Checks that the forward request tensor being sent by the dendrite is well formed.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Forward input tensor to be sent on the wire.
            Returns:
                is_success (:obj:`bool`, `required`):
                    Did the forward_response_tensor meet requirements.
                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Check return code. Success of ResponseShapeException.
                message (:obj:`str`, `required`):
                    Message associated with check.
        """
        if  ( len(foward_request_tensor.shape) != 2 ):
            return False, bittensor.proto.ReturnCode.RequestShapeException, "foward_request_tensor.shape:{} is not correct for synapse: {}".format( foward_request_tensor.shape, self )
        else:
            return True, bittensor.proto.ReturnCode.Success, "Success"

    def check_forward_response_shape( self, foward_request_tensor, forward_response_tensor ) -> Tuple[ bool, bittensor.proto.ReturnCode,  str ]:
        """ Checks that the forward response tensor being sent by the axon is well formed.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Forward inputs sent on the wire.
                forward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Forward outputs received on the wire.
            Returns:
                is_success (:obj:`bool`, `required`):
                    Did the forward_response_tensor meet requirements.
                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Check return code. Success of ResponseShapeException.
                message (:obj:`str`, `required`):
                    Message associated with check.
        """
        if  ( 
                forward_response_tensor.size(0) != foward_request_tensor.size(0) or 
                forward_response_tensor.size(1) != self.num_to_generate or 
                forward_response_tensor.size(2) != self.topk
            ):
            return False, bittensor.proto.ReturnCode.ResponseShapeException, "output.shape:{} does not match inputs:{} for synapse: {}".format( forward_response_tensor.shape, foward_request_tensor.shape, self )
        else:
            return True, bittensor.proto.ReturnCode.Success, "Success"

    def serialize_forward_request_tensor( self, foward_request_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the dendrite side to serialize the synapse inputs.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_foward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.forward_request_serializer_type )
        return serializer.serialize( foward_request_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def serialize_forward_response_tensor( self, foward_response_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the axon side to serialize the synapse outputs.
            Args:
                foward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_foward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.forward_response_serializer_type )
        return serializer.serialize( foward_response_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def deserialize_backward_request_tensor( self, backward_request_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the dendrite side to serialize the synapse gradients.
            Args:
                backward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_backward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.backward_request_serializer_type )
        return serializer.serialize( backward_request_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def serialize_backward_response_tensor( self, backward_response_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the axon side to serialize the synapse output gradients.
            Args:
                backward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_backward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.backward_response_serializer_type )
        return serializer.serialize( backward_response_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def encode_forward_request_tensor (self, foward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the dendrite side before sending it over the wire. 
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Single torch tensor which should be encoded by the synapse before sending 
                    over the wire.
            Returns:
                encoded_foward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Encoded forward request tensor in format ready to be sent over the wire.
        """
        return torch.where( torch.isnan(foward_request_tensor), torch.zeros_like(foward_request_tensor), foward_request_tensor).detach() 

    def decode_forward_request_tensor (self, foward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the axon side before sending the tensor to the synapse function.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Torch tensor which should should be decoded via the inverse of the function 
                    of encode_forward_request_tensor.
            Returns:
                decoded_foward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Synapse decoded forward request tensor.

        """
        return torch.where( torch.isnan(foward_request_tensor), torch.zeros_like(foward_request_tensor), foward_request_tensor).detach() 

    def encode_forward_response_tensor (self, forward_response_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the axon side before sending the response over the wire. 
            Args:
                forward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be encoded before sending over the wire to the dendrite.
            Returns:
                encoded_forward_response_tensor(:obj:`torch.Tensor`, `required`):
                    Synapse encoded forward response tensor.
        """
        return torch.where( torch.isnan(forward_response_tensor), torch.zeros_like(forward_response_tensor), forward_response_tensor).detach() 

    def decode_forward_response_tensor (self, forward_response_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the dendrite to decode the tensor recieved from the axon on the wire.
            Args:
                forward_response_tensor  (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be decoded via the inverse of the function encode forward response 
                    tensor.

             Returns:
                decoded_forward_response_tensor  (:obj:`torch.Tensor`, `required`):
                    Synapse decoded forward response tensor.
        """
        return torch.where( torch.isnan(forward_response_tensor), torch.zeros_like(forward_response_tensor), forward_response_tensor).detach() 


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
        super(TextLastHiddenState).__init__ (
            forward_request_serializer_type,
            forward_response_serializer_type,
            backward_request_serializer_type,
            backward_response_serializer_type
        )

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
        instance_proto = bittensor.proto.Synapse.TextLastHiddenState.ParseFromString( wire_proto.synapse_data )
        return TextLastHiddenState.deserialize_from_instance_proto( instance_proto )

    def serialize_to_instance_proto( self ) -> 'bittensor.proto.Synapse.TextLastHiddenState':
        """ Serializes the class instance to a Synapse instance proto.
            Returns:
                serialized_synapse_as_instance_proto (:obj:`torch.Tensor`, `required`):
                    Instance class serialized to a instance proto.
        """
        return bittensor.proto.Synapse.TextLastHiddenState ( 
            forward_request_serializer_type = self.forward_request_serializer_type,
            forward_response_serializer_type = self.forward_response_serializer_type,
            backward_request_serializer_type = self.backward_request_serializer_type,
            backward_response_serializer_type = self.backward_response_serializer_type,
        )

    def serialize_to_wire_proto( self ) -> 'bittensor.proto.Synapse':
        """ Serializes the class instance to a Synapse wire proto.
            Returns:
                serialized_synapse_as_wire_proto (:obj:`torch.Tensor`, `required`):
                    Instance class serialized to a wire proto.
        """
        return bittensor.proto.Synapse (
                synapse_data = self.serialize_to_instance_proto().SerializeToString(),
                synapse_type = TextLastHiddenState.synapse_type,
            )

    def nill_forward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite forward call when the call fails.
            Args:
                forward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Tensor being sent as forward request.
            Returns:
                nill_forward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Zeroed forward response tensor.
        """
        return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), bittensor.__network_dim__ ), dtype=torch.float32)

    def nill_backward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite backward call when the call fails.
            Args:
                forward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Tensor being sent as forward request.
            Returns:
                nill_backward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Zeroed backward response tensor.
        """
        return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), forward_request_tensor.size(2) ), dtype=torch.float32)

    def check_forward_request ( self, foward_request_tensor ) -> Tuple[ bool, bittensor.proto.ReturnCode,  str ]:
        """ Checks that the forward request tensor being sent by the dendrite is well formed.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Forward input tensor to be sent on the wire.
            Returns:
                is_success (:obj:`bool`, `required`):
                    Did the forward_response_tensor meet requirements.
                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Check return code. Success of ResponseShapeException.
                message (:obj:`str`, `required`):
                    Message associated with check.
        """
        if  ( len(foward_request_tensor.shape) != 2 ):
            return False, bittensor.proto.ReturnCode.RequestShapeException, "foward_request_tensor.shape:{} is not correct for synapse: {}".format( foward_request_tensor.shape, self )
        else:
            return True, bittensor.proto.ReturnCode.Success, "Success"

    def check_forward_response_shape( self, foward_request_tensor, forward_response_tensor ) -> Tuple[ bool, bittensor.proto.ReturnCode,  str ]:
        """ Checks that the forward response tensor being sent by the axon is well formed.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Forward inputs sent on the wire.
                forward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Forward outputs received on the wire.
            Returns:
                is_success (:obj:`bool`, `required`):
                    Did the forward_response_tensor meet requirements.
                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Check return code. Success of ResponseShapeException.
                message (:obj:`str`, `required`):
                    Message associated with check.
        """
        if  ( 
                forward_response_tensor.size(0) != foward_request_tensor.size(0) or 
                forward_response_tensor.size(1) != foward_request_tensor.size(1) or 
                forward_response_tensor.size(2) != bittensor.__network_dim__
            ):
            return False, bittensor.proto.ReturnCode.ResponseShapeException, "output.shape:{} does not match inputs:{} for synapse: {}".format( forward_response_tensor.shape, foward_request_tensor.shape, self )
        else:
            return True, bittensor.proto.ReturnCode.Success, "Success"

    def serialize_forward_request_tensor( self, foward_request_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the dendrite side to serialize the synapse inputs.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_foward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.forward_request_serializer_type )
        return serializer.serialize( foward_request_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def serialize_forward_response_tensor( self, foward_response_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the axon side to serialize the synapse outputs.
            Args:
                foward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_foward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.forward_response_serializer_type )
        return serializer.serialize( foward_response_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def deserialize_backward_request_tensor( self, backward_request_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the dendrite side to serialize the synapse gradients.
            Args:
                backward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_backward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.backward_request_serializer_type )
        return serializer.serialize( backward_request_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def serialize_backward_response_tensor( self, backward_response_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        """ Function to be called on the axon side to serialize the synapse output gradients.
            Args:
                backward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Tensor to be serialized.
            Returns:
                serialized_backward_response_tensor (:obj:`torch.Tensor`, `required`):
                    Serialzied tensor.
        """
        serializer = bittensor.serializer( self.backward_response_serializer_type )
        return serializer.serialize( backward_response_tensor, from_type = bittensor.proto.TensorType.TORCH )

    def encode_forward_request_tensor (self, foward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the dendrite side before sending it over the wire. 
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Single torch tensor which should be encoded by the synapse before sending 
                    over the wire.
            Returns:
                encoded_foward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Encoded forward request tensor in format ready to be sent over the wire.
        """
        return torch.where( torch.isnan(foward_request_tensor), torch.zeros_like(foward_request_tensor), foward_request_tensor).detach() 

    def decode_forward_request_tensor (self, foward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the axon side before sending the tensor to the synapse function.
            Args:
                foward_request_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Torch tensor which should should be decoded via the inverse of the function 
                    of encode_forward_request_tensor.
            Returns:
                decoded_foward_request_tensor (:obj:`torch.Tensor`, `required`):
                    Synapse decoded forward request tensor.

        """
        return torch.where( torch.isnan(foward_request_tensor), torch.zeros_like(foward_request_tensor), foward_request_tensor).detach() 

    def encode_forward_response_tensor (self, forward_response_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the axon side before sending the response over the wire. 
            Args:
                forward_response_tensor (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be encoded before sending over the wire to the dendrite.
            Returns:
                encoded_forward_response_tensor(:obj:`torch.Tensor`, `required`):
                    Synapse encoded forward response tensor.
        """
        return torch.where( torch.isnan(forward_response_tensor), torch.zeros_like(forward_response_tensor), forward_response_tensor).detach() 

    def decode_forward_response_tensor (self, forward_response_tensor: torch.Tensor ) -> torch.Tensor:
        """ Function to be called on the dendrite to decode the tensor recieved from the axon on the wire.
            Args:
                forward_response_tensor  (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be decoded via the inverse of the function encode forward response 
                    tensor.

             Returns:
                decoded_forward_response_tensor  (:obj:`torch.Tensor`, `required`):
                    Synapse decoded forward response tensor.
        """
        return torch.where( torch.isnan(forward_response_tensor), torch.zeros_like(forward_response_tensor), forward_response_tensor).detach() 


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
    def TextCasualLM ( 
        topk:int = 512,
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ) -> TextCasualLM:
        """ Factory function which returns a TextCasualLM synapse adapter given arguments.
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
                TextCasualLM (:obj:`TextCasualLM`, `required`):
                    TextCasualLM instance adapter class.
        """
        return TextCasualLM ( 
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
        if synapse_wire_proto.synapse_type == bittensor.proto.SynapseType.TEXT_LAST_HIDDEN_STATE:
            return TextLastHiddenState.deserialize_from_wire_proto ( synapse_wire_proto )
        elif synapse_wire_proto.synapse_type == bittensor.proto.SynapseType.TEXT_CAUSAL_LM:
            return TextCasualLM.deserialize_from_wire_proto( synapse_wire_proto )
        elif synapse_wire_proto.synapse_type == bittensor.proto.SynapseType.TEXT_SEQ_2_SEQ:
            return TextSeq2Seq.deserialize_from_wire_proto( synapse_wire_proto )
        else:
            raise ValueError("Synapse type is unknown.")