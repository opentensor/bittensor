
from yaml import serialize_all
import bittensor
import torch
from typing import Union, List, Tuple, Optional


class Synapse:

    def __init__(
        self, 
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type',
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type',
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type',
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type',
    ) -> 'Synapse':
        self.forward_request_serializer_type = forward_request_serializer_type
        self.forward_response_serializer_type = forward_response_serializer_type
        self.backward_request_serializer_type = backward_request_serializer_type
        self.backward_response_serializer_type = backward_response_serializer_type

    @staticmethod
    def deserialize_from_instance_proto ( proto: bittensor.proto.Synapse ) -> 'Synapse':
        raise NotImplementedError("deserialize_from_instance_proto should be implemented by the subclass.")

    @staticmethod
    def deserialize_from_wire_proto ( proto: bittensor.proto.Synapse ) -> 'Synapse':
        raise NotImplementedError("deserialize_from_wire_proto should be implemented by the subclass.")

    def serialize_to_instance_proto( self ) -> 'bittensor.proto.Synapse':
        raise NotImplementedError("serialize_to_instance_proto should be implemented by the subclass.")

    def serialize_to_wire_proto( self ) -> 'bittensor.proto.Synapse':
        raise NotImplementedError("serialize_to_wire_proto should be implemented by the subclass.")

    def nill_response_for_inputs( self ) -> torch.Tensor:
        raise NotImplementedError("nill_response_for_input should be implemented by the subclass.")

    def check_forward_request_shape( self, foward_request_tensor ) -> Tuple[ bool, bittensor.proto.ReturnCode,  str ]:
        raise NotImplementedError("check_response should be implemented by the subclass.")

    def check_forward_response_shape( self, foward_request_tensor, forward_response_tensor ) -> Tuple[ bool, bittensor.proto.ReturnCode,  str ]:
        raise NotImplementedError("check_response should be implemented by the subclass.")

    def serialize_forward_request_tensor( self, foward_request_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        return bittensor.serializer( serialize_type = self.forward_request_serializer_type ).serialize( foward_request_tensor )

    def serialize_forward_response_tensor( self, foward_response_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        return bittensor.serializer( serialize_type = self.forward_response_serializer_type ).serialize( foward_response_tensor )

    def serialize_backward_request_tensor( self, backward_request_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        return bittensor.serializer( serialize_type = self.backward_request_serializer_type ).serialize( backward_request_tensor )

    def serialize_backward_response_tensor( self, backward_response_tensor: torch.Tensor ) -> bittensor.proto.Tensor:
        return bittensor.serializer( serialize_type = self.backward_response_serializer_type ).serialize( backward_response_tensor )

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
    synapse_type: bittensor.proto.Synapse.SynapseType = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM
    def __init__( self, topk: int = 512 ):
        self.topk = topk

    @staticmethod
    def deserialize_from_instance_proto ( proto: bittensor.proto.Synapse.TextCausalLM ):
        return TextCasualLM( topk = proto.topk )

    @staticmethod
    def deserialize_from_wire_proto ( wire_proto: 'bittensor.proto.Synapse' ) -> 'TextCasualLM':
        instance_proto = bittensor.proto.Synapse.TextCasualLM.ParseFromString( wire_proto.synapse_data )
        return TextCasualLM.deserialize_from_instance_proto( instance_proto )

    def serialize_to_instance_proto ( self ) -> bittensor.proto.Synapse.TextCausalLM:
        return bittensor.proto.Synapse.TextCausalLM( topk = self.topk )

    def serialize_to_wire_proto ( self ) -> bittensor.proto.Synapse:
        return bittensor.proto.Synapse (
                synapse_data = self.to_proto().SerializeToString(),
                synapse_type = TextCasualLM.synapse_type,
            )

    def nill_response_for_inputs ( inputs: torch.Tensor ) -> torch.Tensor:
        return torch.zeros( ( inputs.size(0), inputs.size(1), bittensor.__vocab_size__ ), dtype=torch.float32)

    def check_forward_response_shape( self, inputs_request, outputs_response ) -> Tuple[ bool, bittensor.proto.ReturnCode,  str ]:
        if  ( 
                outputs_response.size(0) != inputs_request.size(0) or 
                outputs_response.size(1) != inputs_request.size(1) or 
                outputs_response.size(2) != self.topk
            ):
            return False, bittensor.proto.ReturnCode.ResponseShapeException, "output.shape:{} does not match inputs:{} for synapse: {}".format( outputs_response.shape, inputs_request.shape, self )
        else:
            return True, bittensor.proto.ReturnCode.Success, "Success"

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
        return torch.where( torch.isnan(output_tensor), torch.zeros_like(output_tensor), output_tensor).detach() 

class TextSeq2Seq (Synapse):
    synapse_type: bittensor.proto.Synapse.SynapseType = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ
    def __init__( self, topk:int = 512, num_to_generate: int = 512 ):
        self.topk = topk
        self.num_to_generate = num_to_generate

    @staticmethod
    def deserialize_from_instance_proto ( instance_proto: 'bittensor.proto.Synapse.TestSeq2Seq' ) -> 'TextSeq2Seq':
        return TextSeq2Seq (
            topk = instance_proto.topk,
            num_to_generate = instance_proto.num_to_generate 
        )

    @staticmethod
    def deserialize_from_wire_proto ( wire_proto: 'bittensor.proto.Synapse' ) -> 'TextSeq2Seq':
        instance_proto = bittensor.proto.Synapse.TestSeq2Seq.ParseFromString( wire_proto.synapse_data )
        return TextSeq2Seq.deserialize_from_instance_proto( instance_proto )

    def serialize_to_instance_proto( self ) -> bittensor.proto.Synapse.TestSeq2Seq:
        return bittensor.proto.Synapse.TestSeq2Seq( num_to_generate = self.num_to_generate )

    def serialize_to_wire_proto( self ) -> bittensor.proto.Synapse:
        return bittensor.proto.Synapse (
                synapse_data = self.to_proto().SerializeToString(),
                synapse_type = TextSeq2Seq.synapse_type,
            )

    def nill_response_for_inputs ( inputs: torch.Tensor ) -> torch.Tensor:
        return torch.zeros( ( inputs.size(0), inputs.size(1), bittensor.__vocab_size__ ), dtype=torch.float32)

    def check_forward_response_shape( self, inputs_request, outputs_response ) -> Tuple[ bool, bittensor.proto.ReturnCode,  str ]:
        if  ( 
                outputs_response.size(0) != inputs_request.size(0) or 
                outputs_response.size(1) != self.num_to_generate or 
                outputs_response.size(2) != self.topk
            ):
            return False, bittensor.proto.ReturnCode.ResponseShapeException, "output.shape:{} does not match inputs:{} for synapse: {}".format( outputs_response.shape, inputs_request.shape, self )
        else:
            return True, bittensor.proto.ReturnCode.Success, "Success"
    
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
        return torch.where( torch.isnan(output_tensor), torch.zeros_like(output_tensor), output_tensor).detach() 
  

class TextLastHiddenState (Synapse):
    synapse_type: bittensor.proto.Synapse.SynapseType = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE
    def __init__( self ):
        pass

    @staticmethod
    def deserialize_from_instance_proto ( instance_proto: 'bittensor.proto.Synapse.TextLastHiddenState' ) -> 'TextLastHiddenState':
        return TextLastHiddenState( num_to_generate = instance_proto.num_to_generate )

    @staticmethod
    def deserialize_from_wire_proto ( wire_proto: 'bittensor.proto.Synapse' ) -> 'TextLastHiddenState':
        instance_proto = bittensor.proto.Synapse.TextLastHiddenState.ParseFromString( wire_proto.synapse_data )
        return TextLastHiddenState.deserialize_from_instance_proto( instance_proto )

    def serialize_to_instance_proto ( self ) -> bittensor.proto.Synapse.TextLastHiddenState:
        return bittensor.proto.Synapse.TextLastHiddenState()

    def serialize_to_wire_proto ( self ) -> bittensor.proto.Synapse:
        return bittensor.proto.Synapse (
                synapse_data = self.to_proto().SerializeToString(),
                synapse_type = TextLastHiddenState.synapse_type,
            )

    def nill_response_for_inputs ( inputs: torch.Tensor ) -> torch.Tensor:
        return torch.zeros( ( inputs.size(0), inputs.size(1), bittensor.__network_dim__ ), dtype=torch.float32)

    def check_forward_response_shape( self, inputs_request, outputs_response ) -> Tuple[ bool, bittensor.proto.ReturnCode,  str ]:
        if  ( 
                outputs_response.size(0) != inputs_request.size(0) or 
                outputs_response.size(1) != self.num_to_generate or 
                outputs_response.size(2) != self.topk
            ):
            return False, bittensor.proto.ReturnCode.ResponseShapeException, "output.shape:{} does not match inputs:{} for synapse: {}".format( outputs_response.shape, inputs_request.shape, self )
        else:
            return True, bittensor.proto.ReturnCode.Success, "Success"

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
        return torch.where( torch.isnan(output_tensor), torch.zeros_like(output_tensor), output_tensor).detach() 
  

class synapse:
    
    @staticmethod
    def TextLastHiddenState ( ) -> TextLastHiddenState:
        return TextLastHiddenState ( )

    @staticmethod
    def TextCasualLM ( topk:int = 512 ) -> TextCasualLM:
        return TextCasualLM ( 
            topk = topk 
        )

    @staticmethod
    def TextSeq2Seq ( topk:int = 512, num_to_generate: int = 512 ) -> TextSeq2Seq:
        return TextSeq2Seq ( 
            topk = topk, 
            num_to_generate = num_to_generate 
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