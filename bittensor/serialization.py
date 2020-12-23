import torch
from bittensor import bittensor_pb2
from bittensor.serializers.pytorchpickle import PyTorchPickleSerializer

class SerializationException (Exception):
    """ Raised during serialization """
    pass

class DeserializationException (Exception):
    """ Raised during deserialization """
    pass

class NoSerializerForEnum (Exception):
    """ Raised if there is no serializer for the passed type """
    pass

class SerializationTypeNotImplementedException (Exception):
    """ Raised if serialization/deserialization is not implemented for the passed object type """
    pass

class BittensorSerializerBase(object):
    r""" Bittensor base serialization object for converting between bittensor_pb2.Tensor and their
    various python tensor equivalents. i.e. torch.Tensor or tensorflow.Tensor
    """

    @staticmethod
    def serialize (tensor_obj: object, from_type: int) -> bittensor_pb2.Tensor:
        """Serializes a torch object to bittensor_pb2.Tensor wire format.

        Args:
            tensor_obj (:obj:`object`, `required`): 
                general tensor object i.e. torch.Tensor or tensorflow.Tensor

            from_type (`obj`: bittensor_pb2.TensorType, `required`): 
                Serialization from this type. i.e. bittensor_pb2.TensorType.TORCH or bittensor_pb2.TensorType.TENSORFLOW

        Returns:
            tensor_pb2: (obj: `bittensor_pb2.Tensor`, `required`): 
                Serialized tensor as bittensor_pb2.proto. 

        Raises:
            SerializationTypeNotImplementedException (Exception):
                Raised if the serializer does not implement the conversion between the passed type and a bittensor_pb2.Tensor

            SerializationException: (Exception): 
                Raised when the subclass serialization throws an error for the passed object.
        """
        # TODO (const): add deserialization types for torch -> tensorflow 
        if from_type == bittensor_pb2.TensorType.TORCH:
            return BittensorSerializerBase.serialize_from_torch( tensor_obj )

        elif from_type == bittensor_pb2.TensorType.NUMPY:
            return BittensorSerializerBase.serialize_from_numpy( tensor_obj )

        elif from_type == bittensor_pb2.TensorType.TENSORFLOW:
            return BittensorSerializerBase.serialize_from_tensorflow( tensor_obj )

        else:
            raise SerializationTypeNotImplementedException("Serialization from type {} not implemented.".format(from_type))

        raise NotImplementedError

    @staticmethod
    def deserialize (tensor_pb2: bittensor_pb2.Tensor, to_type: int) -> object:
        """Serializes a torch object to bittensor_pb2.Tensor wire format.

        Args:
            tensor_pb2 (`obj`: bittensor_pb2.Tensor, `required`): 
                Serialized tensor as bittensor_pb2.proto. 

            to_type (`obj`: bittensor_pb2.TensorType, `required`): 
                Deserialization to this type. i.e. bittensor_pb2.TensorType.TORCH or bittensor_pb2.TensorType.TENSORFLOW

        Returns:
            tensor_obj (:obj:`torch.FloatTensor`, `required`): 
                tensor object of type from_type in bittensor_pb2.TensorType

        Raises:
            SerializationTypeNotImplementedException (Exception):
                Raised if the serializer does not implement the conversion between the pb2 and the passed type.
          
            DeserializationException: (Exception): 
                Raised when the subclass deserializer throws an error for the passed object.
        """
        # TODO (const): add deserialization types for torch -> tensorflow 
        if to_type == bittensor_pb2.TensorType.TORCH:
            return BittensorSerializerBase.derserialize_to_torch( tensor_pb2 )

        elif to_type == bittensor_pb2.TensorType.NUMPY:
            return BittensorSerializerBase.derserialize_to_numpy( tensor_pb2 )

        elif to_type == bittensor_pb2.TensorType.TENSORFLOW:
            return BittensorSerializerBase.derserialize_to_tensorflow( tensor_pb2 )

        else:
            raise SerializationTypeNotImplementedException("Deserialization to type {} not implemented.".format(to_type))

    @staticmethod
    def serialize_from_torch(torch_tensor: torch.Tensor) -> bittensor_pb2.Tensor:
        raise SerializationTypeNotImplementedException


    @staticmethod
    def derserialize_to_torch(tensor_pb2: bittensor_pb2.Tensor) -> torch.Tensor:
        raise SerializationTypeNotImplementedException

    @staticmethod
    def serialize_from_tensorflow(torch_tensor: torch.Tensor) -> bittensor_pb2.Tensor:
        raise SerializationTypeNotImplementedException


    @staticmethod
    def derserialize_to_tensorflow(tensor_pb2: bittensor_pb2.Tensor) -> torch.Tensor:
        raise SerializationTypeNotImplementedException

    @staticmethod
    def serialize_from_numpy(torch_tensor: torch.Tensor) -> bittensor_pb2.Tensor:
        raise SerializationTypeNotImplementedException


    @staticmethod
    def derserialize_to_numpy(tensor_pb2: bittensor_pb2.Tensor) -> torch.Tensor:
        raise SerializationTypeNotImplementedException



def get_serializer_for_type(serialzer_type: bittensor_pb2.Serializer) -> BittensorSerializerBase:
    r"""Returns the correct serializer object for the passed Serializer enum. 

        Args:
            serialzer_type (:obj:`bittensor_pb2.Serializer`, `required`): 
                The serialzer_type ENUM from bittensor_pb2.

        Returns:
            BittensorSerializerBase: (obj: `BittensorSerializerBase`, `required`): 
                The bittensor serializer/deserialzer for the passed type.

        Raises:
            NoSerializerForEnum: (Exception): 
                Raised if the passed there is no serialzier for the passed type. 
    """

    if serialzer_type == bittensor_pb2.Serializer.PICKLE:
        return PyTorchPickleSerializer
    else:
        raise NoSerializerForEnum("No known serialzier for proto type {}".format(serialzer_type))

