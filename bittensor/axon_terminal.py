import argparse
import grpc
from munch import Munch
from loguru import logger
from typing import List, Tuple
import sys, traceback
import bittensor
from bittensor import bittensor_pb2
from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor.nucleus import Nucleus
from bittensor.serializer import PyTorchSerializer

class AxonTerminal(bittensor_grpc.BittensorServicer):

    def __init__(self, config, keypair, synapse, nucleus):
        self._config = config
        self.__keypair = keypair
        self.synapse = synapse
        self.nucleus = nucleus
        logger.info('init axon terminal.')

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        return config

    async def Forward(self, request: bittensor_pb2.TensorMessage,
                context: grpc.ServicerContext):

        r""" GRPC Forward requests called by other neurons.

            Args:
                request (:obj:`bittensor_pb2`, `required`): 
                    Tensor request Proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
        """
        logger.info('terminal async recieved forward.')
        response = bittensor_pb2.TensorMessage(
            version = bittensor.__version__, 
            public_key = self.__keypair.public_key, 
        )
        return response

        # output, message, code = self._forward(request)
        # response = bittensor_pb2.TensorMessage(
        #     version = bittensor.__version__, 
        #     public_key = self.__keypair.public_key, 
        #     return_code = code,
        #     tensors = [ output ],
        #     message = message
        # )
        # return response

    async def Backward(self, request: bittensor_pb2.TensorMessage,
                 context: grpc.ServicerContext):

        logger.info('terminal recieved async backward.')
        response = bittensor_pb2.TensorMessage(
            version = bittensor.__version__, 
            public_key = self.__keypair.public_key, 
        )
        return response

        # output, message, code = self._backward(request)
        # response = bittensor_pb2.TensorMessage(
        #     version = bittensor.__version__, 
        #     public_key = self.__keypair.public_key, 
        #     return_code = code,
        #     tensors = [ output ],
        #     message = message
        # )
        # return response

    def _forward(self, request):
        # Check synapse exists.
        if self.synapse == None:
            message = "Remote axon not serving a synapse"
            code = bittensor_pb2.ReturnCode.NotServingSynapse,
            return None, message, code

        # Check Empty request.
        if len(request.tensors) == 0:
            message = "Forward request contains {} tensors, expected 1 tensor in the forward call".format(len(request.tensors))
            code = bittensor_pb2.ReturnCode.EmptyRequest
            return None, message, code

        # Check deserialization.
        inputs = request.tensors[0]
        try:
            x = PyTorchSerializer.deserialize(inputs)
        except Exception as e:
            message  = "Forward request deserialization failed with error {}".format(e)
            code = bittensor_pb2.ReturnCode.RequestDeserializationException
            return None, message, code


        # Check shape and modality.
        if x.shape[0] < 1:
            message = "Froward request batch dim exception with batch_size = {} ".format(x.shape[0])
            code = bittensor_pb2.ReturnCode.RequestShapeException,
            return None, message, code

        if x.shape[1] < 1:
            message = "Forward request sequence dim exception with sequence_dim = {} ".format(x.shape[1])
            code =  bittensor_pb2.ReturnCode.RequestShapeException,
            return None, message, code

        if inputs.modality == bittensor_pb2.Modality.TEXT:
            if len(x.shape) != 2:
                message = "Forward text input shape exception with len(request.shape) = {} must have rank 2.".format(len(x.shape))
                code =  bittensor_pb2.ReturnCode.RequestShapeException,
                return None, message, code
            
        if inputs.modality == bittensor_pb2.Modality.IMAGE:
            if len(x.shape) != 5:
                message =  "Forward image input shape exception for len(shape) = {}  must have rank 5".format(len(x.shape))
                code =  bittensor_pb2.ReturnCode.RequestShapeException,
                return None, message, code

        if inputs.modality == bittensor_pb2.Modality.TENSOR:
            if len(x.shape) != 3:
                message = "Forward message tensor input shape exception len(shape) = {} must have rank 3".format(len(x.shape))
                code = bittensor_pb2.ReturnCode.RequestShapeException,
                return None, message, code

        # Call forward pool in nucleus.
        try:
            outputs, message, code = self.nucleus.forward(synapse = self.synapse, inputs = x, mode = inputs.modality, priority = 10)

        except Exception as e:
            logger.error(e)
            message = "Unknown exception when calling nucleus forward {}".format(e)
            code =  bittensor_pb2.ReturnCode.UnknownException,
            traceback.print_exc(file=sys.stdout)
            return None, message, code

        # Serialize response.
        try:
            outputs_serialized = PyTorchSerializer.serialize_tensor(outputs)
        
        except Exception as e:
            message = "Serializtion of forward response failed with error {} and inputs: {}".format(e, outputs)
            code = bittensor_pb2.ReturnCode.ResponseDeserializationException,
            return None, message, code

        # Return successful response
        return outputs_serialized, message, code

    def _backward(self, request):
        
        if self.synapse == None:
            message = "Remote axon not serving a synapse"
            code = bittensor_pb2.ReturnCode.NotServingSynapse
            return None, message, code

        if len(request.tensors) != 2:
            message = "During backward: There are {} tensors in the request, expected 2.".format(len(request.tensors))
            code = bittensor_pb2.ReturnCode.InvalidRequest
            return None, message, code
        else:
            inputs_x = request.tensors[0]
            grads_dy = request.tensors[1]
           
        try:
            inputs_x = PyTorchSerializer.deserialize(inputs_x)
            grads_dy = PyTorchSerializer.deserialize(grads_dy)        
        except Exception as e:
            message = "Backward request deserialization failed with error {}".format(e)
            code =  bittensor_pb2.ReturnCode.RequestDeserializationException
            return None, message, code

        # Make nucleus backward call.
        try:
            outputs, message, code = self.nucleus.backward(self.synapse, inputs_x, grads_dy, priority = 10)

        except Exception as e:
            message = "Unkown exception when calling nucleus backward with error {}".format(e)
            code = bittensor_pb2.ReturnCode.UnknownException
            traceback.print_exc(file=sys.stdout)
            return None, message, code

        try:
            outputs_serialized = PyTorchSerializer.serialize_tensor(outputs)
        except Exception as e:
            message = "Backward request serialization failed with error {} and inputs {}".format(e, outputs)
            code =  bittensor_pb2.ReturnCode.ResponseSerializationException
            return None, message, code

        # Success.
        return outputs_serialized, message, code
        