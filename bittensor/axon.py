import argparse
import grpc
import random
import requests
import threading
import torch
import validators

from concurrent import futures
from munch import Munch
from loguru import logger

import bittensor
from bittensor.nucleus import Nucleus
from bittensor.synapse import Synapse
from bittensor import bittensor_pb2
from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor.serializer import PyTorchSerializer

def obtain_ip(config: Munch) -> Munch:
    if config.axon.remote_ip != None:
        return config
    try:
        value = requests.get('https://api.ipify.org').text
    except:
        logger.error("CONIG: Could not retrieve public facing IP from IP API.")
        raise SystemError('CONFIG: Could not retrieve public facing IP from IP API.')
    if not validators.ipv4(value):
        logger.error("CONFIG: Response from IP API is not a valid IP with ip {}", value)
        raise SystemError('CONFIG: Response from IP API is not a valid IP with ip {}'.format(value))
    config.axon.remote_ip = value
    logger.info('remote ip: {}', value)
    return config

class Axon(bittensor_grpc.BittensorServicer):
    r"""
    The Axon serves a bittensor.synapse.Synapse to recieve remote Forward & Backward calls on the network.
    
    """
    def __init__(self, config, keypair):
        r""" Serves a Synapse to the axon server replacing the previous Synapse if exists.

            Args:
                config (:obj:`Munch`, `required`): 
                    bittensor Munch config.
                keypair (:obj:`substrateinterface.Keypair`, `required`): 
                    bittensor keypair.
        """
        self._config = config
        self.__keypair = keypair

        # Init server objects.
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        bittensor_grpc.add_BittensorServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' + str(self._config.axon.port))

        # Local synapse.
        self.synapse = None

        # Serving thread.
        self._thread = None

        self._nucleus = Nucleus(config)

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--axon.port', default=8091, type=int, 
                            help='Port to serve axon')

        parser.add_argument('--axon.remote_ip', default=None, type=str, 
                            help='Remote IP to serve to chain.')
        parser = Nucleus.add_args(parser)
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        config = Nucleus.check_config(config)
        logger.info('optaining remote ip ...')
        config = obtain_ip(config)

        assert config.axon.port > 1024 and config.axon.port < 65535, 'config.axon.port must be in range [1024, 65535]'
        return config

    def __del__(self):
        r""" Called when this axon is deleted, ensures background threads shut down properly.
        """
        self.stop()

    def start(self):
        r""" Starts the standalone axon GRPC server thread.
        """
        self._thread = threading.Thread(target=self._serve, daemon=False)
        self._thread.start()

    def _serve(self):
        try:
            self._server.start()
        except (KeyboardInterrupt, SystemExit):
            self.stop()
        except Exception as e:
            logger.error(e)


    def stop(self):
        r""" Stop the axon grpc server.
        """
        logger.info('Shutting down the Nucleus...')
        self._nucleus.stop()
        if self._server != None:
            self._server.stop(0)

    def serve(self, synapse: Synapse):
        r""" Serves a Synapse to the axon server replacing the previous Synapse if exists.

            Args:
                synapse (:obj:`bittensor.Synapse`, `required`): 
                    synpase object to serve on the axon server.
        """
        self.synapse = synapse


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

        try:
            outputs, message, code = self._nucleus.forward(synapse = self.synapse, inputs = x, mode = inputs.modality, priority = random.random())

        except Exception as e:
            logger.error(e)
            message = "Unknown exception when calling nucleus forward {}".format(e)
            code =  bittensor_pb2.ReturnCode.UnknownException,
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
            

    def Forward(self, request: bittensor_pb2.TensorMessage,
                context: grpc.ServicerContext):

        r""" Function called by remote GRPC Forward requests by other neurons.

            Args:
                request (:obj:`bittensor_pb2`, `required`): 
                    Tensor request Proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
        """
        tensor, message, code = self._forward(request)
        response = bittensor_pb2.TensorMessage(
            version = bittensor.__version__, 
            public_key = self.__keypair.public_key, 
            return_code = code,
            message = message,
            tensors = [tensor]
        )
        return response


    def Backward(self, request: bittensor_pb2.TensorMessage,
                 context: grpc.ServicerContext):
        try:

            # Check axon is serving synapse.
            if self.synapse == None:
                error_msg = "Remote axon not serving a synapse"
                logger.warning(error_msg)
                response = bittensor_pb2.TensorMessage(
                    version = bittensor.__version__, 
                    public_key = self.__keypair.public_key, 
                    return_code =  bittensor_pb2.ReturnCode.NotServingSynapse,
                    message = error_msg)
                return response

            # Check request inputs.
            if len(request.tensors) == 2:
                inputs_x = request.tensors[0]
                grads_dy = request.tensors[1]
            else:
                error_msg = "During backward: There are {} tensors in the request, expected 2.".format(len(request.tensors))
                logger.warning(error_msg)
                response = bittensor_pb2.TensorMessage(
                    version = bittensor.__version__, 
                    public_key = self.__keypair.public_key, 
                    return_code =  bittensor_pb2.ReturnCode.InvalidRequest,
                    message = error_msg)
                return response

            
            # Deserialize request.
            try:
                inputs_x = PyTorchSerializer.deserialize(inputs_x)
                grads_dy = PyTorchSerializer.deserialize(grads_dy)
                
            except Exception as e:
                error_msg  = "Backward request deserialization failed with unknown error {}".format(e)
                logger.warning(error_msg)
                response = bittensor_pb2.TensorMessage(
                    version = bittensor.__version__, 
                    public_key = self.__keypair.public_key, 
                    return_code =  bittensor_pb2.ReturnCode.RequestDeserializationException,
                    message = error_msg)
                return response

            # Get grads by calling backward.
            try:
                inputs_dx = self.synapse.call_backward(inputs_x, grads_dy)

            except Exception as e:
                error_msg  = "Unkown exception when calling backward with error {}".format(e)
                logger.warning(error_msg)
                response = bittensor_pb2.TensorMessage(
                    version = bittensor.__version__, 
                    public_key = self.__keypair.public_key, 
                    return_code =  bittensor_pb2.ReturnCode.UnknownException,
                    message = error_msg)
                return response

            # Deserialize response.
            try:
                inputs_dx_serialized = PyTorchSerializer.serialize_tensor(inputs_dx)

            except Exception as e:
                error_msg  = "Backward request serialization failed with error {} and inputs {}".format(e, inputs_dx)
                logger.warning(error_msg)
                response = bittensor_pb2.TensorMessage(
                    version = bittensor.__version__, 
                    public_key = self.__keypair.public_key, 
                    return_code =  bittensor_pb2.ReturnCode.ResponseSerializationException,
                    message = error_msg)
                return response


            # Final response.
            response = bittensor_pb2.TensorMessage(
                version = bittensor.__version__,
                public_key = self.__keypair.public_key,
                return_code =  bittensor_pb2.ReturnCode.Success,
                message = "success",
                tensors = [inputs_dx_serialized])

            return response

        # Final catch of unknown exceptions.
        except Exception as e:
            error_msg  = "Calling backward request failed with unknown error {}".format(e)
            logger.warning(error_msg)
            response = bittensor_pb2.TensorMessage(
                version = bittensor.__version__, 
                public_key = self.__keypair.public_key, 
                return_code =  bittensor_pb2.ReturnCode.UnknownException,
                message = error_msg)
            return response

