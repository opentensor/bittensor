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
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        bittensor_grpc.add_BittensorServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' + str(self._config.axon.port))

        # Local synapse.
        self._synapse = None

        # Serving thread.
        self._thread = None

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--axon.port', default=8091, type=int, 
                            help='Port to serve axon')

        parser.add_argument('--axon.remote_ip', default=None, type=str, 
                            help='Remote IP to serve to chain.')
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
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
        self._thread = threading.Thread(target=self._serve, daemon=True)
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
        if self._server != None:
            self._server.stop(0)

    def serve(self, synapse: Synapse):
        r""" Serves a Synapse to the axon server replacing the previous Synapse if exists.

            Args:
                synapse (:obj:`bittensor.Synapse`, `required`): 
                    synpase object to serve on the axon server.
        """
        self._synapse = synapse

    def Forward(self, request: bittensor_pb2.TensorMessage,
                context: grpc.ServicerContext):

        r""" Function called by remote GRPC Forward requests by other neurons.

            Args:
                request (:obj:`bittensor_pb2`, `required`): 
                    Tensor request Proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
        """
    
        try:

            # Check tensor size.
            if len(request.tensors) > 0:
                inputs = request.tensors[0]
            else:
                error_msg = "Forward: This request contains {} tensors, expected 1 tensor in the forward call".format(len(request.tensors))
                logger.error(error_msg)
                response = bittensor_pb2.TensorMessage(
                    version = bittensor.__version__, 
                    public_key = self.__keypair.public_key, 
                    return_code = bittensor_pb2.ReturnCode.EmptyRequest,
                    message = error_msg)
                return response

            # Deserialize request.
            try:
                x = PyTorchSerializer.deserialize(inputs)

            except bittensor.exceptions.DeserializationException as e:
                error_msg = "Deserialization of inputs failed with error {} and inputs: {}".format(e, inputs)
                logger.error(error_msg)
                response = bittensor_pb2.TensorMessage(
                    version = bittensor.__version__, 
                    public_key = self.__keypair.public_key,
                    return_code = bittensor_pb2.ReturnCode.RequestDeserializationException,
                    message = error_msg)
                return response

            except Exception as e:
                error_msg  = "Request deserialization failed with unknown error {}".format(e)
                logger.error(error_msg)
                response = bittensor_pb2.TensorMessage(
                                version = bittensor.__version__, 
                                public_key = self.__keypair.public_key, 
                                return_code = bittensor_pb2.ReturnCode.RequestDeserializationException,
                                message = error_msg)
                return response

            # Check batch size.
            if x.shape[0] < 1:
                error_msg = "Request batch dim exception with batch_size = {} ".format(x.shape[0])
                logger.error(error_msg)
                response = bittensor_pb2.TensorMessage( version = bittensor.__version__, 
                                                        public_key = self.__keypair.public_key, 
                                                        return_code = bittensor_pb2.ReturnCode.RequestShapeException,
                                                        message = error_msg)
                return response

            # Check sequence dimension.
            if x.shape[1] < 1:
                error_msg = "Request sequence dim exception with sequence_dim = {} ".format(x.shape[1])
                logger.error(error_msg)
                response = bittensor_pb2.TensorMessage( version = bittensor.__version__, 
                                                        public_key = self.__keypair.public_key, 
                                                        return_code = bittensor_pb2.ReturnCode.RequestShapeException,
                                                        message = error_msg)
                return response

            if inputs.modality == bittensor_pb2.Modality.TEXT:
                if len(x.shape) != 2:
                    error_msg = "Text input shape exception with len(request.shape) = {} must have rank 2.".format(len(x.shape))
                    logger.error(error_msg)
                    response = bittensor_pb2.TensorMessage( 
                        version = bittensor.__version__, 
                        public_key = self.__keypair.public_key, 
                        return_code = bittensor_pb2.ReturnCode.RequestShapeException,
                        message = error_msg)
                    return response
            

            elif inputs.modality == bittensor_pb2.Modality.IMAGE:
                if len(x.shape) != 5:
                    error_msg =  "Image input shape exception for len(shape) = {}  must have rank 5".format(len(x.shape))
                    logger.error(error_msg)
                    response = bittensor_pb2.TensorMessage( 
                        version = bittensor.__version__, 
                        public_key = self.__keypair.public_key, 
                        return_code = bittensor_pb2.ReturnCode.RequestShapeException,
                        message = error_msg)
                    return response


            elif inputs.modality == bittensor_pb2.Modality.TENSOR:
                if len(x.shape) != 3:
                    error_msg = "Tensor input shape exception len(shape) = {} must have rank 3".format(len(x.shape))
                    logger.error(error_msg)
                    response = bittensor_pb2.TensorMessage( 
                        version = bittensor.__version__, 
                        public_key = self.__keypair.public_key, 
                        return_code = bittensor_pb2.ReturnCode.RequestShapeException,
                        message = error_msg)
                    return response

            # Call forward network. May call NotImplementedError:
            try:
                y = self._synapse.call_forward(x, inputs.modality)
            
            # Catch not implemented.
            except NotImplementedError:
                error_msg = "Synapse has not implemented the modality {}".format(inputs.modality)
                logger.error(error_msg)
                response = bittensor_pb2.TensorMessage( 
                        version = bittensor.__version__, 
                        public_key = self.__keypair.public_key, 
                        return_code = bittensor_pb2.ReturnCode.NotImplemented,
                        message = error_msg)
                return response
            
            # Catch unknown exceptions.
            except Exception as e:
                error_msg = "Unknown exception when calling remote synapse with error {}".format(e)
                logger.error(error_msg)
                response = bittensor_pb2.TensorMessage( 
                        version = bittensor.__version__, 
                        public_key = self.__keypair.public_key, 
                        return_code = bittensor_pb2.ReturnCode.UnknownException,
                        message = error_msg)
                return response

            # Serialize responses.
            try:
                y_serialized = [PyTorchSerializer.serialize_tensor(y)]
            
            except bittensor.exceptions.SerializationException as e:
                error_msg = "Serializtion of response failed with error {} and inputs: {}".format(e, inputs)
                logger.error(error_msg)
                response = bittensor_pb2.TensorMessage(
                    version = bittensor.__version__, 
                    public_key = self.__keypair.public_key,
                    return_code = bittensor_pb2.ReturnCode.ResponseDeserializationException,
                    message = error_msg)
                return response

            except Exception as e:
                error_msg  = "Response serialization failed with unknown error {}".format(e)
                logger.error(error_msg)
                response = bittensor_pb2.TensorMessage(
                                version = bittensor.__version__, 
                                public_key = self.__keypair.public_key, 
                                return_code = bittensor_pb2.ReturnCode.ResponseDeserializationException,
                                message = error_msg)
                return response

        # Final catch of unknown exceptions.
        except Exception as e:
            error_msg  = "Calling forward request failed with unknown error {}".format(e)
            logger.error(error_msg)
            response = bittensor_pb2.TensorMessage(
                version = bittensor.__version__, 
                public_key = self.__keypair.public_key, 
                return_code = bittensor_pb2.ReturnCode.UnknownException,
                message = error_msg)

        # Finally build response with serialized outputs.
        finally:
            response = bittensor_pb2.TensorMessage(
                version = bittensor.__version__,
                public_key = self.__keypair.public_key,
                return_code = bittensor_pb2.ReturnCode.Success,
                message = "success",
                tensors = y_serialized)
        
        return response



    def Backward(self, request: bittensor_pb2.TensorMessage,
                 context: grpc.ServicerContext):
        try:

            # Check request inputs.
            if len(request.tensors) == 2:
                inputs_x = request.tensors[0]
                grads_dy = request.tensors[1]
            else:
                error_msg = "During backward: There are {} tensors in the request, expected 2.".format(len(request.tensors))
                logger.trace(error_msg)
                response = bittensor_pb2.TensorMessage(
                    version = bittensor.__version__, 
                    public_key = self.__keypair.public_key, 
                    code = bittensor.exceptions.InvalidRequestExceptionCode,
                    message = error_msg)
                return response

            
            # Deserialize request.
            try:
                inputs_x = PyTorchSerializer.deserialize(inputs_x)
                grads_dy = PyTorchSerializer.deserialize(grads_dy)

            except bittensor.exceptions.DeserializationException as e:
                error_msg  = "Backward request deserialization failed with error {} and inputs {} and grads {}".format(e, inputs_x, grads_dy)
                logger.error(error_msg)
                response = bittensor_pb2.TensorMessage(
                    version = bittensor.__version__, 
                    public_key = self.__keypair.public_key, 
                    code = bittensor.exceptions.RequestDeserializationExceptionCode,
                    message = error_msg)
                return response

            except Exception as e:
                error_msg  = "Backward request deserialization failed with unknown error {}".format(e)
                logger.error(error_msg)
                response = bittensor_pb2.TensorMessage(
                    version = bittensor.__version__, 
                    public_key = self.__keypair.public_key, 
                    code = bittensor.exceptions.RequestDeserializationExceptionCode,
                    message = error_msg)
                return response


            # Get grads by calling backward.
            try:
                inputs_dx = self._synapse.call_backward(inputs_x, grads_dy)

            except Exception as e:
                error_msg  = "Unkown exception when calling backward with error {}".format(e)
                logger.error(error_msg)
                response = bittensor_pb2.TensorMessage(
                    version = bittensor.__version__, 
                    public_key = self.__keypair.public_key, 
                    code = bittensor.exceptions.UnknownExceptionCode,
                    message = error_msg)
                return response

            # Deserialize response.
            try:
                inputs_dx_serialized = PyTorchSerializer.serialize_tensor(inputs_dx)

            except bittensor.exceptions.SerializationException as e:
                error_msg  = "Backward request serialization failed with error {} and inputs {}".format(e, inputs_dx)
                logger.error(error_msg)
                response = bittensor_pb2.TensorMessage(
                    version = bittensor.__version__, 
                    public_key = self.__keypair.public_key, 
                    code = bittensor.exceptions.RequestSerializationExceptionCode,
                    message = error_msg)
                return response

            except Exception as e:
                error_msg  = "Backward request deserialization failed with unknown error {}".format(e)
                logger.error(error_msg)
                response = bittensor_pb2.TensorMessage(
                    version = bittensor.__version__, 
                    public_key = self.__keypair.public_key, 
                    code = bittensor.exceptions.RequestDeserializationExceptionCode,
                    message = error_msg)
                return response

        # Final catch of unknown exceptions.
        except Exception as e:
            error_msg  = "Calling backward request failed with unknown error {}".format(e)
            logger.error(error_msg)
            response = bittensor_pb2.TensorMessage(
                version = bittensor.__version__, 
                public_key = self.__keypair.public_key, 
                code = bittensor.exceptions.UnknownExceptionCode,
                message = error_msg)
            return response

        # Finally build response with serialized outputs.
        finally:
            response = bittensor_pb2.TensorMessage(
                version = bittensor.__version__,
                public_key = self.__keypair.public_key,
                code = bittensor.exceptions.SuccessCode,
                message = "success",
                tensors = inputs_dx_serialized)
        
        return response