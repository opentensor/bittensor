import argparse
import sys
import torch
import traceback
import queue

from loguru import logger
from munch import Munch
from typing import List, Tuple

from bittensor import bittensor_pb2
from bittensor.synapse import Synapse
from bittensor.utils.ptp import ThreadPoolExecutor

class Nucleus ():
    r""" Processing core of a bittensor Neuron. Runs behind an Axon endpoint to process requests on the served synapse.
        The nucleus uses a prioritized thread pool to process requests according in priority. Priority is set by the Axon.
    """

    def __init__(self, config):
        r""" Initializes a nucleus backward and forward threading pools.
        """
        self._config = config
        self._forward_pool = ThreadPoolExecutor(maxsize = self._config.nucleus.queue_maxsize, max_workers=self._config.nucleus.max_workers)
        self._backward_pool = ThreadPoolExecutor(maxsize = self._config.nucleus.queue_maxsize, max_workers=self._config.nucleus.max_workers)

    def forward(self, synapse: Synapse, inputs: torch.Tensor, mode: bittensor_pb2.Modality, priority: float) -> Tuple[torch.FloatTensor, str, int]:
        r""" Accepts a synapse object with inputs and priority, submits them to the forward work pool
            and waits for a response from the threading future. Processing errors or timeouts result in
            error codes which propagate back to the calling Axon.

            Args:
                synapse (:obj:`bittensor.synapse.Synapse`, `required`): 
                    synapse to pass to the worker. Note: the synapse.call_forward must be thread safe. 
                inputs (:obj:`torch.Tensor`, `required`): 
                    tensor inputs to be passed to synapse.call_forward
                mode (:enum:`bittensor_pb2.Modality`, `required`):
                    input modality enum signaling between IMAGE, TEXT or TENSOR inputs.
                priority (`float`, `required`):
                    processing priority, a unique number from amongst current calls and less than sys.maxsize.
                    calls are processed in this order.
                    NOTE: priority must be unique amongst current calls.
            Returns:
                outputs (:obj:`torch.FloatTensor`, `required`): 
                    response from the synapse.call_forward call or None in the case of errors.
                message: (str, `required`): 
                    message associated with forward call, potentially error, or 'success'.
                code: (:obj:`bittensor_pb2.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
        """
        # Build future request.
        call_params = [synapse, inputs, mode]
        try:
            future = self._forward_pool.submit( fn = self._forward, call_params = call_params, priority = priority)
        except queue.Full:
            code = bittensor_pb2.ReturnCode.NucleusFull
            message = 'processing queue is full. try Backoff.'
            return None, message, code
        except Exception as e:
            message = 'Unknown error on nucleus submit with error {}'.format(e)
            logger.error(message)
            code = bittensor_pb2.ReturnCode.UnknownException
            return None, message, code

        # Try to get response or error on timeout.
        try:
            outputs = future.result (timeout = self._config.nucleus.queue_timeout)
            tensor = outputs[0]
            message = outputs[1]
            code = outputs[2]
        # Catch all Exception.
        # Errors in the synapse call are caught in _backward and returned with the corresponding code.
        except Exception as e:
            tensor = None
            message = 'timeout with error {}'.format(e)
            code = bittensor_pb2.ReturnCode.NucleusTimeout
        return tensor, message, code

    def backward(self, synapse: Synapse, inputs_x: torch.Tensor, grads_dy: torch.FloatTensor, mode: bittensor_pb2.Modality, priority: float) -> Tuple[torch.FloatTensor, str, int]:
        r""" Accepts a synapse object with tensor inputs, grad inputs, and priority. 
            Submits inputs to the backward work pool for processing and waits waits for a response.
            Processing errors or timeouts result in error codes which propagate back to the calling Axon.

            Args:
                synapse (:obj:`bittensor.synapse.Synapse`, `required`): 
                    synapse to pass to the worker. Note: the synapse.call_backward must be thread safe. 
                inputs_x (:obj:`torch.Tensor`, `required`): 
                    tensor inputs from a previous call to be passed to synapse.call_backward
                grads_dy (:obj:`torch.Tensor`, `required`): 
                    gradients associated wiht inputs for backward call.
                mode (:enum:`bittensor_pb2.Modality`, `required`):
                    input modality enum signaling between IMAGE, TEXT or TENSOR inputs.
                priority (`float`, `required`):
                    processing priority, a unique number from amongst current calls and less than sys.maxsize.
                    calls are processed in this order.
                    NOTE: priority must be unique amongst current calls.
            Returns:
                outputs (:obj:`torch.FloatTensor`, `required`): 
                    response from the synapse.call_backward call (i.e. inputs_dx) or None in the case of errors.
                message: (str, `required`): 
                    message associated with forward call, potentially error, or 'success'.
                code: (:obj:`bittensor_pb2.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
        """
        # Build call params and submit the task to the pool.
        call_params = [synapse, inputs_x, grads_dy, mode]
        future = self._backward_pool.submit( fn =  self._backward, call_params = call_params, priority = priority )
        # Recieve respoonse from the future or fail.
        try:
            outputs = future.result (timeout = self._config.nucleus.queue_timeout)
            tensor = outputs[0]
            message = outputs[1]
            code = outputs[2]
        # Catch all exception which returns a timeout code. 
        # Errors in the synapse call are caught in _backward and returned with the corresponding code.
        except Exception as e:
            tensor = None
            message = 'timeout with error {}'.format(e)
            code = bittensor_pb2.ReturnCode.NucleusTimeout
        return tensor, message, code

    def _forward(self, call_params: List) -> Tuple[torch.FloatTensor, str, int]:
        r""" Actual processing function for the forward call. The passed synapse.call_forward must be thread safe.

            Args:
                call_params (:obj:`List[bittensor.synapse.Synapse, inputs, mode]`, `required`): 
                    call params containing the synapse to be called and inputs with modality.
            Returns:
                outputs (:obj:`torch.FloatTensor`, `required`): 
                    response from the synapse.call_backward call (i.e. inputs_dx) or None in the case of errors.
                message: (str, `required`): 
                    message associated with forward call, potentially error, or 'success'.
                code: (:obj:`bittensor_pb2.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
        """
        synapse = call_params[0]
        inputs = call_params[1]
        mode = call_params[2]
        try:
            tensor = synapse.call_forward(inputs, mode)
            message = 'success'
            code = bittensor_pb2.ReturnCode.Success
                
        except NotImplementedError:
            tensor = None
            message = 'modality not implemented'
            code = bittensor_pb2.ReturnCode.NotImplemented

        except Exception as e:
            tensor = None
            message = 'Unknown error when calling Synapse forward with errr {}'.format(e)
            code = bittensor_pb2.ReturnCode.UnknownException

        return [tensor, message, code]

    def _backward(self, call_params: List):
        r""" Actual processing function for the backward call. The passed synapse.call_backward must be thread safe.

            Args:
                call_params (:obj:`List[bittensor.synapse.Synapse, inputs_x, grads_dy]`, `required`): 
                    call params containing the synapse to be called and inputs with grads.
            Returns:
                outputs (:obj:`torch.FloatTensor`, `required`): 
                    response from the synapse.call_backward call (i.e. inputs_dx) or None in the case of errors.
                message: (str, `required`): 
                    message associated with forward call, potentially error, or 'success'.
                code: (:obj:`bittensor_pb2.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
        """
       
        synapse = call_params[0]
        inputs_x = call_params[1]
        grads_dy = call_params[2]
        mode = call_params[3]
        try:
            tensor = synapse.grad(inputs_x, grads_dy, modality = mode)
            message = 'success'
            code = bittensor_pb2.ReturnCode.Success
        except Exception as e:
            tensor = None
            message = 'Unknown error when calling Synapse backward with errr {}'.format(e)
            code = bittensor_pb2.ReturnCode.UnknownException
        return [tensor, message, code]

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        r""" Adds this nucleus's command line arguments to the passed parser.
            Args:
                parser (:obj:`argparse.ArgumentParser`, `required`): 
                    parser argument to append args to.
        """
        parser.add_argument('--nucleus.max_workers', default=5, type=int, help='Nucleus priority queue workers.')
        parser.add_argument('--nucleus.queue_timeout', default=5, type=int, help='Nucleus future timeout.')
        parser.add_argument('--nucleus.queue_maxsize', default=1000, type=int, help='Maximum number of pending tasks allowed in the threading priority queue.')

    @staticmethod   
    def check_config(config: Munch):
        r""" Checks the passed config items for validity.
            Args:
                config (:obj:`munch.Munch, `required`): 
                    config to check.
        """
        pass

    def __del__(self):
        """ Calls nucleus stop for clean threadpool closure """
        self.stop()

    def stop(self):
        """ Safely shutsdown the forward and backward pool thread """
        self._forward_pool.shutdown()
        self._backward_pool.shutdown()


