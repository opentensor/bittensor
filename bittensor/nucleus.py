'''
The MIT License (MIT)
Copyright © 2021 Opentensor.ai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
'''
import argparse
import sys
import torch
import traceback
import queue

from loguru import logger
from munch import Munch
from typing import List, Tuple

import bittensor 
import bittensor.config as config_utils
import bittensor.utils.ptp as ptp

class Nucleus ():
    r""" Processing core of a bittensor Neuron. Runs behind an Axon endpoint to process requests on the served synapse.
        The nucleus uses a prioritized thread pool to process requests according in priority. Priority is set by the Axon.
    """

    def __init__(self, config: Munch = None, wallet: 'bittensor.wallet.Wallet' = None, metagraph: 'metagraph.Metagraph' = None):
        r""" Initializes a new tensor processing backend
            Args:
                config (:obj:`Munch`, `optional`): 
                    nucleus.Nucleus.config()
                wallet (:obj:`bittensor.nucleus.Nucleus`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                metagraph (:obj:`bittensor.metagraph.Metagraph`, `optional`):
                    bittensor network metagraph.
        """
        if config == None:
            config = Nucleus.build_config()
        self.config = config

        if wallet == None:
            wallet = bittensor.wallet.Wallet(self.config)
        self.wallet = wallet

        if metagraph == None:
            metagraph = bittensor.metagraph.Metagraph( config = self.config, wallet = self.wallet )
        self.metagraph = metagraph

        self._forward_pool = ptp.ThreadPoolExecutor(maxsize = self.config.nucleus.queue_maxsize, max_workers=self.config.nucleus.max_workers)
        self._backward_pool = ptp.ThreadPoolExecutor(maxsize = self.config.nucleus.queue_maxsize, max_workers=self.config.nucleus.max_workers)

    @staticmethod   
    def build_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        Nucleus.add_args(parser) 
        config = config_utils.Config.to_config(parser); 
        Nucleus.check_config(config)
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        r""" Adds this nucleus's command line arguments to the passed parser.
            Args:
                parser (:obj:`argparse.ArgumentParser`, `required`): 
                    parser argument to append args to.
        """
        bittensor.metagraph.Metagraph.add_args( parser ) # Add wallet args from within metagraph.
        try:
            # Can be called twice.
            parser.add_argument('--nucleus.max_workers', default=5, type=int, 
                    help='''The maximum number of outstanding nucleuss priority queue workers.
                            Adding additional work to the work queue past this point does not trigger additional thread creation.''')
            parser.add_argument('--nucleus.queue_timeout', default=5, type=int, 
                    help='''Nucleus future timeout. Work futures timout after 5 second and return a null response.''')
            parser.add_argument('--nucleus.queue_maxsize', default=1000, type=int, 
                    help=''' The maximum number of pending tasks allowed in the threading priority queue. 
                            Adding additional work to the work queue blocks until space becomes available.''')
        except:
            pass

    @staticmethod   
    def check_config(config: Munch):
        r""" Checks the passed config items for validity.
            Args:
                config (:obj:`munch.Munch, `required`): 
                    config to check.
        """
        bittensor.metagraph.Metagraph.check_config( config )

    def forward(self, synapse: 'bittensor.synapse.Synapse', inputs: torch.Tensor, mode: bittensor.proto.Modality, priority: float) -> Tuple[torch.FloatTensor, str, int]:
        r""" Accepts a synapse object with inputs and priority, submits them to the forward work pool
            and waits for a response from the threading future. Processing errors or timeouts result in
            error codes which propagate back to the calling Axon.

            Args:
                synapse (:obj:`bittensor.synapse.synapse.Synapse`, `required`): 
                    synapse to pass to the worker. Note: the synapse.call_forward must be thread safe. 
                inputs (:obj:`torch.Tensor`, `required`): 
                    tensor inputs to be passed to synapse.call_forward
                mode (:enum:`bittensor.proto.Modality`, `required`):
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
                code: (:obj:`bittensor.proto.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
        """
        # Build future request.
        call_params = [synapse, inputs, mode]
        try:
            future = self._forward_pool.submit( fn = self._forward, call_params = call_params, priority = priority)
        except queue.Full:
            code = bittensor.proto.ReturnCode.NucleusFull
            message = 'processing queue is full. try Backoff.'
            return None, message, code
        except Exception as e:
            message = 'Unknown error on nucleus submit with error {}'.format(e)
            logger.error(message)
            code = bittensor.proto.ReturnCode.UnknownException
            return None, message, code

        # Try to get response or error on timeout.
        try:
            outputs = future.result (timeout = self.config.nucleus.queue_timeout)
            tensor = outputs[0]
            message = outputs[1]
            code = outputs[2]
        # Catch all Exception.
        # Errors in the synapse call are caught in _backward and returned with the corresponding code.
        except Exception as e:
            tensor = None
            message = 'timeout with error {}'.format(e)
            code = bittensor.proto.ReturnCode.NucleusTimeout
        return tensor, message, code

    def backward(self, synapse: 'bittensor.synapse.Synapse', inputs_x: torch.Tensor, grads_dy: torch.FloatTensor, mode: bittensor.proto.Modality, priority: float) -> Tuple[torch.FloatTensor, str, int]:
        r""" Accepts a synapse object with tensor inputs, grad inputs, and priority. 
            Submits inputs to the backward work pool for processing and waits waits for a response.
            Processing errors or timeouts result in error codes which propagate back to the calling Axon.

            Args:
                synapse (:obj:`bittensor.synapse.synapse.Synapse`, `required`): 
                    synapse to pass to the worker. Note: the synapse.call_backward must be thread safe. 
                inputs_x (:obj:`torch.Tensor`, `required`): 
                    tensor inputs from a previous call to be passed to synapse.call_backward
                grads_dy (:obj:`torch.Tensor`, `required`): 
                    gradients associated wiht inputs for backward call.
                mode (:enum:`bittensor.proto.Modality`, `required`):
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
                code: (:obj:`bittensor.proto.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
        """
        # Build call params and submit the task to the pool.
        call_params = [synapse, inputs_x, grads_dy, mode]
        future = self._backward_pool.submit( fn =  self._backward, call_params = call_params, priority = priority )
        # Recieve respoonse from the future or fail.
        try:
            outputs = future.result (timeout = self.config.nucleus.queue_timeout)
            tensor = outputs[0]
            message = outputs[1]
            code = outputs[2]
        # Catch all exception which returns a timeout code. 
        # Errors in the synapse call are caught in _backward and returned with the corresponding code.
        except Exception as e:
            tensor = None
            message = 'timeout with error {}'.format(e)
            code = bittensor.proto.ReturnCode.NucleusTimeout
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
                code: (:obj:`bittensor.proto.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
        """
        synapse = call_params[0]
        inputs = call_params[1]
        mode = call_params[2]
        try:
            tensor = synapse.call_forward(inputs, mode)
            message = 'success'
            code = bittensor.proto.ReturnCode.Success
                
        except NotImplementedError:
            tensor = None
            message = 'modality not implemented'
            code = bittensor.proto.ReturnCode.NotImplemented

        except Exception as e:
            tensor = None
            message = 'Unknown error when calling synapse.Synapse forward with errr {}, {}'.format(e, traceback.format_exc())
            code = bittensor.proto.ReturnCode.UnknownException

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
                code: (:obj:`bittensor.proto.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
        """
       
        synapse = call_params[0]
        inputs_x = call_params[1]
        grads_dy = call_params[2]
        mode = call_params[3]
        try:
            tensor = synapse.grad(inputs_x, grads_dy, modality = mode)
            message = 'success'
            code = bittensor.proto.ReturnCode.Success
        except Exception as e:
            tensor = None
            message = 'Unknown error when calling synapse.Synapse backward with errr {}'.format(e)
            code = bittensor.proto.ReturnCode.UnknownException
        return [tensor, message, code]

    def __del__(self):
        """ Calls nucleus stop for clean threadpool closure """
        self.stop()

    def stop(self):
        """ Safely shutsdown the forward and backward pool thread """
        self._forward_pool.shutdown()
        self._backward_pool.shutdown()


