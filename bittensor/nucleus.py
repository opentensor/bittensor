import argparse
import torch
from munch import Munch
import sys, traceback
from bittensor.synapse import Synapse
from bittensor.utils.priority_thread_pool import PriorityThreadPoolExecutor
from bittensor import bittensor_pb2
from typing import List, Tuple

class Nucleus ():
    def __init__(self, config):
        self._forward_pool = PriorityThreadPoolExecutor(max_workers=config.nucleus.max_workers)
        self._backward_pool = PriorityThreadPoolExecutor(max_workers=config.nucleus.max_workers)

    def __del__(self):
        self._forward_pool.shutdown()
        self._backward_pool.shutdown()

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--nucleus.max_workers', default=5, type=int, 
                            help='Nuclesu priority queue workers.')
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        return config

    def forward(self, synapse: Synapse, inputs: torch.Tensor, mode: bittensor_pb2.Modality, priority: int) -> Tuple[torch.FloatTensor, str, int]:
        
        call_params = [synapse, inputs, mode]
        future = self._forward_pool.submit( fn = self._forward, call_params = call_params, priority = priority)
        try:
            outputs = future.result (timeout = 1)
            tensor = outputs[0]
            message = outputs[1]
            code = outputs[2]

        except Exception as e:
            tensor = None
            message = 'timeout with error {}'.format(e)
            traceback.print_exc(file=sys.stdout)
            code = bittensor_pb2.ReturnCode.NucleusTimeout

        return tensor, message, code

    def backward(self, synapse: Synapse, inputs_x: torch.Tensor, grads_dy: torch.FloatTensor, priority: int) -> Tuple[torch.FloatTensor, str, int]:

        
        call_params = [synapse, inputs_x, grads_dy]
        future = self._backward_pool.submit( fn =  self._backward, call_params = call_params, priority = priority )
        try:
            outputs = future.result (timeout = 1)
            tensor = outputs[0]
            message = outputs[1]
            code = outputs[2]

        except Exception as e:
            tensor = None
            message = 'timeout with error {}'.format(e)
            traceback.print_exc(file=sys.stdout)
            code = bittensor_pb2.ReturnCode.NucleusTimeout
        return tensor, message, code

    def _forward(self, call_params: List):
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
            code = bittensor_pb2.ReturnCode.NotImplementedError

        except Exception as e:
            tensor = None
            message = 'Unknown error when calling Synapse forward with errr {}'.format(e)
            #traceback.print_exc(file=sys.stdout)
            code = bittensor_pb2.ReturnCode.UnknownException

        return [tensor, message, code]

    def _backward(self, call_params: List):
       
        synapse = call_params[0]
        inputs_x = call_params[1]
        grads_dy = call_params[2]
        try:
            tensor = synapse.call_backward(inputs_x, grads_dy)
            message = 'success'
            code = bittensor_pb2.ReturnCode.Success

        except Exception as e:
            tensor = None
            message = 'Unknown error when calling Synapse backward with errr {}'.format(e)
            code = bittensor_pb2.ReturnCode.UnknownException

        return [tensor, message, code]



