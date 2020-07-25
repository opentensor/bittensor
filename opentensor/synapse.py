from opentensor import opentensor_pb2

import hivemind
import torch
import torch.nn as nn

class Synapse(nn.Module):
    """ Implementation of a opentensor.synapse. A single ip/port tensor processing unit """
    def __init__(self):
        super(Synapse, self).__init__()
        
        input_schema = self.input_schema()
        outputs_schema = self.outputs_schema()
        
        if outputs_schema is None:
            # run expert once to get outputs schema
            dummy_args = tuple(sample.make_empty(DUMMY_BATCH_SIZE) for sample in args_schema)
            dummy_kwargs = {key: sample.make_empty(DUMMY_BATCH_SIZE) for key, sample in kwargs_schema.items()}
            dummy_outputs = self.expert(*dummy_args, **dummy_kwargs)
            outputs_schema = nested_map(BatchTensorDescriptor.from_tensor, dummy_outputs)

    def input_schema(self) -> hivemind.BatchTensorDescriptor:
        raise NotImplementedError
    
    def output_schema(self) -> hivemind.BatchTensorDescriptor:
        return None  

    def _indef(self) -> opentensor_pb2.TensorDef:
        """ Returns the opentensor_pb2.TensorDef for the input """
        raise NotImplementedError

    def _outdef(self) -> opentensor_pb2.TensorDef:
        """ Returns the opentensor_pb2.TensorDef for the output """
        raise NotImplementedError
