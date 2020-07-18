from opentensor import opentensor_pb2_grpc as opentensor_grpc
from opentensor import opentensor_pb2
import opentensor

from loguru import logger
from typing import List

import os
import grpc
import torch


class Dendrite:
    def __init__(self, config: opentensor.Config):
        self._config = config
        
    def forward(self, x: List[torch.Tensor], synapses: List[opentensor_pb2.Synapse]):
        """ forward tensor processes """

        version = 1.0
        source_uid = self._config.identity.public_key()
        nounce = os.urandom(12)

        results = []
        for idx, tensor in enumerate(x):
            synapse = synapses[idx]

            # Create endpoint stub
            target_id = synapse.identity
            address = synapse.address
            # Loop back if the synapse is local.
            if address == self._config.remote_ip:
                address = 'localhost'
            channel = grpc.insecure_channel(address + ':' + synapse.port)
            stub = opentensor_grpc.OpentensorStub(channel)

            serialized_tensor = opentensor.Serializer.serialize(tensor)
            # TODO(const) The extra public_key is redundant.
            request = opentensor_pb2.TensorMessage(version=version,
                                                   neuron_key=source_uid,
                                                   source_id=source_uid,
                                                   target_id=synapse.identity,
                                                   nounce=nounce,
                                                   tensors=[serialized_tensor])

            try:
                response = stub.Fwd(request)
                out_tensor = opentensor.Serializer.deserialize(
                    response.tensors[0])
            except:
                out_tensor = opentensor.Serializer.zeros_for_def(
                    tensor, synapse.outdef)
            results.append(out_tensor)

        return results
