from opentensor import opentensor_pb2_grpc as opentensor_grpc
from opentensor import opentensor_pb2
import opentensor

from loguru import logger
from typing import List

import os
import grpc
import torch


class Dendrite:
    def __init__(self, identity):
        self._identity = identity

    def forward(self, x: List[torch.Tensor], axons: List[opentensor_pb2.Axon]):
        """ forward tensor processes """

        version = 1.0
        source_uid = b''
        nounce = os.urandom(12)

        results = []
        for idx, tensor in enumerate(x):
            axon = axons[idx]

            # Create endpoint stub
            target_id = axon.identity
            address = axon.address + ":" + axon.port
            channel = grpc.insecure_channel(address)
            stub = opentensor_grpc.OpentensorStub(channel)

            tensor = opentensor.Serializer.serialize(tensor)
            # TODO(const) The extra public_key is redundant.
            request = opentensor_pb2.TensorMessage(version=version,
                                                   neuron_key=axon.neuron_key,
                                                   source_id=axon.identity,
                                                   target_id=target_id,
                                                   nounce=nounce,
                                                   tensors=[tensor])

            logger.info('->', axon)
            response = stub.Fwd(request)
            tensor = opentensor.Serializer.deserialize(response.tensors[0])
            results.append(tensor)

        return results
