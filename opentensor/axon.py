from opentensor import opentensor_pb2_grpc as opentensor_grpc
from opentensor import opentensor_pb2
import opentensor

class Axon(opentensor_grpc.OpentensorServicer):
    def __init__(self, metagraph):
        self._metagraph = metagraph
    
    def Fwd(self, request, context):
        version = request.version
        public_key = request.public_key
        source_id = request.source_id
        target_id = request.target_id
        #nounce = request.nounce
        tensor = request.tensors[0]

        tensor = opentensor.Serializer.deserialize(tensor)
        tensor = self._metagraph.Fwd(source_id, target_id, tensor)  
        tensor = opentensor.Serializer.serialize(tensor)
        response = opentensor_pb2.TensorMessage(
            version = version,
            public_key = self._metagraph.identity.public_key(),
            source_id = target_id,
            target_id = source_id,
            tensors = [tensor]
        )
        return response 
    
    def Bwd(self, request, context):
        self._metagraph.Bwd(request, context)


