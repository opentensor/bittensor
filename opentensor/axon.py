from opentensor import opentensor_pb2_grpc as opentensor_grpc
from opentensor import opentensor_pb2
import opentensor


class Axon(opentensor_grpc.OpentensorServicer):
    def __init__(self, metagraph):
        self._metagraph = metagraph

        self._axon_address = 'localhost'
        self._axon_port = str(random.randint(8000, 30000))
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        opentensor_grpc.add_OpentensorServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' + self._axon_port)

        self._thread = None

    def __del__(self):
        self.stop()

    def start(self):
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self):
        self._server.stop(0)

    def gossip(self, request, context):
        self._metagraph.recv_gossip(request)

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
            version=version,
            public_key=self._metagraph.identity.public_key(),
            source_id=target_id,
            target_id=source_id,
            tensors=[tensor])
        return response

    def Bwd(self, request, context):
        self._metagraph.Bwd(request, context)
