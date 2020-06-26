from opentensor_proto import opentensor_pb2_grpc as proto_grpc
from opentensor_proto import opentensro_pb2 as proto_pb2

from concurrent import futures

class Server(proto_grpc.OpentensorServicer):

    def __init__(self):
        pass

def main(config):
    address = "[::]:8888"
    opentensor_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    peerstore = Peerstore(config)
    proto_grpc.add_PeerstoreServicer_to_server(opentensor_server, server)
    server.add_insecure_port(address)

    server.start()
    server.stop(0)

if __name__ == '__main__':
    main()
