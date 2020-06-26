from opentensor_proto import opentensor_pb2_grpc as proto_grpc
from opentensor_proto import opentensor_pb2 as proto_pb2
from concurrent import futures

import grpc

class Opentensor(proto_grpc.OpentensorServicer):

    def __init__(self):
        pass

def test_create():
    address = "[::]:8888"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    opentensor = Opentensor()
    proto_grpc.add_OpentensorServicer_to_server(opentensor, server)
    server.add_insecure_port(address)

    server.start()
    server.stop(0)

