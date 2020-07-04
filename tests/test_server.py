from opentensor import opentensor_pb2_grpc as proto_grpc
from opentensor import opentensor_pb2 as proto_pb2
from concurrent import futures

import grpc

class Opentensor(proto_grpc.OpentensorServicer):

    def __init__(self):
        pass

    def Fwd(self, context, request):
        response = proto_pb2.TensorMessage()
        return response
  

    def Bwd(self, contect, request):
        response = proto_pb2.TensorMessage()
        return response
 
def create_server():
    address = "[::]:8888"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    opentensor = Opentensor()
    proto_grpc.add_OpentensorServicer_to_server(opentensor, server)
    server.add_insecure_port(address)
    return server
    
def test_create():
    server = create_server()
    server.start()
    server.stop(0)

def test_client():

    server = create_server()
    server.start()

    address = "localhost:8888"
    channel = grpc.insecure_channel(address)
    stub = proto_grpc.OpentensorStub(channel)

    request = proto_pb2.TensorMessage()
    response = stub.Fwd(request)

    request = proto_pb2.TensorMessage()
    response = stub.Bwd(request)
    server.stop(0)

