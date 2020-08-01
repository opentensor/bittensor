from bittensor import bittensor_pb2_grpc as proto_grpc
from bittensor import bittensor_pb2 as proto_pb2
from concurrent import futures

import grpc


class Axon(proto_grpc.BittensorServicer):

    def __init__(self):
        pass

    def Forward(self, context, request):
        response = proto_pb2.TensorMessage()
        return response

    def Backward(self, contect, request):
        response = proto_pb2.TensorMessage()
        return response


def create_server():
    address = "[::]:8812"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    axon = Axon()
    proto_grpc.add_BittensorServicer_to_server(axon, server)
    server.add_insecure_port(address)
    return server


def test_create():
    server = create_server()
    server.start()
    server.stop(0)


def test_client():

    server = create_server()
    server.start()

    address = "localhost:8812"
    channel = grpc.insecure_channel(address)
    stub = proto_grpc.BittensorStub(channel)

    request = proto_pb2.TensorMessage()
    response = stub.Forward(request)

    request = proto_pb2.TensorMessage()
    response = stub.Backward(request)
    server.stop(0)
