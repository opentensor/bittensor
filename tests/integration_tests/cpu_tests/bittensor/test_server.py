import bittensor
from concurrent import futures

import grpc


class Axon(bittensor.grpc.BittensorServicer):

    def __init__(self):
        pass

    def Forward(self, context, request):
        response = bittensor.proto.TensorMessage()
        return response

    def Backward(self, contect, request):
        response = bittensor.proto.TensorMessage()
        return response


def create_server():
    address = "[::]:8812"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    axon = bittensor.Axon()
    bittensor.grpc.add_BittensorServicer_to_server(axon, server)
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
    stub = bittensor.grpc.BittensorStub(channel)

    request = bittensor.proto.TensorMessage()
    stub.Forward(request)

    request = bittensor.proto.TensorMessage()
    stub.Backward(request)
    server.stop(0)
