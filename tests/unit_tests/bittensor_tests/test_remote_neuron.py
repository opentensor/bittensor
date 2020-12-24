import grpc
import torchvision.transforms as transforms
import torch
import unittest
import bittensor
import pytest
import torchvision

from bittensor.config import Config
from bittensor.subtensor.interface import Keypair
from bittensor.dendrite import RemoteNeuron, _RemoteModuleCall
from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor import bittensor_pb2
from unittest.mock import MagicMock
import bittensor.serialization as serialization
from munch import Munch

config = {'neuron':
              {'datapath': 'data/', 'learning_rate': 0.01, 'momentum': 0.9, 'batch_size_train': 64,
               'batch_size_test': 64, 'log_interval': 10, 'sync_interval': 100, 'priority_interval': 100,
               'name': 'mnist', 'trial_id': '1608070667'},
          'synapse': {'target_dim': 10},
          'dendrite': {'key_dim': 100, 'topk': 10, 'stale_emit_filter': 10000, 'pass_gradients': True, 'timeout': 0.5,
                       'do_backoff': True, 'max_backoff': 100}, 'axon': {'local_port': 8091, 'external_ip': '191.97.53.53'},
          'nucleus': {'max_workers': 5, 'queue_timeout': 5, 'queue_maxsize': 1000},
          'metagraph': {'chain_endpoint': '206.189.254.5:12345', 'stale_emit_filter': 10000},
          'meta_logger': {'log_dir': 'data/'},
          'session': {'keyfile': None, 'keypair': None }
          }

config = Munch.fromDict(config)

config.dendrite.do_backoff = False
mnemonic = Keypair.generate_mnemonic()
keypair = Keypair.create_from_mnemonic(mnemonic)

neuron = bittensor_pb2.Neuron(
    version = bittensor.__version__,
    public_key = keypair.public_key,
    address = '0.0.0.0',
    port = 22424,
)
remote = RemoteNeuron (neuron, config, keypair)
channel = grpc.insecure_channel('localhost',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])          
stub = bittensor_grpc.BittensorStub(channel)

def test_remote_neuron_text():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, ops = remote.forward( x, bittensor_pb2.Modality.TEXT)
    print (out, ops)
    assert ops.item() == bittensor_pb2.ReturnCode.Unavailable
    assert list(out.shape) == [2, 4, bittensor.__network_dim__]

def test_remote_neuron_image():
    x = torch.tensor([ [ [ [ [ 1 ] ] ] ] ])
    out, ops = remote.forward( x, bittensor_pb2.Modality.IMAGE)
    assert ops.item() == bittensor_pb2.ReturnCode.Unavailable
    assert list(out.shape) == [1, 1, bittensor.__network_dim__]

def test_remote_neuron_tensor():
    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops = remote.forward( x, bittensor_pb2.Modality.TENSOR)
    assert ops.item() == bittensor_pb2.ReturnCode.Unavailable
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]

def test_remote_neuron_request_empty():
    x = torch.tensor([])
    out, ops = remote.forward( x, bittensor_pb2.Modality.TEXT)
    assert ops.item() == bittensor_pb2.ReturnCode.EmptyRequest
    assert list(out.shape) == [0]

def test_remote_neuron_mock_server():
    y = torch.rand(3, 3, bittensor.__network_dim__)
    
    serializer = serialization.get_serializer( serialzer_type = bittensor_pb2.Serializer.PICKLE )
    y_serialized = serializer.serialize(y, modality = bittensor_pb2.Modality.TENSOR, from_type = bittensor_pb2.TensorType.TORCH)
            
    mock_return_val = bittensor_pb2.TensorMessage(
            version = bittensor.__version__,
            public_key = keypair.public_key,
            return_code = bittensor_pb2.ReturnCode.Success,
            tensors = [y_serialized])

    stub.Forward = MagicMock( return_value=mock_return_val )
    remote.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops = remote.forward(x, bittensor_pb2.Modality.TENSOR)
    assert ops.item() == bittensor_pb2.ReturnCode.Success
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_remote_neuron_mock_server_deserialization_error():
    y = dict() # bad response
    mock_return_val = bittensor_pb2.TensorMessage(
            version = bittensor.__version__,
            public_key = keypair.public_key,
            return_code = bittensor_pb2.ReturnCode.Success,
            tensors = [y])

    stub.Forward = MagicMock( return_value=mock_return_val )
    remote.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops = remote.forward(x, bittensor_pb2.Modality.TENSOR)
    assert ops.item() == bittensor_pb2.ReturnCode.ResponseDeserializationException
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_remote_neuron_mock_server_shape_error():
    y = torch.rand(1, 3, bittensor.__network_dim__)

    serializer = serialization.get_serializer( serialzer_type = bittensor_pb2.Serializer.PICKLE )
    y_serialized = serializer.serialize(y, modality = bittensor_pb2.Modality.TENSOR, from_type = bittensor_pb2.TensorType.TORCH)
   
    mock_return_val = bittensor_pb2.TensorMessage(
            version = bittensor.__version__,
            public_key = keypair.public_key,
            return_code = bittensor_pb2.ReturnCode.Success,
            tensors = [y_serialized])

    stub.Forward = MagicMock( return_value=mock_return_val )
    remote.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops = remote.forward(x, bittensor_pb2.Modality.TENSOR)
    assert ops.item() == bittensor_pb2.ReturnCode.ResponseShapeException
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]

if __name__ == "__main__":
    test_remote_neuron_text ()
    test_remote_neuron_image ()
    test_remote_neuron_tensor ()
    test_remote_neuron_request_empty ()
    test_remote_neuron_mock_server ()
    test_remote_neuron_mock_server_deserialization_error ()
    test_remote_neuron_mock_server_shape_error ()