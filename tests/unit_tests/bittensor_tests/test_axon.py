from unittest.mock import MagicMock
from bittensor.config import Config
from bittensor.nucleus import Nucleus
from bittensor.subtensor.interface import Keypair
from bittensor.synapse import Synapse
from bittensor import bittensor_pb2
import bittensor.serialization as serialization
import bittensor.utils.serialization_utils as serialization_utils
import bittensor
import unittest
import random
import torch
from munch import Munch

config = {'session':
              {'datapath': 'data/', 'learning_rate': 0.01, 'momentum': 0.9, 'batch_size_train': 64,
               'batch_size_test': 64, 'log_interval': 10, 'sync_interval': 100, 'priority_interval': 100,
               'name': 'mnist', 'trial_id': '1608070667'},
          'synapse': {'target_dim': 10},
          'dendrite': {'key_dim': 100, 'topk': 10, 'stale_emit_filter': 10000, 'pass_gradients': True, 'timeout': 0.5,
                       'do_backoff': True, 'max_backoff': 100}, 'axon': {'local_port': 8091, 'external_ip': '191.97.53.53', 'max_workers': 5, 'max_gradients': 1000},
          'nucleus': {'max_workers': 5, 'queue_timeout': 5, 'queue_maxsize': 1000},
          'metagraph': {'chain_endpoint': '206.189.254.5:12345', 'stale_emit_filter': 10000},
          'meta_logger': {'log_dir': 'data/'},
          'neuron': {'keyfile': None, 'keypair': None }
          }

config = Munch.fromDict(config)
mnemonic = Keypair.generate_mnemonic()
keypair = Keypair.create_from_mnemonic(mnemonic)
config.neuron.keypair = keypair

nucleus = Nucleus(config)
axon = bittensor.axon.Axon(config, nucleus)
synapse = Synapse(config, None)

def test_serve():
    assert axon.synapse == None
    for _ in range(0, 10):
        axon.serve(synapse)
    assert axon.synapse != None


def test_forward_not_implemented():
    axon.serve(synapse)
    nucleus.forward = MagicMock(return_value=[None, 'not implemented', bittensor_pb2.ReturnCode.NotImplemented])
    x = torch.rand(3, 3, bittensor.__network_dim__)

    serializer = serialization.get_serializer( serialzer_type = bittensor_pb2.Serializer.MSGPACK )
    x_serialized = serializer.serialize(x, modality = bittensor_pb2.Modality.TENSOR, from_type = bittensor_pb2.TensorType.TORCH)
  
    request = bittensor_pb2.TensorMessage(
        version = bittensor.__version__,
        public_key = keypair.public_key,
        tensors=[x_serialized]
    )
    response = axon.Forward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.NotImplemented


def test_forward_not_serving():
    axon.synapse = None
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
    )
    response = axon.Forward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.NotServingSynapse


def test_empty_forward_request():
    axon.serve(synapse)
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
    )
    response = axon.Forward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.EmptyRequest


def test_forward_deserialization_error():
    axon.serve(synapse)
    x = dict()
    y = dict()  # Not tensors that can be deserialized.
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
        tensors=[x, y]
    )
    response = axon.Forward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.RequestDeserializationException


def test_forward_success():
    axon.synapse = synapse
    x = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor_pb2.Serializer.MSGPACK )
    x_serialized = serializer.serialize(x, modality = bittensor_pb2.Modality.TENSOR, from_type = bittensor_pb2.TensorType.TORCH)
  
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
        tensors=[x_serialized]
    )
    nucleus.forward = MagicMock(return_value=[x, 'success', bittensor_pb2.ReturnCode.Success])

    response = axon.Forward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.Success
    assert len(response.tensors) == 1
    assert response.tensors[0].shape == [3, 3, bittensor.__network_dim__]
    assert serialization_utils.bittensor_dtype_to_torch_dtype(response.tensors[0].dtype) == torch.float32


def test_backward_not_serving():
    axon.synapse = None
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
    )
    response = axon.Backward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.NotServingSynapse


def test_empty_backward_request():
    axon.serve(synapse)
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
    )
    response = axon.Backward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.InvalidRequest


def test_single_item_backward_request():
    axon.serve(synapse)
    x = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor_pb2.Serializer.MSGPACK )
    x_serialized = serializer.serialize(x, modality = bittensor_pb2.Modality.TENSOR, from_type = bittensor_pb2.TensorType.TORCH)
  
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
        tensors=[x_serialized]
    )
    response = axon.Backward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.InvalidRequest


def test_backward_deserialization_error():
    axon.serve(synapse)
    x = dict()
    y = dict()  # Not tensors that can be deserialized.
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
        tensors=[x, y]
    )
    response = axon.Backward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.RequestDeserializationException


def test_backward_success():
    axon.serve(synapse)
    x = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor_pb2.Serializer.MSGPACK )
    x_serialized = serializer.serialize(x, modality = bittensor_pb2.Modality.TENSOR, from_type = bittensor_pb2.TensorType.TORCH)
  
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
        tensors=[x_serialized, x_serialized]
    )
    nucleus.backward = MagicMock(return_value=[x, 'success', bittensor_pb2.ReturnCode.Success])
    response = axon.Backward(request, None)

    assert response.return_code == bittensor_pb2.ReturnCode.Success
    assert len(response.tensors) == 1
    assert response.tensors[0].shape == [3, 3, bittensor.__network_dim__]
    assert serialization_utils.bittensor_dtype_to_torch_dtype(response.tensors[0].dtype) == torch.float32



if __name__ == "__main__":    
    test_backward_success()