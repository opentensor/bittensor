from unittest.mock import MagicMock
from bittensor.serializer import PyTorchSerializer, torch_dtype_to_bittensor_dtype, bittensor_dtype_to_torch_dtype
from bittensor import bittensor_pb2
import bittensor
import unittest
import random
import torch


class TestAxon(unittest.TestCase):
    def setUp(self):
        axon_port = random.randint(8000, 9000)
        bp_host = 'localhost'
        bp_port = metagraph_port
        self._config = Config.load(
                                            axon_port = axon_port,
                                            metagraph_port = metagraph_port,
                                            bp_host = bp_host,
                                            bp_port = bp_port
                                        )
        self._config.log()
        dendrite = bittensor.Dendrite(self._config)
        self.axon = bittensor.Axon(self._config)
        meta = bittensor.Metagraph(self._config)

        private_key = bittensor.crypto.Crypto.generate_private_ed25519()
        public_key = bittensor.crypto.Crypto.public_key_from_private(private_key)
        synapse_config = bittensor.SynapseConfig(
            synapse_key=bittensor.crypto.Crypto.public_key_to_string(public_key)
            )
        self.synapse = bittensor.Synapse(synapse_config, dendrite, meta)
        self.neuron_key=bittensor.crypto.Crypto.public_key_to_string(public_key)
    
    def test_serve(self):
        assert len(self.axon._local_synapses) == 0

        for _ in range(0, 10):
            self.axon.serve(self.synapse)
        
        # No matter how many identical synapses we add, we should still have only one counted.
        assert len(self.axon._local_synapses) == 1

    def test_forward(self):
        request = bittensor_pb2.TensorMessage()

        # Check for null response by sending a request with no tensors in it
        response = self.axon.Forward(request, None)
        assert response == bittensor_pb2.TensorMessage(
                                version=bittensor.__version__,
                                neuron_key=self._config.neuron_key,
                                synapse_key=request.synapse_key
                            )

        # Let's add a synapse and call forward on the axon
        self.axon.serve(self.synapse)
        x = torch.rand(3, 3, bittensor.__network_dim__)
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            neuron_key=self.neuron_key,
            synapse_key=self.synapse.synapse_key(),
            tensors=[PyTorchSerializer.serialize_tensor(x)]
        )

        self.axon._local_synapses[request.synapse_key].call_forward = MagicMock(return_value=x)
        response = self.axon.Forward(request, None)
        assert response.tensors[0].shape == [3, 3, bittensor.__network_dim__]
        assert bittensor_dtype_to_torch_dtype(response.tensors[0].dtype) == torch.float32
    
    def test_backward(self):
        # Let's add a synapse and call backward on the axon
        self.axon.serve(self.synapse)
        x = torch.rand(3, 3, bittensor.__network_dim__)
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            neuron_key=self.neuron_key,
            synapse_key=self.synapse.synapse_key(),
            tensors=[PyTorchSerializer.serialize_tensor(x)]
        )

        self.axon._local_synapses[request.synapse_key].call_forward = MagicMock(return_value=x)
        response = self.axon.Backward(request, None)

        # We should get back a null response since the number of tensors does not match.
        assert response ==  bittensor_pb2.TensorMessage(
                version=bittensor.__version__,
                neuron_key=self._config.neuron_key,
                synapse_key=request.synapse_key)

        x2 = torch.rand(3, 3, bittensor.__network_dim__)
        
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            neuron_key=self.neuron_key,
            synapse_key=self.synapse.synapse_key(),
            tensors=[PyTorchSerializer.serialize_tensor(x), PyTorchSerializer.serialize_tensor(x2)]
        )

        response = self.axon.Backward(request, None)

        assert len(response.tensors) == 1
        assert response.tensors[0].shape == [1,1]
        assert bittensor_dtype_to_torch_dtype(response.tensors[0].dtype) == torch.float32
  
