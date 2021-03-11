

import torch
import pytest
import time
from munch import Munch
from unittest.mock import MagicMock
from torch.utils.tensorboard import SummaryWriter
import bittensor

dendrite = bittensor.dendrite.Dendrite()
neuron_a = bittensor.proto.Neuron(
    version = bittensor.__version__,
    public_key = "A",
    address = '0',
    port = 1,
)
neuron_b = bittensor.proto.Neuron(
    version = bittensor.__version__,
    public_key = "B",
    address = '0',
    port = 1,
)

def test_create_receptors():
    receptor_a = dendrite.get_receptor_for_neuron( neuron_a ) 
    receptor_b = dendrite.get_receptor_for_neuron( neuron_b )
    assert receptor_a.neuron.public_key == 'A'
    assert receptor_b.neuron.public_key == 'B'
    assert dendrite.getReceptors()['A'].neuron.public_key == 'A'
    assert dendrite.getReceptors()['B'].neuron.public_key == 'B'

def test_dendrite_foward():
    receptor_a = dendrite.get_receptor_for_neuron( neuron_a ) 
    receptor_b = dendrite.get_receptor_for_neuron( neuron_b )

    receptor_a.forward = MagicMock(return_value = [torch.tensor([1]), [0], '']) 
    receptor_b.forward = MagicMock(return_value = [torch.tensor([1]), [0], '']) 
    outputs, codes, messages = dendrite.forward(
        neurons = [neuron_a, neuron_b],
        inputs = [torch.tensor([1,1,1]), torch.tensor([1,1,2])],
        mode = bittensor.proto.Modality.TEXT
    )
    assert len(outputs) == 2
    assert len(codes) == 2
    assert outputs[0] == torch.tensor([1])
    assert codes[0] == [0]

def test_dendrite_backward():
    receptor_a = dendrite.get_receptor_for_neuron( neuron_a ) 
    receptor_b = dendrite.get_receptor_for_neuron( neuron_b )
    receptor_a.backward = MagicMock(return_value = [torch.tensor([1]), [0], '']) 
    receptor_b.backward = MagicMock(return_value = [torch.tensor([1]), [0], '']) 
    outputs, codes, messages = dendrite.backward(
        neurons = [neuron_a, neuron_b],
        inputs = [torch.tensor([1,1,1]), torch.tensor([1,1,2])],
        grads = [torch.tensor([1,1,1]), torch.tensor([1,1,2])],
        codes = [[0], [0]],
        mode = bittensor.proto.Modality.TEXT
    )
    assert len(outputs) == 2
    assert len(codes) == 2
    assert outputs[0] == torch.tensor([1])
    assert codes[0] == [0]

def test_dendrite_to_string():
    dendrite.toString()

def test_dendrite_to_tensorboard():
    summary_writer = SummaryWriter()
    dendrite.toTensorboard(summary_writer, 1)

def test_dendrite_full_to_string():
    dendrite.fullToString()


if __name__ == "__main__":
    test_create_receptors()
    test_dendrite_foward()