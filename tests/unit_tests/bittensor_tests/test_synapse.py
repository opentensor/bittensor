import torch
import bittensor

def test_synapse():
    synapse = bittensor.synapse.Synapse()
    del synapse

def test_synapse_deepcopy():
    synapse = bittensor.synapse.Synapse()
    synapse2 = synapse.deepcopy()

class MockSynapse(bittensor.synapse.Synapse):
    def forward_text(self, text: torch.LongTensor) -> torch.FloatTensor:
        return torch.tensor([1])

    def forward_image(self, images: torch.FloatTensor) -> torch.FloatTensor:
        return torch.tensor([2])

    def forward_tensor(self, tensors: torch.FloatTensor) -> torch.FloatTensor:
        return torch.tensor([3])

def test_synapse_text():
    synapse = MockSynapse()
    response = synapse.call_forward(torch.tensor([1]), bittensor.proto.Modality.TEXT)
    assert response == torch.tensor([1])

def test_synapse_image():
    synapse = MockSynapse()
    response = synapse.call_forward(torch.tensor([1]), bittensor.proto.Modality.IMAGE)
    assert response == torch.tensor([2])

def test_synapse_tensor():
    synapse = MockSynapse()
    response = synapse.call_forward(torch.tensor([1]), bittensor.proto.Modality.TENSOR)
    assert response == torch.tensor([3])

def test_synapse_text_no_grad():
    synapse = MockSynapse()
    response = synapse.call_forward(torch.tensor([1]), bittensor.proto.Modality.TEXT, no_grad=False)
    assert response == torch.tensor([1])

def test_synapse_image_no_grad():
    synapse = MockSynapse()
    response = synapse.call_forward(torch.tensor([1]), bittensor.proto.Modality.IMAGE, no_grad=False)
    assert response == torch.tensor([2])

def test_synapse_tensor_no_grad():
    synapse = MockSynapse()
    response = synapse.call_forward(torch.tensor([1]), bittensor.proto.Modality.TENSOR, no_grad=False)
    assert response == torch.tensor([3])

class MultiplicationSynapse(bittensor.synapse.Synapse):
    def __init__(self):
        super().__init__()
        self.weight = torch.autograd.Variable(torch.tensor([2.0]))
    def forward_tensor(self, tensors: torch.FloatTensor) -> torch.FloatTensor:
        return tensors * self.weight

def test_synapse_multiplication():
    synapse = MultiplicationSynapse()
    response = synapse.call_forward(torch.tensor([1]), bittensor.proto.Modality.TENSOR)
    assert response == torch.tensor([2])

def test_synapse_grad():
    synapse = MultiplicationSynapse()
    grad_dy = torch.autograd.Variable(torch.tensor([1.0]))
    inputs = torch.autograd.Variable(torch.tensor([1.0]), requires_grad=True)
    response = synapse.grad(inputs, grad_dy, bittensor.proto.Modality.TENSOR)
    assert torch.isclose(response, torch.tensor([2.0]))

def test_synapse_backward():
    synapse = MultiplicationSynapse()
    grad_dy = torch.autograd.Variable(torch.tensor([1.0]))
    inputs = torch.autograd.Variable(torch.tensor([1.0]), requires_grad=True)
    synapse.backward(inputs, grad_dy, bittensor.proto.Modality.TENSOR)

test_synapse_grad()
test_synapse_backward()
