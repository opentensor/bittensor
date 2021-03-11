import torch
import bittensor

def test_nucleus():
    nucleus = bittensor.Nucleus()
    del nucleus

def test_nucleus_deepcopy():
    nucleus = bittensor.Nucleus()
    nucleus2 = nucleus.deepcopy()

class MockNucleus(bittensor.Nucleus):
    def forward_text(self, text: torch.LongTensor) -> torch.FloatTensor:
        return torch.tensor([1])

    def forward_image(self, images: torch.FloatTensor) -> torch.FloatTensor:
        return torch.tensor([2])

    def forward_tensor(self, tensors: torch.FloatTensor) -> torch.FloatTensor:
        return torch.tensor([3])

def test_nucleus_text():
    nucleus = MockNucleus()
    response = nucleus.call_forward(torch.tensor([1]), bittensor.proto.Modality.TEXT)
    assert response == torch.tensor([1])

def test_nucleus_image():
    nucleus = MockNucleus()
    response = nucleus.call_forward(torch.tensor([1]), bittensor.proto.Modality.IMAGE)
    assert response == torch.tensor([2])

def test_nucleus_tensor():
    nucleus = MockNucleus()
    response = nucleus.call_forward(torch.tensor([1]), bittensor.proto.Modality.TENSOR)
    assert response == torch.tensor([3])

def test_nucleus_text_no_grad():
    nucleus = MockNucleus()
    response = nucleus.call_forward(torch.tensor([1]), bittensor.proto.Modality.TEXT, no_grad=False)
    assert response == torch.tensor([1])

def test_nucleus_image_no_grad():
    nucleus = MockNucleus()
    response = nucleus.call_forward(torch.tensor([1]), bittensor.proto.Modality.IMAGE, no_grad=False)
    assert response == torch.tensor([2])

def test_nucleus_tensor_no_grad():
    nucleus = MockNucleus()
    response = nucleus.call_forward(torch.tensor([1]), bittensor.proto.Modality.TENSOR, no_grad=False)
    assert response == torch.tensor([3])

class MultiplicationNucleus(bittensor.Nucleus):
    def __init__(self):
        super().__init__()
        self.weight = torch.autograd.Variable(torch.tensor([2.0]))
    def forward_tensor(self, tensors: torch.FloatTensor) -> torch.FloatTensor:
        return tensors * self.weight

def test_nucleus_multiplication():
    nucleus = MultiplicationNucleus()
    response = nucleus.call_forward(torch.tensor([1]), bittensor.proto.Modality.TENSOR)
    assert response == torch.tensor([2])

def test_nucleus_grad():
    nucleus = MultiplicationNucleus()
    grad_dy = torch.autograd.Variable(torch.tensor([1.0]))
    inputs = torch.autograd.Variable(torch.tensor([1.0]), requires_grad=True)
    response = nucleus.grad(inputs, grad_dy, bittensor.proto.Modality.TENSOR)
    assert torch.isclose(response, torch.tensor([2.0]))

def test_nucleus_backward():
    nucleus = MultiplicationNucleus()
    grad_dy = torch.autograd.Variable(torch.tensor([1.0]))
    inputs = torch.autograd.Variable(torch.tensor([1.0]), requires_grad=True)
    nucleus.backward(inputs, grad_dy, bittensor.proto.Modality.TENSOR)

test_nucleus_grad()
test_nucleus_backward()
