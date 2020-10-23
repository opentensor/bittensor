import unittest
import torch

class TestCUDA(unittest.TestCase):
    device = None
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_cuda(self):
        assert self.device == torch.device("cuda")