import torch
import unittest
import pytest
from munch import Munch
import bittensor
from bittensor.substrate import Keypair

metagraph = None

def test_create( ):
    global metagraph
    metagraph = bittensor.metagraph.Metagraph()
    assert True

def test_print_empty():
    metagraph = bittensor.metagraph.Metagraph()
    print (metagraph)

def test_sync( ):
    metagraph = bittensor.metagraph.Metagraph()
    metagraph.sync()
    assert type(metagraph.n) == type(0)
    assert type(metagraph.W) == type(torch.tensor([]))
    assert type(metagraph.S) == type(torch.tensor([]))
    assert type(metagraph.R) == type(torch.tensor([]))
    assert type(metagraph.I) == type(torch.tensor([]))
    assert type(metagraph.weights) == type(torch.tensor([]))
    assert type(metagraph.uids) == type(torch.tensor([]))
    assert type(metagraph.public_keys) == type([])
    assert type(metagraph.neurons) == type([])










   
