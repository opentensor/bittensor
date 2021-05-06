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

# def test_ranks():
#     assert torch.all(metagraph.ranks.eq(torch.tensor(list(range(100)))))
#     assert torch.all(metagraph.R.eq(torch.tensor(list(range(100)))))

# def test_I():
#     Ipr = torch.tensor(list(range(100))) * torch.tensor(0.5) / torch.sum(torch.tensor(list(range(100))))
#     assert torch.all(metagraph.I.eq(Ipr))

# def test_W():
#     for i in range(100):
#         row = torch.zeros(100)
#         row[i] = 1
#         assert torch.all(metagraph.W[i, :].eq(row))

#     for i in range(100):
#         col = torch.zeros(100)
#         col[i] = 1
#         assert torch.all(metagraph.W[:, i].eq(col))











   
