import torch
import unittest
import pytest
from munch import Munch
import bittensor
from bittensor.substrate import Keypair
import multiprocessing
import time

def test_create( ):
    metagraph = bittensor.metagraph.Metagraph()
    assert True

def test_print_empty():
    metagraph = bittensor.metagraph.Metagraph()
    print (metagraph)

def test_metagraph_create_null():
    metagraph = bittensor.metagraph.Metagraph()
    metagraph._update(
        n = 0,
        tau = 0.5,
        block = 0,
        uids = [0],
        stake = [0],
        lastemit = [0],
        weight_vals = [[0]],
        weight_uids = [[0]],
        neurons = []
    )
    assert metagraph.n == 0
    assert metagraph.block == 0
    assert metagraph.tau == 0.5
    assert torch.all(metagraph.uids.eq(torch.tensor([0])))
    assert torch.all(metagraph.stake.eq(torch.tensor([0])))
    assert torch.all(metagraph.lastemit.eq(torch.tensor([0])))
    assert torch.all(metagraph.S.eq(torch.tensor([0])))
    assert torch.all(metagraph.I.eq(torch.tensor([0])))
    assert torch.all(metagraph.R.eq(torch.tensor([0])))
    assert torch.all(metagraph.W.eq(torch.tensor([[0]])))

def test_metagraph_create_100():
    metagraph = bittensor.metagraph.Metagraph()
    metagraph._update(
        n = 100,
        tau = 0.5,
        block = 0,
        uids = range(100),
        stake = [0] * 100,
        lastemit = [0] * 100,
        weight_vals = [ [0 for _ in range(100)] for _ in range(100) ],
        weight_uids = [ [i for i in range(100)] for _ in range(100) ],
        neurons = [
            bittensor.proto.Neuron(
                version = bittensor.__version__,
                public_key = str(i),
                address = str(i),
                port = i,
            ) for i in range(100) 
        ]
    )
    assert metagraph.n == 100
    assert metagraph.block == 0
    assert metagraph.tau == 0.5
    assert torch.all(metagraph.uids.eq(torch.tensor([i for i in range(100) ])))
    assert torch.all(metagraph.stake.eq(torch.tensor([0 for i in range(100) ])))
    assert torch.all(metagraph.lastemit.eq(torch.tensor([0 for i in range(100) ])))
    assert torch.all(metagraph.S.eq(torch.tensor([0 for i in range(100) ])))
    assert torch.all(metagraph.R.eq(torch.tensor([0 for i in range(100) ])))
    assert torch.all(metagraph.I.eq(torch.tensor([0 for i in range(100) ])))
    assert torch.all(metagraph.W.eq(torch.tensor([[0 for _ in range(100)] for _ in range(100) ])))

    assert metagraph.neurons[0].public_key == '0'
    assert metagraph.neurons[1].public_key == '1'
    assert metagraph.neurons[99].public_key == '99'
    assert metagraph.public_keys[0] == '0'
    assert metagraph.public_keys[1] == '1'
    assert metagraph.public_keys[99] == '99'
    assert metagraph.uids_to_neurons(torch.tensor(range(100)))[0].public_key == '0'
    assert metagraph.uids_to_neurons(torch.tensor(range(100)))[1].public_key == '1'
    assert metagraph.uids_to_neurons(torch.tensor(range(100)))[99].public_key == '99'
    assert torch.all(metagraph.neurons_to_uids( metagraph.neurons ).eq( torch.tensor(range(100))))
    for i in range(100):
        assert metagraph.uid_for_pubkey(str(i)) == i
        assert metagraph.neuron_for_uid(i).public_key == str(i)


def test_sync():
    metagraph = bittensor.metagraph.Metagraph(
        subtensor = bittensor.subtensor.Subtensor(
            network = 'kusanagi'
        )
    )
    print ('sync')
    metagraph.sync()
    print ('update')
    metagraph.sync()
    print (metagraph.n)
    print (metagraph.block)
    print (metagraph.tau)
    print (metagraph.uids)
    print (metagraph.stake)
    print (metagraph.lastemit)
    print (metagraph.S)
    print (metagraph.R)
    print (metagraph.I)
    print (metagraph.W)
    print (metagraph.neurons)
    print (metagraph.public_keys)
    print (metagraph.uids_to_neurons(metagraph.uids))
    print (metagraph.neurons_to_uids(metagraph.neurons))

def test_multprocessing_access():
    metagraph = bittensor.metagraph.Metagraph(
        subtensor = bittensor.subtensor.Subtensor(
            network = 'kusanagi'
        )
    )
    metagraph.sync()
    def call_methods( object ):
        for _ in range(5):
            metagraph.n
            metagraph.block
            metagraph.tau
            metagraph.uids
            metagraph.stake
            metagraph.lastemit
            metagraph.S
            metagraph.R
            metagraph.I
            metagraph.W
            metagraph.neurons
            metagraph.public_keys
            metagraph.uids_to_neurons(metagraph.uids)
            metagraph.neurons_to_uids(metagraph.neurons)
            time.sleep(0.5)

    p1 = multiprocessing.Process(target=call_methods, args=(metagraph,))
    p2 = multiprocessing.Process(target=call_methods, args=(metagraph,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

if __name__ == "__main__":
    test_multprocessing_access()
    #test_print_empty()
    #test_chain_state()