import bittensor
import torch
import unittest

metagraph = None
def test_create():
    global metagraph
    metagraph = bittensor.metagraph(network = 'nobunaga')

def test_print_empty():
    print (metagraph)

def test_sync():
    metagraph.sync()
    metagraph.sync(600000)

def test_load_sync_save():
    metagraph.sync()
    metagraph.save()
    metagraph.load()
    metagraph.save()

def test_factory():
    metagraph.load().sync().save()

def test_forward():
    row = torch.ones( (metagraph.n), dtype = torch.float32 )
    for i in range( metagraph.n ):
        metagraph(i, row)
    metagraph.sync()
    row = torch.ones( (metagraph.n), dtype = torch.float32 )
    for i in range( metagraph.n ):
        metagraph(i, row)

def test_state_dict():
    metagraph.load()
    state = metagraph.state_dict()
    assert 'uids' in state
    assert 'stake' in state
    assert 'last_update' in state
    assert 'block' in state
    assert 'tau' in state
    assert 'weights' in state
    assert 'endpoints' in state

def test_properties():
    metagraph.hotkeys
    metagraph.coldkeys
    metagraph.endpoints
    metagraph.R
    metagraph.T
    metagraph.S
    metagraph.D
    metagraph.C

def test_retrieve_cached_neurons():
    n = metagraph.retrieve_cached_neurons()
    assert len(n) >= 2000
    
    n = metagraph.retrieve_cached_neurons(600000)
    assert len(n) >= 2000

def test_to_dataframe():
    df = metagraph.to_dataframe()
    assert not df.empty()