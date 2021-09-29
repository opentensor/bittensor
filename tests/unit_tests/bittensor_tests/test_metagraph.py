import bittensor
import torch

def test_create( ):
    global metagraph
    metagraph = bittensor.metagraph()
    assert True

def test_print_empty():
    metagraph = bittensor.metagraph()
    print (metagraph)

def test_forward():
    meta = bittensor.metagraph()
    row = torch.ones( (meta.n), dtype = torch.float32 )
    for i in range( meta.n ):
        meta(i, row)
    meta.sync()
    row = torch.ones( (meta.n), dtype = torch.float32 )
    for i in range( meta.n ):
        meta(i, row)

def test_load_sync_save():
    metagraph = bittensor.metagraph()
    metagraph.sync()
    metagraph.save()
    metagraph.load()
    metagraph.save()

def test_factory():
    graph = bittensor.metagraph().load().sync().save()

def test_state_dict():
    metagraph = bittensor.metagraph()
    metagraph.load()
    state = metagraph.state_dict()
    assert 'uids' in state
    assert 'stake' in state
    assert 'last_update' in state
    assert 'block' in state
    assert 'tau' in state
    assert 'weights' in state
    assert 'endpoints' in state

test_load_sync_save()








   
