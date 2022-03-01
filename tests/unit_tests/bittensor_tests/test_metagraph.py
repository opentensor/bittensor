# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import bittensor
import torch
import unittest

sub = bittensor.subtensor.mock()

metagraph = None
def test_create():
    global metagraph
    metagraph = bittensor.metagraph(subtensor=sub)

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
    
def test_to_dataframe():
    df = metagraph.to_dataframe()

def test_sync_from_mock():
    g = bittensor.metagraph( subtensor = bittensor.subtensor.mock() )
    g.sync()
