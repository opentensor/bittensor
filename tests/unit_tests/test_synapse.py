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

import typing
import bittensor as bt

def test_synapse_create():
    synapse = bt.Synapse()
    assert isinstance( synapse, bt.Synapse )
    assert synapse.name == 'Synapse'
    assert synapse.timeout == 12.0
    assert synapse.header_size == 0
    assert synapse.total_size == 0
    headers = synapse.to_headers()
    assert isinstance( headers, dict )
    assert 'timeout' in headers
    assert 'name' in headers
    assert 'header_size' in headers
    assert 'total_size' in headers
    assert headers['name'] == 'Synapse'
    assert headers['timeout'] == '12.0'
    next_synapse = synapse.from_headers( synapse.to_headers() )
    assert next_synapse.timeout == 12.0

def test_custom_synapse():        
    class Test( bt.Synapse ):
        a: int # Carried through because required.
        b: int = None # Not carried through headers
        c: typing.Optional[int]  # Not carried through headers
        d: typing.Optional[typing.List[int]]  # Not carried through headers
        e: typing.List[int]  # Not carried through headers

    synapse = Test( 
        a = 1, 
        c = 3, 
        d = [1,2,3,4], 
        e = [1,2,3,4],
    )
    assert isinstance( synapse, Test )
    assert synapse.name == 'Test'
    assert synapse.a == 1
    assert synapse.b == None
    assert synapse.c == 3
    assert synapse.d == [1,2,3,4]
    assert synapse.e == [1,2,3,4]

    headers = synapse.to_headers()
    assert 'bt_header_input_obj_a' in headers
    assert 'bt_header_input_obj_b' not in headers
    next_synapse = synapse.from_headers( synapse.to_headers() )
    assert next_synapse.a == 1
    assert next_synapse.b == None
    assert next_synapse.c == None
    assert next_synapse.d == None
    assert next_synapse.e == [1,2,3,4]


if __name__  == "__main__":
    test_synapse_create()
    test_custom_synapse()