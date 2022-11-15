# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

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

def test_create_last_hidden_state():
    bittensor.synapse.TextLastHiddenState()

def test_create_casuallm():
    bittensor.synapse.TextCausalLM()

def test_create_casuallm_next():
    bittensor.synapse.TextCausalLMNext()

def test_create_seq2seq():
    bittensor.synapse.TextSeq2Seq()

def test_last_hidden_state_encode_forward_response_tensor_no_mask():
    synapse = bittensor.synapse.TextLastHiddenState()
    forward_response_tensor = torch.randn( 30, 256, 1024)
    encoded_forward_response_tensor = synapse.encode_forward_response_tensor( forward_response_tensor )
    assert len(encoded_forward_response_tensor.shape) == 3
    assert encoded_forward_response_tensor.shape[0] == 30
    assert encoded_forward_response_tensor.shape[1] == 256
    assert encoded_forward_response_tensor.shape[2] == 1024
    assert torch.all(torch.eq(encoded_forward_response_tensor[0, 0, :], forward_response_tensor[0, 0, :]))

def test_last_hidden_state_encode_forward_response_tensor_mask_first():
    # Test mask all but first.
    synapse = bittensor.synapse.TextLastHiddenState(
        mask = [0], # Only return the first representation from each batch.
    )
    forward_response_tensor = torch.randn( 30, 256, 1024)
    encoded_forward_response_tensor = synapse.encode_forward_response_tensor( forward_response_tensor )
    assert len(encoded_forward_response_tensor.shape) == 2
    assert encoded_forward_response_tensor.shape[0] == 30
    assert encoded_forward_response_tensor.shape[1] == 1024
    assert torch.all(torch.eq(encoded_forward_response_tensor[0, :], forward_response_tensor[0, 0, :]))


def test_last_hidden_state_encode_forward_response_tensor_mask_last():
    # Test mask last
    synapse = bittensor.synapse.TextLastHiddenState(
        mask = [-1], # Only return the first representation from each batch.
    )
    forward_response_tensor = torch.randn( 30, 256, 1024)
    encoded_forward_response_tensor = synapse.encode_forward_response_tensor( forward_response_tensor )
    assert len(encoded_forward_response_tensor.shape) == 2
    assert encoded_forward_response_tensor.shape[0] == 30
    assert encoded_forward_response_tensor.shape[1] == 1024
    assert torch.all( torch.eq(encoded_forward_response_tensor[0, :], forward_response_tensor[0, 255, :]) ) 

def test_last_hidden_state_encode_forward_response_tensor_mask_multiple():
    # Test mask last
    n = 5
    synapse = bittensor.synapse.TextLastHiddenState(
        mask = list(range(n)), # Only return the first representation from each batch.
    )
    forward_response_tensor = torch.randn( 30, 256, 1024)
    encoded_forward_response_tensor = synapse.encode_forward_response_tensor( forward_response_tensor )
    assert len(encoded_forward_response_tensor.shape) == 2
    assert encoded_forward_response_tensor.shape[0] == 30 * n
    assert encoded_forward_response_tensor.shape[1] == 1024
    
    # Iterate through all and check equality.
    idx = 0
    for j in range(30):
        for i in range( n ):
            assert torch.all( torch.eq(encoded_forward_response_tensor[ idx, :], forward_response_tensor[ j, i, :]) ) 
            idx += 1




if __name__ == "__main__":
    test_create_last_hidden_state()