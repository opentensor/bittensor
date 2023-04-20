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

# To run these tests.
# python3 -m pytest -s bittensor/_synapse/text_last_hidden_state/tests.py 

import torch
import bittensor

def test_create_create_calls():
    bittensor.TextLastHiddenStateForwardCall(
        text_inputs = torch.LongTensor(1, 1),
    )
    bittensor.TextLastHiddenStateBackwardCall(
        text_inputs = torch.LongTensor(1, 1),
        hidden_states = torch.FloatTensor(1, 1, bittensor.__network_dim__),
        hidden_states_grads = torch.FloatTensor(1, 1, bittensor.__network_dim__),
    )

def test_pre_process_forward_call_to_request_proto_passes():
    wallet = bittensor.wallet.mock()
    endpoint = bittensor.endpoint.dummy()
    module = bittensor.text_last_hidden_state( wallet = wallet, endpoint = endpoint )
    request = module.pre_process_forward_call_to_request_proto(
        forward_call = bittensor.TextLastHiddenStateForwardCall(
            text_inputs = torch.LongTensor(1, 1),
            mask = torch.zeros( 1, 1, dtype=torch.bool )
        )
    )


if __name__ == "__main__":
    test_create_create_calls()
    test_pre_process_forward_call_to_request_proto_passes()
