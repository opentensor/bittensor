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

import torch
from typing import Tuple, List

def convert_weight_uids_and_vals_to_tensor( n: int, uids: List[int], weights: List[int] ):
    r""" Converts weights and uids from chain representation into a torch tensor (inverse operation from convert_weights_and_uids_for_emit)
        Returns:
            n: int:
                number of neurons on network.
            uids (:obj:`List[int],`):
                Tensor of uids as destinations for passed weights.
            weights (:obj:`List[int],`):
                Tensor of weights.
    """
    row_weights = torch.zeros( [ n ], dtype=torch.float32 )
    for uid_j, wij in list(zip( uids, weights )):
        row_weights[ uid_j ] = float( wij ) / float(4294967295)
    return row_weights

def convert_weights_and_uids_for_emit( uids: torch.LongTensor, weights: torch.FloatTensor ) -> Tuple[List[int], List[int]]:
    r""" Converts weights into integer u32 representation that sum to MAX_INT_WEIGHT.
        Returns:
            uids (:obj:`torch.LongTensor,`):
                Tensor of uids as destinations for passed weights.
            weights (:obj:`torch.LongTensor,`):
                Tensor of weights.
    """
    # Checks.
    weights = weights.tolist()
    uids = uids.tolist()
    if min(weights) < 0:
        raise ValueError('Passed weight is negative cannot exist on chain {}'.format(weights))
    if min(uids) < 0:
        raise ValueError('Passed uid is negative cannot exist on chain {}'.format(uids))
    if len(uids) != len(weights):
        raise ValueError('Passed weights and uids must have the same length, got {} and {}'.format(len(uids), len(weights)))
    if sum(weights) == 0:
        return [],[] # Nothing to set on chain.
    else:
        weights = [ float(value) / sum(weights) for value in weights] # Initial normalization.

    remainder = 4294967295 
    weight_vals = []
    weight_uids = []
    for i, (weight_i, uid_i) in enumerate(list(zip(weights, uids))):
        uint32_val = int(float(weight_i) * int(4294967295)) # convert to int representation.
        remainder -= uint32_val
        
        # Fix overflow
        if remainder < 0:
            uint32_val += remainder
            remainder = 0
        
        # Fix underflow
        if i == (len(weights) -1) and remainder > 0:
            uint32_val += remainder 
            remainder = 0

        # Filter zeros
        if uint32_val != 0: # Filter zeros
            weight_vals.append( uint32_val )
            weight_uids.append( uid_i ) 

    return weight_uids, weight_vals 