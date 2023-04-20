""" Conversion for weight between chain representation and torch tensor
"""
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

from typing import Tuple, List
import torch

U32_MAX = 4294967295
U16_MAX = 65535

def normalize_max_weight(  x: torch.FloatTensor, limit:float = 0.1 ) -> 'torch.FloatTensor':
    r""" Normalizes the tensor x so that sum(x) = 1 and the max value is not greater than the limit.
        Args:
            x (:obj:`torch.FloatTensor`):
                Tensor to be max_value normalized.
            limit: float:
                Max value after normalization.     
        Returns:
            y (:obj:`torch.FloatTensor`):
                Normalized x tensor.
    """
    epsilon = 1e-7 #For numerical stability after normalization
    
    weights =  x.clone()
    values, _ = torch.sort(weights)

    if x.sum() == 0 or len(x)*limit <= 1:
        return torch.ones_like(x)/x.size(0)
    else:
        estimation = values/values.sum()
        
        if estimation.max() <= limit:
            return weights/weights.sum()

        # Find the cumlative sum and sorted tensor
        cumsum = torch.cumsum(estimation,0)

        # Determine the index of cutoff
        estimation_sum = torch.tensor([(len(values)-i-1)*estimation[i] for i in range(len(values))])
        n_values = (estimation/(estimation_sum+cumsum+epsilon)<limit).sum()

        # Determine the cutoff based on the index        
        cutoff_scale = (limit*cumsum[n_values-1]-epsilon)/(1-(limit*(len(estimation)-n_values)))
        cutoff= cutoff_scale*values.sum()

        # Applying the cutoff
        weights[weights > cutoff] = cutoff

        y = weights/weights.sum()

        return y

def convert_weight_uids_and_vals_to_tensor( n: int, uids: List[int], weights: List[int] ) -> 'torch.FloatTensor':
    r""" Converts weights and uids from chain representation into a torch tensor (inverse operation from convert_weights_and_uids_for_emit)
        Args:
            n: int:
                number of neurons on network.
            uids (:obj:`List[int],`):
                Tensor of uids as destinations for passed weights.
            weights (:obj:`List[int],`):
                Tensor of weights.
        Returns:
            row_weights ( torch.FloatTensor ):
                Converted row weights.
    """
    row_weights = torch.zeros( [ n ], dtype=torch.float32 )
    for uid_j, wij in list(zip( uids, weights )):
        row_weights[ uid_j ] = float( wij )  # assumes max-upscaled values (w_max = U16_MAX).
    row_sum = row_weights.sum()
    if row_sum > 0:
        row_weights /= row_sum  # normalize
    return row_weights

def convert_bond_uids_and_vals_to_tensor( n: int, uids: List[int], bonds: List[int] ) -> 'torch.LongTensor':
    r""" Converts bond and uids from chain representation into a torch tensor.
        Args:
            n: int:
                number of neurons on network.
            uids (:obj:`List[int],`):
                Tensor of uids as destinations for passed bonds.
            bonds (:obj:`List[int],`):
                Tensor of bonds.
        Returns:
            row_bonds ( torch.FloatTensor ):
                Converted row bonds.
    """
    row_bonds = torch.zeros( [ n ], dtype=torch.int64 )
    for uid_j, bij in list(zip( uids, bonds )):
        row_bonds[ uid_j ] = int( bij ) 
    return row_bonds

def convert_weights_and_uids_for_emit( uids: torch.LongTensor, weights: torch.FloatTensor ) -> Tuple[List[int], List[int]]:
    r""" Converts weights into integer u32 representation that sum to MAX_INT_WEIGHT.
        Args:
            uids (:obj:`torch.LongTensor,`):
                Tensor of uids as destinations for passed weights.
            weights (:obj:`torch.FloatTensor,`):
                Tensor of weights.
        Returns:
            weight_uids (List[int]):
                Uids as a list.
            weight_vals (List[int]):
                Weights as a list.
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
        max_weight = float(max(weights))
        weights = [float(value) / max_weight for value in weights]  # max-upscale values (max_weight = 1).

    weight_vals = []
    weight_uids = []
    for i, (weight_i, uid_i) in enumerate(list(zip(weights, uids))):
        uint16_val = round(float(weight_i) * int(U16_MAX))  # convert to int representation.

        # Filter zeros
        if uint16_val != 0: # Filter zeros
            weight_vals.append( uint16_val )
            weight_uids.append( uid_i ) 

    return weight_uids, weight_vals