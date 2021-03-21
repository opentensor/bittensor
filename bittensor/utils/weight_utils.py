import torch

def convert_weights_and_uids_for_emit( uid:torch.LongTensor, weights: torch.FloatTensor ) -> Tuple[List[int], List[int]]:
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
    for i, weight_i, uid_i in enumerate(list(zip(weights, uids))):
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