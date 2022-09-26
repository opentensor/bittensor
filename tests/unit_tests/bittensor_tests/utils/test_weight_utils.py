import torch
import bittensor.utils.weight_utils as weight_utils
import pytest

def test_convert_weight_and_uids():
    uids = torch.tensor(list(range(10))) 
    weights = torch.rand(10)
    weight_utils.convert_weights_and_uids_for_emit( uids, weights )

    # min weight < 0
    weights[5] = -1
    with pytest.raises(ValueError) as pytest_wrapped_e:
        weight_utils.convert_weights_and_uids_for_emit( uids, weights )
    
    # min uid < 0
    weights[5] = 0
    uids[3] = -1
    with pytest.raises(ValueError) as pytest_wrapped_e:
        weight_utils.convert_weights_and_uids_for_emit( uids, weights )
    
    # len(uids) != len(weights)
    uids[3] = 3
    with pytest.raises(ValueError) as pytest_wrapped_e:
        weight_utils.convert_weights_and_uids_for_emit( uids, weights[1:] )
    
    # sum(weights) == 0
    weights = torch.zeros(10)
    weight_utils.convert_weights_and_uids_for_emit( uids, weights )

    # test for overflow and underflow
    for _ in range (5):
        uids = torch.tensor(list(range(10))) 
        weights = torch.rand(10)
        weight_utils.convert_weights_and_uids_for_emit( uids, weights )

def test_normalize_with_min_max():
    weights = torch.rand(10)
    wn = weight_utils.normalize_max_multiple( weights, multiple = 10 )
    assert wn.max() / wn.min() <= 11

    weights = torch.rand(2)
    wn = weight_utils.normalize_max_multiple( weights, multiple = 10 )
    assert wn.max() / wn.min() <= 11

    weights = torch.randn(10)
    wn = weight_utils.normalize_max_multiple( weights, multiple = 10 )
    assert wn.max() / wn.min() <= 11

    weights = torch.eye(10)[0]
    wn = weight_utils.normalize_max_multiple( weights, multiple = 10 )
    assert wn.max() / wn.min() <= 11

    weights = torch.zeros(10)
    wn = weight_utils.normalize_max_multiple( weights, multiple = 10 )
    assert wn.max() / wn.min() <= 11

    weights = torch.rand(10)
    wn = weight_utils.normalize_max_multiple( weights, multiple = 2 )
    assert wn.max() / wn.min() <= 3

    weights = torch.rand(2)
    wn = weight_utils.normalize_max_multiple( weights, multiple = 2 )
    assert wn.max() / wn.min() <= 3

    weights = torch.randn(10)
    wn = weight_utils.normalize_max_multiple( weights, multiple = 2 )
    assert wn.max() / wn.min() <= 3

    weights = torch.eye(10)[0]
    wn = weight_utils.normalize_max_multiple( weights, multiple = 2 )
    assert wn.max() / wn.min() <= 3

    weights = torch.zeros(10)
    wn = weight_utils.normalize_max_multiple( weights, multiple = 2 )
    assert wn.max() / wn.min() <= 3
