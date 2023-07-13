# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

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
import bittensor.utils.weight_utils as weight_utils
import pytest


def test_convert_weight_and_uids():
    uids = torch.tensor(list(range(10)))
    weights = torch.rand(10)
    weight_utils.convert_weights_and_uids_for_emit(uids, weights)

    # min weight < 0
    weights[5] = -1
    with pytest.raises(ValueError) as pytest_wrapped_e:
        weight_utils.convert_weights_and_uids_for_emit(uids, weights)

    # min uid < 0
    weights[5] = 0
    uids[3] = -1
    with pytest.raises(ValueError) as pytest_wrapped_e:
        weight_utils.convert_weights_and_uids_for_emit(uids, weights)

    # len(uids) != len(weights)
    uids[3] = 3
    with pytest.raises(ValueError) as pytest_wrapped_e:
        weight_utils.convert_weights_and_uids_for_emit(uids, weights[1:])

    # sum(weights) == 0
    weights = torch.zeros(10)
    weight_utils.convert_weights_and_uids_for_emit(uids, weights)

    # test for overflow and underflow
    for _ in range(5):
        uids = torch.tensor(list(range(10)))
        weights = torch.rand(10)
        weight_utils.convert_weights_and_uids_for_emit(uids, weights)


def test_normalize_with_max_weight():
    weights = torch.rand(1000)
    wn = weight_utils.normalize_max_weight(weights, limit=0.01)
    assert wn.max() <= 0.01

    weights = torch.zeros(1000)
    wn = weight_utils.normalize_max_weight(weights, limit=0.01)
    assert wn.max() <= 0.01

    weights = torch.rand(1000)
    wn = weight_utils.normalize_max_weight(weights, limit=0.02)
    assert wn.max() <= 0.02

    weights = torch.zeros(1000)
    wn = weight_utils.normalize_max_weight(weights, limit=0.02)
    assert wn.max() <= 0.02

    weights = torch.rand(1000)
    wn = weight_utils.normalize_max_weight(weights, limit=0.03)
    assert wn.max() <= 0.03

    weights = torch.zeros(1000)
    wn = weight_utils.normalize_max_weight(weights, limit=0.03)
    assert wn.max() <= 0.03

    # Check for Limit
    limit = 0.001
    weights = torch.rand(2000)
    w = weights / weights.sum()
    wn = weight_utils.normalize_max_weight(weights, limit=limit)
    assert (w.max() >= limit and (limit - wn.max()).abs() < 0.001) or (
        w.max() < limit and wn.max() < limit
    )

    # Check for Zeros
    limit = 0.01
    weights = torch.zeros(2000)
    wn = weight_utils.normalize_max_weight(weights, limit=limit)
    assert wn.max() == 1 / 2000

    # Check for Ordering after normalization
    weights = torch.rand(100)
    wn = weight_utils.normalize_max_weight(weights, limit=1)
    assert torch.equal(wn, weights / weights.sum())

    # Check for eplison changes
    eplison = 0.01
    weights, _ = torch.sort(torch.rand(100))
    x = weights / weights.sum()
    limit = x[-10]
    change = eplison * limit
    y = weight_utils.normalize_max_weight(x, limit=limit - change)
    z = weight_utils.normalize_max_weight(x, limit=limit + change)
    assert (y - z).abs().sum() < eplison
