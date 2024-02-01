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


@pytest.mark.parametrize(
    "test_id, n, uids, weights, expected",
    [
        ("happy-path-1", 3, [0, 1, 2], [15, 5, 80], torch.tensor([0.15, 0.05, 0.8])),
        ("happy-path-2", 4, [1, 3], [50, 50], torch.tensor([0.0, 0.5, 0.0, 0.5])),
    ],
)
def test_convert_weight_uids_and_vals_to_tensor_happy_path(
    test_id, n, uids, weights, expected
):
    # Act
    result = weight_utils.convert_weight_uids_and_vals_to_tensor(n, uids, weights)

    # Assert
    assert torch.allclose(result, expected), f"Failed {test_id}"


@pytest.mark.parametrize(
    "test_id, n, uids, weights, expected",
    [
        ("edge_case_empty", 5, [], [], torch.zeros(5)),
        ("edge_case_single", 1, [0], [100], torch.tensor([1.0])),
        ("edge_case_all_zeros", 4, [0, 1, 2, 3], [0, 0, 0, 0], torch.zeros(4)),
    ],
)
def test_convert_weight_uids_and_vals_to_tensor_edge_cases(
    test_id, n, uids, weights, expected
):
    # Act
    result = weight_utils.convert_weight_uids_and_vals_to_tensor(n, uids, weights)

    # Assert
    assert torch.allclose(result, expected), f"Failed {test_id}"


@pytest.mark.parametrize(
    "test_id, n, uids, weights, exception",
    [
        ("error-case-mismatched-lengths", 3, [0, 1, 3, 4, 5], [10, 20, 30], IndexError),
        ("error-case-negative-n", -1, [0, 1], [10, 20], RuntimeError),
        ("error-case-invalid-uids", 3, [0, 3], [10, 20], IndexError),
    ],
)
def test_convert_weight_uids_and_vals_to_tensor_error_cases(
    test_id, n, uids, weights, exception
):
    # Act / Assert
    with pytest.raises(exception):
        weight_utils.convert_weight_uids_and_vals_to_tensor(n, uids, weights)


@pytest.mark.parametrize(
    "test_id, n, uids, weights, subnets, expected",
    [
        (
            "happy-path-1",
            3,
            [0, 1, 2],
            [15, 5, 80],
            [0, 1, 2],
            torch.tensor([0.15, 0.05, 0.8]),
        ),
        (
            "happy-path-2",
            3,
            [0, 2],
            [300, 300],
            [0, 1, 2],
            torch.tensor([0.5, 0.0, 0.5]),
        ),
    ],
)
def test_convert_root_weight_uids_and_vals_to_tensor_happy_paths(
    test_id, n, uids, weights, subnets, expected
):
    # Act
    result = weight_utils.convert_root_weight_uids_and_vals_to_tensor(
        n, uids, weights, subnets
    )

    # Assert
    assert torch.allclose(result, expected, atol=1e-4), f"Failed {test_id}"


@pytest.mark.parametrize(
    "test_id, n, uids, weights, subnets, expected",
    [
        (
            "edge-1",
            1,
            [0],
            [0],
            [0],
            torch.tensor([0.0]),
        ),  # Single neuron with zero weight
        (
            "edge-2",
            2,
            [0, 1],
            [0, 0],
            [0, 1],
            torch.tensor([0.0, 0.0]),
        ),  # All zero weights
    ],
)
def test_convert_root_weight_uids_and_vals_to_tensor_edge_cases(
    test_id, n, uids, weights, subnets, expected
):
    # Act
    result = weight_utils.convert_root_weight_uids_and_vals_to_tensor(
        n, uids, weights, subnets
    )

    # Assert
    assert torch.allclose(result, expected, atol=1e-4), f"Failed {test_id}"


@pytest.mark.parametrize(
    "test_id, n, uids, weights, subnets, exception",
    [
        ("error-1", 3, [1, 3], [100, 200], [1, 2], Exception),  # uid not in subnets
        ("error-2", 3, [1, 2, 3], [100, 200], [1], Exception),  # More uids than subnets
    ],
)
def test_convert_root_weight_uids_and_vals_to_tensor_error_cases(
    test_id, n, uids, weights, subnets, exception
):
    # Act and Assert
    with pytest.raises(exception):
        weight_utils.convert_root_weight_uids_and_vals_to_tensor(
            n, uids, weights, subnets
        )
        print(f"Failed {test_id}")


@pytest.mark.parametrize(
    "test_id, n, uids, bonds, expected_output",
    [
        (
            "happy-path-1",
            5,
            [1, 3, 4],
            [10, 20, 30],
            torch.tensor([0, 10, 0, 20, 30], dtype=torch.int64),
        ),
        (
            "happy-path-2",
            3,
            [0, 1, 2],
            [7, 8, 9],
            torch.tensor([7, 8, 9], dtype=torch.int64),
        ),
        ("happy-path-3", 4, [2], [15], torch.tensor([0, 0, 15, 0], dtype=torch.int64)),
    ],
)
def test_happy_path(test_id, n, uids, bonds, expected_output):
    # Act
    result = weight_utils.convert_bond_uids_and_vals_to_tensor(n, uids, bonds)

    # Assert
    assert torch.equal(result, expected_output), f"Failed {test_id}"


@pytest.mark.parametrize(
    "test_id, n, uids, bonds, expected_output",
    [
        ("edge-1", 1, [0], [0], torch.tensor([0], dtype=torch.int64)),  # Single element
        (
            "edge-2",
            10,
            [],
            [],
            torch.zeros(10, dtype=torch.int64),
        ),  # Empty uids and bonds
    ],
)
def test_edge_cases(test_id, n, uids, bonds, expected_output):
    # Act
    result = weight_utils.convert_bond_uids_and_vals_to_tensor(n, uids, bonds)

    # Assert
    assert torch.equal(result, expected_output), f"Failed {test_id}"


@pytest.mark.parametrize(
    "test_id, n, uids, bonds, exception",
    [
        ("error-1", 5, [1, 3, 6], [10, 20, 30], IndexError),  # uid out of bounds
        ("error-2", -1, [0], [10], RuntimeError),  # Negative number of neurons
    ],
)
def test_error_cases(test_id, n, uids, bonds, exception):
    # Act / Assert
    with pytest.raises(exception):
        weight_utils.convert_bond_uids_and_vals_to_tensor(n, uids, bonds)
