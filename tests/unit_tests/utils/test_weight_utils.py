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

import logging
import numpy as np
from hypothesis import settings

import bittensor.utils.weight_utils as weight_utils
import pytest

from bittensor.utils import torch
from bittensor.core.settings import version_as_int


def test_convert_weight_and_uids():
    uids = np.arange(10)
    weights = np.random.rand(10)
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
    weights = np.zeros(10)
    weight_utils.convert_weights_and_uids_for_emit(uids, weights)

    # test for overflow and underflow
    for _ in range(5):
        uids = np.arange(10)
        weights = np.random.rand(10)
        weight_utils.convert_weights_and_uids_for_emit(uids, weights)


def test_convert_weight_and_uids_torch(force_legacy_torch_compatible_api):
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
    weights = np.random.rand(1000)
    wn = weight_utils.normalize_max_weight(weights, limit=0.01)
    assert wn.max() <= 0.01

    weights = np.zeros(1000)
    wn = weight_utils.normalize_max_weight(weights, limit=0.01)
    assert wn.max() <= 0.01

    weights = np.random.rand(1000)
    wn = weight_utils.normalize_max_weight(weights, limit=0.02)
    assert wn.max() <= 0.02

    weights = np.zeros(1000)
    wn = weight_utils.normalize_max_weight(weights, limit=0.02)
    assert wn.max() <= 0.02

    weights = np.random.rand(1000)
    wn = weight_utils.normalize_max_weight(weights, limit=0.03)
    assert wn.max() <= 0.03

    weights = np.zeros(1000)
    wn = weight_utils.normalize_max_weight(weights, limit=0.03)
    assert wn.max() <= 0.03

    # Check for Limit
    limit = 0.001
    weights = np.random.rand(2000)
    w = weights / weights.sum()
    wn = weight_utils.normalize_max_weight(weights, limit=limit)
    assert abs((w.max() >= limit and (limit - wn.max())) < 0.001) or (
        w.max() < limit and wn.max() < limit
    )

    # Check for Zeros
    limit = 0.01
    weights = np.zeros(2000)
    wn = weight_utils.normalize_max_weight(weights, limit=limit)
    assert wn.max() == 1 / 2000

    # Check for Ordering after normalization
    weights = np.random.rand(100)
    wn = weight_utils.normalize_max_weight(weights, limit=1)
    assert np.array_equal(wn, weights / weights.sum())

    # Check for epsilon changes
    epsilon = 0.01
    weights = np.sort(np.random.rand(100))
    x = weights / weights.sum()
    limit = x[-10]
    change = epsilon * limit
    y = weight_utils.normalize_max_weight(x, limit=limit - change)
    z = weight_utils.normalize_max_weight(x, limit=limit + change)
    assert np.abs(y - z).sum() < epsilon


def test_normalize_with_max_weight__legacy_torch_api_compat(
    force_legacy_torch_compatible_api,
):
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
    assert torch.isclose(wn, weights / weights.sum(), atol=1e-08, rtol=0).all()

    # Check for epsilon changes
    epsilon = 0.01
    weights, _ = torch.sort(torch.rand(100))
    x = weights / weights.sum()
    limit = x[-10]
    change = epsilon * limit
    y = weight_utils.normalize_max_weight(x, limit=limit - change)
    z = weight_utils.normalize_max_weight(x, limit=limit + change)
    assert (y - z).abs().sum() < epsilon


@pytest.mark.parametrize(
    "test_id, n, uids, weights, expected",
    [
        ("happy-path-1", 3, [0, 1, 2], [15, 5, 80], np.array([0.15, 0.05, 0.8])),
        ("happy-path-2", 4, [1, 3], [50, 50], np.array([0.0, 0.5, 0.0, 0.5])),
    ],
)
def test_convert_weight_uids_and_vals_to_tensor_happy_path(
    test_id, n, uids, weights, expected
):
    # Act
    result = weight_utils.convert_weight_uids_and_vals_to_tensor(n, uids, weights)

    # Assert
    assert np.allclose(result, expected), f"Failed {test_id}"


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
def test_convert_weight_uids_and_vals_to_tensor_happy_path_torch(
    test_id, n, uids, weights, subnets, expected, force_legacy_torch_compatible_api
):
    # Act
    result = weight_utils.convert_weight_uids_and_vals_to_tensor(n, uids, weights)

    # Assert
    assert torch.allclose(result, expected), f"Failed {test_id}"


@pytest.mark.parametrize(
    "test_id, n, uids, weights, expected",
    [
        ("edge_case_empty", 5, [], [], np.zeros(5)),
        ("edge_case_single", 1, [0], [100], np.array([1.0])),
        ("edge_case_all_zeros", 4, [0, 1, 2, 3], [0, 0, 0, 0], np.zeros(4)),
    ],
)
def test_convert_weight_uids_and_vals_to_tensor_edge_cases(
    test_id, n, uids, weights, expected
):
    # Act
    result = weight_utils.convert_weight_uids_and_vals_to_tensor(n, uids, weights)

    # Assert
    assert np.allclose(result, expected), f"Failed {test_id}"


@pytest.mark.parametrize(
    "test_id, n, uids, weights, exception",
    [
        ("error-case-mismatched-lengths", 3, [0, 1, 3, 4, 5], [10, 20, 30], IndexError),
        ("error-case-negative-n", -1, [0, 1], [10, 20], ValueError),
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
            np.array([0.15, 0.05, 0.8]),
        ),
        (
            "happy-path-2",
            3,
            [0, 2],
            [300, 300],
            [0, 1, 2],
            np.array([0.5, 0.0, 0.5]),
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
    assert np.allclose(result, expected, atol=1e-4), f"Failed {test_id}"


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
    test_id, n, uids, weights, subnets, expected, force_legacy_torch_compatible_api
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
            np.array([0.0]),
        ),  # Single neuron with zero weight
        (
            "edge-2",
            2,
            [0, 1],
            [0, 0],
            [0, 1],
            np.array([0.0, 0.0]),
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
    assert np.allclose(result, expected, atol=1e-4), f"Failed {test_id}"


@pytest.mark.parametrize(
    "test_id, n, uids, weights, subnets, exception",
    [
        # uid not in subnets
        (
            "error-1",
            3,
            [1, 3],
            [100, 200],
            [1, 2],
            "The subnet is unavailable at the moment.",
        ),
        # More uids than subnets
        (
            "error-2",
            3,
            [1, 2, 3],
            [100, 200],
            [1],
            "The subnet is unavailable at the moment.",
        ),
    ],
)
def test_convert_root_weight_uids_and_vals_to_tensor_error_cases(
    test_id, n, uids, weights, subnets, exception, caplog
):
    with caplog.at_level(logging.WARNING):
        weight_utils.convert_root_weight_uids_and_vals_to_tensor(
            n, uids, weights, subnets
        )

        assert any(
            exception in record.message and record.levelname == "WARNING"
            for record in caplog.records
        )


@pytest.mark.parametrize(
    "test_id, n, uids, bonds, expected_output",
    [
        (
            "happy-path-1",
            5,
            [1, 3, 4],
            [10, 20, 30],
            np.array([0, 10, 0, 20, 30], dtype=np.int64),
        ),
        (
            "happy-path-2",
            3,
            [0, 1, 2],
            [7, 8, 9],
            np.array([7, 8, 9], dtype=np.int64),
        ),
        ("happy-path-3", 4, [2], [15], np.array([0, 0, 15, 0], dtype=np.int64)),
    ],
)
def test_happy_path(test_id, n, uids, bonds, expected_output):
    # Act
    result = weight_utils.convert_bond_uids_and_vals_to_tensor(n, uids, bonds)

    # Assert
    assert np.array_equal(result, expected_output), f"Failed {test_id}"


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
def test_happy_path_torch(
    test_id, n, uids, bonds, expected_output, force_legacy_torch_compatible_api
):
    # Act
    result = weight_utils.convert_bond_uids_and_vals_to_tensor(n, uids, bonds)

    # Assert
    assert torch.equal(result, expected_output), f"Failed {test_id}"


@pytest.mark.parametrize(
    "test_id, n, uids, bonds, expected_output",
    [
        ("edge-1", 1, [0], [0], np.array([0], dtype=np.int64)),  # Single element
        (
            "edge-2",
            10,
            [],
            [],
            np.zeros(10, dtype=np.int64),
        ),  # Empty uids and bonds
    ],
)
def test_edge_cases(test_id, n, uids, bonds, expected_output):
    # Act
    result = weight_utils.convert_bond_uids_and_vals_to_tensor(n, uids, bonds)

    # Assert
    assert np.array_equal(result, expected_output), f"Failed {test_id}"


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
def test_edge_cases_torch(
    test_id, n, uids, bonds, expected_output, force_legacy_torch_compatible_api
):
    # Act
    result = weight_utils.convert_bond_uids_and_vals_to_tensor(n, uids, bonds)

    # Assert
    assert torch.equal(result, expected_output), f"Failed {test_id}"


@pytest.mark.parametrize(
    "test_id, n, uids, bonds, exception",
    [
        ("error-1", 5, [1, 3, 6], [10, 20, 30], IndexError),  # uid out of bounds
        ("error-2", -1, [0], [10], ValueError),  # Negative number of neurons
    ],
)
def test_error_cases(test_id, n, uids, bonds, exception):
    # Act / Assert
    with pytest.raises(exception):
        weight_utils.convert_bond_uids_and_vals_to_tensor(n, uids, bonds)


def test_process_weights_for_netuid(mocker):
    """Test the process_weights_for_netuid function."""
    # Prep
    fake_uids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    fake_weights = np.array([1.0, 2.5, 3.3, 4.7, 5.9], dtype=np.float32)
    fake_netuid = 1
    fake_subtensor = mocker.MagicMock()
    fake_metagraph = mocker.MagicMock()
    fake_exclude_quantile = 0

    fake_subtensor.min_allowed_weights.return_value = 0.1
    fake_subtensor.max_weight_limit.return_value = 1.0
    fake_metagraph.n = 1
    mocked_normalize_max_weight = mocker.patch.object(
        weight_utils, "normalize_max_weight"
    )

    # Call
    result = weight_utils.process_weights_for_netuid(
        uids=fake_uids,
        weights=fake_weights,
        netuid=fake_netuid,
        subtensor=fake_subtensor,
        metagraph=fake_metagraph,
        exclude_quantile=fake_exclude_quantile,
    )

    # Asserts
    fake_subtensor.min_allowed_weights.assert_called_once_with(netuid=fake_netuid)
    fake_subtensor.max_weight_limit.assert_called_once_with(netuid=fake_netuid)

    res1, res2 = result
    assert np.array_equal(res1, fake_uids)
    assert res2 == mocked_normalize_max_weight.return_value


def test_process_weights_with_all_zero_weights(mocker):
    """Test the process_weights_for_netuid function with all zero weights."""
    # Prep
    fake_uids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    fake_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    fake_netuid = 1
    fake_subtensor = mocker.MagicMock()
    fake_metagraph = mocker.MagicMock()
    fake_exclude_quantile = 0

    fake_subtensor.min_allowed_weights.return_value = 0.1
    fake_subtensor.max_weight_limit.return_value = 1.0
    fake_metagraph.n = 1

    # Call
    result = weight_utils.process_weights_for_netuid(
        uids=fake_uids,
        weights=fake_weights,
        netuid=fake_netuid,
        subtensor=fake_subtensor,
        metagraph=fake_metagraph,
        exclude_quantile=fake_exclude_quantile,
    )

    # Asserts
    fake_subtensor.min_allowed_weights.assert_called_once_with(netuid=fake_netuid)
    fake_subtensor.max_weight_limit.assert_called_once_with(netuid=fake_netuid)

    res1, res2 = result
    assert np.array_equal(res1, np.array([0]))
    assert np.array_equal(res2, np.array([1.0]))


def test_process_weights_for_netuid_with_nzs_less_min_allowed_weights(mocker):
    """Tests process_weights_for_netuid method when non-zero weights are less than the min allowed weights."""
    # Prep
    fake_uids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    fake_weights = np.array([0.1, 0.2, 0.3, 0.0, 0.0], dtype=np.float32)
    fake_netuid = 1
    fake_subtensor = mocker.MagicMock()
    fake_metagraph = None
    fake_exclude_quantile = 0

    fake_subtensor.min_allowed_weights.return_value = 4
    fake_subtensor.max_weight_limit.return_value = 1.0
    fake_subtensor.metagraph.return_value.n = 5
    mocked_np_arange = mocker.patch.object(np, "arange")
    mocked_normalize_max_weight = mocker.patch.object(
        weight_utils, "normalize_max_weight"
    )

    # Call
    result = weight_utils.process_weights_for_netuid(
        uids=fake_uids,
        weights=fake_weights,
        netuid=fake_netuid,
        subtensor=fake_subtensor,
        metagraph=fake_metagraph,
        exclude_quantile=fake_exclude_quantile,
    )

    # Asserts
    fake_subtensor.metagraph.assert_called_once_with(fake_netuid)
    fake_subtensor.min_allowed_weights.assert_called_once_with(netuid=fake_netuid)
    fake_subtensor.max_weight_limit.assert_called_once_with(netuid=fake_netuid)
    assert result == (
        mocked_np_arange.return_value,
        mocked_normalize_max_weight.return_value,
    )


def test_generate_weight_hash(mocker):
    """Tests weight_utils.generate_weight_hash function."""
    # Prep
    fake_address = "5D1ABCD"
    fake_netuid = 1
    fake_uids = [1, 2]
    fake_values = [10, 20]
    fake_version_key = 80000
    fake_salt = [1, 2]

    mocked_scale_bytes = mocker.patch.object(weight_utils, "ScaleBytes")
    mocked_keypair = mocker.patch.object(weight_utils, "Keypair")
    mocker_vec = mocker.patch.object(weight_utils, "Vec")
    mocked_u16 = mocker.patch.object(weight_utils, "U16")
    mocked_hasher = mocker.patch.object(weight_utils.hashlib, "blake2b")

    # Call
    result = weight_utils.generate_weight_hash(
        address=fake_address,
        netuid=fake_netuid,
        uids=fake_uids,
        values=fake_values,
        version_key=fake_version_key,
        salt=fake_salt,
    )

    # Asserts
    mocked_scale_bytes.assert_called()
    mocked_keypair.assert_called()
    mocker_vec.assert_called()
    mocked_u16.assert_called()
    assert (
        result
        == mocked_hasher.return_value.hexdigest.return_value.__radd__.return_value
    )
