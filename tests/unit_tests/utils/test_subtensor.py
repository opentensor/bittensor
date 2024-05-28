# The MIT License (MIT)
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

import json

import pytest

import bittensor.utils.subtensor as st_utils


class MockPallet:
    def __init__(self, errors):
        self.errors = errors


@pytest.fixture
def pallet_with_errors():
    """Provide a mock pallet with sample errors."""
    return MockPallet(
        [
            {"index": 1, "name": "ErrorOne", "docs": ["Description one."]},
            {
                "index": 2,
                "name": "ErrorTwo",
                "docs": ["Description two.", "Continued."],
            },
        ]
    )


@pytest.fixture
def empty_pallet():
    """Provide a mock pallet with no errors."""
    return MockPallet([])


def test_get_errors_from_pallet_with_errors(pallet_with_errors):
    """Ensure errors are correctly parsed from pallet."""
    expected = {
        "1": {"name": "ErrorOne", "description": "Description one."},
        "2": {"name": "ErrorTwo", "description": "Description two. Continued."},
    }
    assert st_utils._get_errors_from_pallet(pallet_with_errors) == expected


def test_get_errors_from_pallet_empty(empty_pallet):
    """Test behavior with an empty list of errors."""
    assert st_utils._get_errors_from_pallet(empty_pallet) is None


def test_save_errors_to_cache(tmp_path):
    """Ensure that errors are correctly saved to a file."""
    test_file = tmp_path / "subtensor_errors_map.json"
    errors = {"1": {"name": "ErrorOne", "description": "Description one."}}
    st_utils._ERRORS_FILE_PATH = test_file
    st_utils._save_errors_to_cache("0x123", errors)

    with open(test_file, "r") as file:
        data = json.load(file)
        assert data["subtensor_build_id"] == "0x123"
        assert data["errors"] == errors


def test_get_errors_from_cache(tmp_path):
    """Test retrieval of errors from cache."""
    test_file = tmp_path / "subtensor_errors_map.json"
    errors = {"1": {"name": "ErrorOne", "description": "Description one."}}

    st_utils._ERRORS_FILE_PATH = test_file
    with open(test_file, "w") as file:
        json.dump({"subtensor_build_id": "0x123", "errors": errors}, file)
    assert st_utils._get_errors_from_cache() == {
        "subtensor_build_id": "0x123",
        "errors": errors,
    }


def test_get_errors_no_cache(mocker, empty_pallet):
    """Test get_errors function when no cache is available."""
    mocker.patch("bittensor.utils.subtensor._get_errors_from_cache", return_value=None)
    mocker.patch("bittensor.utils.subtensor.SubstrateInterface")
    substrate_mock = mocker.MagicMock()
    substrate_mock.metadata.get_metadata_pallet.return_value = empty_pallet
    substrate_mock.metadata[0].value = "0x123"
    assert st_utils.get_subtensor_errors(substrate_mock) == {}
