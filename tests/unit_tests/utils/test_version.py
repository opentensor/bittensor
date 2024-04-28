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

from pathlib import Path
import pytest
from freezegun import freeze_time
from datetime import datetime, timedelta

from bittensor.utils.version import VERSION_CHECK_THRESHOLD, get_and_save_latest_version
from unittest.mock import MagicMock
from pytest_mock import MockerFixture


@pytest.fixture
def pypi_version():
    return "6.9.3"


@pytest.fixture
def mock_get_version_from_pypi(mocker: MockerFixture, pypi_version: str):
    return mocker.patch("bittensor.utils.version._get_version_from_pypi", return_value=pypi_version, autospec=True)


@pytest.fixture
def version_file_path(mocker: MockerFixture, tmp_path: Path):
    file_path = tmp_path / ".version"

    mocker.patch("bittensor.utils.version._get_version_file_path", return_value=file_path)
    return file_path


def test_get_and_save_latest_version_no_file(mock_get_version_from_pypi: MagicMock, version_file_path: Path, pypi_version: str):
    assert not version_file_path.exists()

    assert get_and_save_latest_version() == pypi_version

    mock_get_version_from_pypi.assert_called_once()
    assert version_file_path.exists()
    assert version_file_path.read_text() == pypi_version


@pytest.mark.parametrize('elapsed', [0, VERSION_CHECK_THRESHOLD - 5])
def test_get_and_save_latest_version_file_fresh_check(mock_get_version_from_pypi: MagicMock, version_file_path: Path, elapsed: int):
    now = datetime.utcnow()

    version_file_path.write_text("6.9.5")

    with freeze_time(now + timedelta(seconds=elapsed)):
        assert get_and_save_latest_version() == "6.9.5"

    mock_get_version_from_pypi.assert_not_called()


def test_get_and_save_latest_version_file_expired_check(mock_get_version_from_pypi: MagicMock, version_file_path: Path, pypi_version: str):
    now = datetime.utcnow()

    version_file_path.write_text("6.9.5")

    with freeze_time(now + timedelta(seconds=VERSION_CHECK_THRESHOLD + 1)):
        assert get_and_save_latest_version() == pypi_version

    mock_get_version_from_pypi.assert_called_once()
    assert version_file_path.read_text() == pypi_version
