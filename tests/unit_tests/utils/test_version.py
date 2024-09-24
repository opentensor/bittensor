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
from datetime import datetime, timedelta, timezone

# from bittensor.utils.version import (
#     VERSION_CHECK_THRESHOLD,
#     VersionCheckError,
#     get_and_save_latest_version,
#     check_version,
#     version_checking,
#     __version__
# )
from bittensor.utils import version

from unittest.mock import MagicMock
from pytest_mock import MockerFixture


@pytest.fixture
def pypi_version():
    return "6.9.3"


@pytest.fixture
def mock_get_version_from_pypi(mocker: MockerFixture, pypi_version: str):
    return mocker.patch(
        "bittensor.utils.version._get_version_from_pypi",
        return_value=pypi_version,
        autospec=True,
    )


@pytest.fixture
def version_file_path(mocker: MockerFixture, tmp_path: Path):
    file_path = tmp_path / ".version"

    mocker.patch(
        "bittensor.utils.version._get_version_file_path", return_value=file_path
    )
    return file_path


def test_get_and_save_latest_version_no_file(
    mock_get_version_from_pypi: MagicMock, version_file_path: Path, pypi_version: str
):
    assert not version_file_path.exists()

    assert version.get_and_save_latest_version() == pypi_version

    mock_get_version_from_pypi.assert_called_once()
    assert version_file_path.exists()
    assert version_file_path.read_text() == pypi_version


@pytest.mark.parametrize("elapsed", [0, version.VERSION_CHECK_THRESHOLD - 5])
def test_get_and_save_latest_version_file_fresh_check(
    mock_get_version_from_pypi: MagicMock, version_file_path: Path, elapsed: int
):
    now = datetime.now(timezone.utc)

    version_file_path.write_text("6.9.5")

    with freeze_time(now + timedelta(seconds=elapsed)):
        assert version.get_and_save_latest_version() == "6.9.5"

    mock_get_version_from_pypi.assert_not_called()


def test_get_and_save_latest_version_file_expired_check(
    mock_get_version_from_pypi: MagicMock, version_file_path: Path, pypi_version: str
):
    now = datetime.now(timezone.utc)

    version_file_path.write_text("6.9.5")

    with freeze_time(now + timedelta(seconds=version.VERSION_CHECK_THRESHOLD + 1)):
        assert version.get_and_save_latest_version() == pypi_version

    mock_get_version_from_pypi.assert_called_once()
    assert version_file_path.read_text() == pypi_version


@pytest.mark.parametrize(
    ("current_version", "latest_version"),
    [
        ("6.9.3", "6.9.4"),
        ("6.9.3a1", "6.9.3a2"),
        ("6.9.3a1", "6.9.3b1"),
        ("6.9.3", "6.10"),
        ("6.9.3", "7.0"),
        ("6.0.15", "6.1.0"),
    ],
)
def test_check_version_newer_available(
    mocker: MockerFixture, current_version: str, latest_version: str, capsys
):
    version.__version__ = current_version
    mocker.patch(
        "bittensor.utils.version.get_and_save_latest_version",
        return_value=latest_version,
    )

    version.check_version()

    captured = capsys.readouterr()

    assert "update" in captured.out
    assert current_version in captured.out
    assert latest_version in captured.out


@pytest.mark.parametrize(
    ("current_version", "latest_version"),
    [
        ("6.9.3", "6.9.3"),
        ("6.9.3", "6.9.2"),
        ("6.9.3b", "6.9.3a"),
    ],
)
def test_check_version_up_to_date(
    mocker: MockerFixture, current_version: str, latest_version: str, capsys
):
    version.__version__ = current_version
    mocker.patch(
        "bittensor.utils.version.get_and_save_latest_version",
        return_value=latest_version,
    )

    version.check_version()

    captured = capsys.readouterr()

    assert captured.out == ""


def test_version_checking(mocker: MockerFixture):
    mock = mocker.patch("bittensor.utils.version.check_version")

    version.version_checking()

    mock.assert_called_once()


def test_version_checking_exception(mocker: MockerFixture):
    mock = mocker.patch(
        "bittensor.utils.version.check_version", side_effect=version.VersionCheckError
    )

    version.version_checking()

    mock.assert_called_once()
