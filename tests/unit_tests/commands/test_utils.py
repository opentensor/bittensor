# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import pytest
from bittensor.commands.utils import get_delegates_details
import bittensor


def test_get_delegates_details_success(mocker):
    """Tests the get_delegates_details function to ensure it successfully retrieves delegate details."""
    # Prep
    mocked_sub = mocker.MagicMock()
    mocker.patch("bittensor.subtensor", return_value=mocked_sub)

    # Call

    fake_subtensor = bittensor.subtensor()
    result = get_delegates_details(fake_subtensor)

    # Assertions
    mocked_sub.get_delegate_identities.assert_called_once()
    assert result == mocked_sub.get_delegate_identities.return_value


def test_get_delegates_details_error(mocker):
    """Tests the get_delegates_details function to ensure it handles errors correctly."""
    # Prep
    test_error_message = "Test exception"
    mocked_sub = mocker.MagicMock()
    mocked_sub.get_delegate_identities.side_effect = Exception(test_error_message)

    mocker.patch("bittensor.subtensor", return_value=mocked_sub)
    mocker.patch("bittensor.logging.exception")

    # Call
    fake_subtensor = bittensor.subtensor()
    result = get_delegates_details(fake_subtensor)

    # Assertions
    mocked_sub.get_delegate_identities.assert_called_once()
    bittensor.logging.exception.assert_called_once_with(f"Unable to get Delegates Identities. Error: {test_error_message}")
    assert result is None
