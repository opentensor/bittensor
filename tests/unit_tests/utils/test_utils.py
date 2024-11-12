# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
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

from bittensor_wallet import Wallet
import pytest

from bittensor import utils
from bittensor.core.settings import SS58_FORMAT


def test_ss58_to_vec_u8(mocker):
    """Tests `utils.ss58_to_vec_u8` function."""
    # Prep
    test_ss58_address = "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
    fake_return = b"2\xa6?"
    mocked_ss58_address_to_bytes = mocker.patch.object(
        utils, "ss58_address_to_bytes", return_value=fake_return
    )

    # Call
    result = utils.ss58_to_vec_u8(test_ss58_address)

    # Asserts
    mocked_ss58_address_to_bytes.assert_called_once_with(test_ss58_address)
    assert result == [int(byte) for byte in fake_return]


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("y", True),
        ("yes", True),
        ("t", True),
        ("true", True),
        ("on", True),
        ("1", True),
        ("n", False),
        ("no", False),
        ("f", False),
        ("false", False),
        ("off", False),
        ("0", False),
    ],
)
def test_strtobool(test_input, expected):
    """Test truthy values."""
    assert utils.strtobool(test_input) is expected


@pytest.mark.parametrize(
    "test_input",
    [
        "maybe",
        "2",
        "onoff",
    ],
)
def test_strtobool_raise_error(test_input):
    """Tests invalid values."""
    with pytest.raises(ValueError):
        utils.strtobool(test_input)


def test_get_explorer_root_url_by_network_from_map():
    """Tests private utils._get_explorer_root_url_by_network_from_map function."""
    # Prep
    # Test with a known network
    network_map = {
        "entity1": {"network1": "url1", "network2": "url2"},
        "entity2": {"network1": "url3", "network3": "url4"},
    }
    # Test with no matching network in the map
    network_map_empty = {
        "entity1": {},
        "entity2": {},
    }

    # Assertions
    assert utils._get_explorer_root_url_by_network_from_map(
        "network1", network_map
    ) == {
        "entity1": "url1",
        "entity2": "url3",
    }
    # Test with an unknown network
    assert (
        utils._get_explorer_root_url_by_network_from_map("unknown_network", network_map)
        == {}
    )
    assert (
        utils._get_explorer_root_url_by_network_from_map("network1", network_map_empty)
        == {}
    )


def test_get_explorer_url_for_network():
    """Tests `utils.get_explorer_url_for_network` function."""
    # Prep
    fake_block_hash = "0x1234567890abcdef"
    fake_map = {"opentensor": {"network": "url"}, "taostats": {"network": "url2"}}

    # Call
    result = utils.get_explorer_url_for_network("network", fake_block_hash, fake_map)

    # Assert
    assert result == {
        "opentensor": f"url/query/{fake_block_hash}",
        "taostats": f"url2/extrinsic/{fake_block_hash}",
    }


def test_ss58_address_to_bytes(mocker):
    """Tests utils.ss58_address_to_bytes function."""
    # Prep
    fake_ss58_address = "ss58_address"
    mocked_scalecodec_ss58_decode = mocker.patch.object(
        utils.scalecodec, "ss58_decode", return_value=""
    )

    # Call
    result = utils.ss58_address_to_bytes(fake_ss58_address)

    # Asserts
    mocked_scalecodec_ss58_decode.assert_called_once_with(
        fake_ss58_address, SS58_FORMAT
    )
    assert result == bytes.fromhex(mocked_scalecodec_ss58_decode.return_value)


@pytest.mark.parametrize(
    "test_input, expected_result",
    [
        (123, False),
        ("0x234SD", True),
        ("5D34SD", True),
        (b"0x234SD", True),
    ],
)
def test_is_valid_bittensor_address_or_public_key(mocker, test_input, expected_result):
    """Tests utils.is_valid_bittensor_address_or_public_key function."""
    # Prep
    mocked_is_valid_ed25519_pubkey = mocker.patch.object(
        utils, "_is_valid_ed25519_pubkey", return_value=True
    )
    mocked_ss58_is_valid_ss58_address = mocker.patch.object(
        utils.ss58, "is_valid_ss58_address", side_effect=[False, True]
    )

    # Call
    result = utils.is_valid_bittensor_address_or_public_key(test_input)

    # Asserts
    if not isinstance(test_input, int) and isinstance(test_input, bytes):
        mocked_is_valid_ed25519_pubkey.assert_called_with(test_input)
    if isinstance(test_input, str) and not test_input.startswith("0x"):
        assert mocked_ss58_is_valid_ss58_address.call_count == 2
    assert result == expected_result


@pytest.mark.parametrize(
    "unlock_type, wallet_method",
    [
        ("coldkey", "unlock_coldkey"),
        ("hotkey", "unlock_hotkey"),
    ],
)
def test_unlock_key(mocker, unlock_type, wallet_method):
    """Test the unlock key function."""
    # Preps
    mock_wallet = mocker.Mock(autospec=Wallet)

    # Call
    result = utils.unlock_key(mock_wallet, unlock_type=unlock_type)

    # Asserts
    getattr(mock_wallet, wallet_method).assert_called_once()
    assert result == utils.UnlockStatus(True, "")


def test_unlock_key_raise_value_error(mocker):
    """Test the unlock key function raises ValueError."""
    with pytest.raises(ValueError):
        utils.unlock_key(wallet=mocker.Mock(autospec=Wallet), unlock_type="coldkeypub")


@pytest.mark.parametrize(
    "side_effect, response",
    [
        (
            utils.KeyFileError("Simulated KeyFileError exception"),
            utils.UnlockStatus(
                False,
                "Coldkey keyfile is corrupt, non-writable, or non-readable, or non-existent.",
            ),
        ),
        (
            utils.PasswordError("Simulated PasswordError exception"),
            utils.UnlockStatus(
                False, "The password used to decrypt your Coldkey keyfile is invalid."
            ),
        ),
    ],
    ids=["PasswordError", "KeyFileError"],
)
def test_unlock_key_errors(mocker, side_effect, response):
    """Test the unlock key function handle the errors."""
    mock_wallet = mocker.Mock(autospec=Wallet)
    mock_wallet.unlock_coldkey.side_effect = side_effect
    result = utils.unlock_key(wallet=mock_wallet)

    assert result == response
