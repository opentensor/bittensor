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

from bittensor.utils.wallet_utils import decode_hex_identity_dict


def test_decode_hex_identity_dict():
    """Tests the decode_hex_identity_dict function to ensure it correctly decodes a hex-encoded identity dictionary."""
    # Prep
    fake_info_dictionary = {
        "additional": [
            (
                {"Raw11": "0x6465736372697074696f6e"},
                {
                    "Raw64": "0x46616b65206465736372697074696f6e"
                },
            )
        ],
        "display": {"Raw17": "0x46616b654e616d65"},
        "legal": "None",
        "web": {"Raw22": "0x687474703a2f2f7777772e626c61626c612d746573742e636f6d"},
        "riot": "None",
        "email": "None",
        "pgp_fingerprint": None,
        "image": "None",
        "twitter": "None",
    }

    expected_result = {
        "additional": [
            ("description", "Fake description")
        ],
        "display": "FakeName",
        "legal": "None",
        "web": "http://www.blabla-test.com",
        "riot": "None",
        "email": "None",
        "pgp_fingerprint": None,
        "image": "None",
        "twitter": "None",
    }

    # Call

    result = decode_hex_identity_dict(fake_info_dictionary)

    # Assertions
    assert result == expected_result


