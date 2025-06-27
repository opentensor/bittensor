import pytest

from bittensor.core.chain_data import utils


@pytest.mark.parametrize(
    "metadata, response",
    [
        (
            {
                "deposit": 0,
                "block": 5415815,
                "info": {
                    "fields": (
                        (
                            {
                                "Raw64": (
                                    (
                                        51,
                                        98,
                                        99,
                                        54,
                                        49,
                                        48,
                                        57,
                                        102,
                                        49,
                                        101,
                                        49,
                                        51,
                                        102,
                                        102,
                                        56,
                                        102,
                                        55,
                                        101,
                                        98,
                                        54,
                                        97,
                                        102,
                                        54,
                                        49,
                                        53,
                                        101,
                                        49,
                                        102,
                                        56,
                                        101,
                                        49,
                                        55,
                                        99,
                                        57,
                                        97,
                                        100,
                                        100,
                                        48,
                                        97,
                                        50,
                                        56,
                                        98,
                                        99,
                                        48,
                                        50,
                                        54,
                                        55,
                                        57,
                                        52,
                                        99,
                                        56,
                                        54,
                                        97,
                                        101,
                                        50,
                                        56,
                                        57,
                                        57,
                                        50,
                                        99,
                                        102,
                                        48,
                                        52,
                                        53,
                                    ),
                                )
                            },
                        ),
                    )
                },
            },
            "3bc6109f1e13ff8f7eb6af615e1f8e17c9add0a28bc026794c86ae28992cf045",
        ),
        (
            {
                "deposit": 0,
                "block": 5866237,
                "info": {"fields": (({"ResetBondsFlag": ()},),)},
            },
            "",
        ),
    ],
)
def test_decode_metadata(metadata, response):
    assert utils.decode_metadata(metadata) == response
