import pytest

from bittensor.core.chain_data import utils


@pytest.mark.parametrize(
    "metadata, response",
    [
        (
            {
                "block": 5097676,
                "deposit": 0,
                "info": {
                    "fields": [
                        {
                            "Raw97": "0x7b27706565725f6964273a2027313244334b6f6f57524e7735344157347a725157655a4c32627568553850373167666f3950585151414855774541653468413334272c20276d6f64656c5f68756767696e67666163655f6964273a204e6f6e657d"
                        }
                    ]
                },
            },
            "{'peer_id': '12D3KooWRNw54AW4zrQWeZL2buhU8P71gfo9PXQQAHUwEAe4hA34', 'model_huggingface_id': None}",
        ),
        (
            {"block": 6161535, "deposit": 0, "info": {"fields": ["ResetBondsFlag"]}},
            "",
        ),
    ],
)
def test_decode_metadata(metadata, response):
    assert utils.decode_metadata(metadata) == response
