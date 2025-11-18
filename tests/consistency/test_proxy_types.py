from bittensor.core.chain_data.proxy import ProxyType
from bittensor.core.extrinsics.pallets import SubtensorModule, Proxy, Balances


def get_proxy_type_fields(meta):
    """Returns list of fields for ProxyType enum from substrate metadata."""
    type_name = "ProxyType"
    fields = []
    for item in meta.portable_registry["types"].value:
        type_ = item.get("type")
        name = None
        if len(type_.get("path")) > 1:
            name = type_.get("path")[1]

        if name == type_name:
            variants = type_.get("def").get("variant").get("variants")
            fields = [v.get("name") for v in variants]
    return fields


def test_make_sure_proxy_type_has_all_fields(subtensor, alice_wallet):
    """Tests that SDK ProxyType have all fields defined in the ProxyType enum."""

    chain_proxy_type_fields = get_proxy_type_fields(subtensor.substrate.metadata)

    assert len(chain_proxy_type_fields) == len(ProxyType)
    assert set(chain_proxy_type_fields) == set(ProxyType.all_types())
