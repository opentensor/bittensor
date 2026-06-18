from bittensor.core.chain_data.proxy import ProxyType


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


def test_proxy_types_match_runtime_api(subtensor, alice_wallet):
    """Tests that SDK ProxyType enum matches ProxyFilterRuntimeApi.getProxyTypes."""
    runtime_types = subtensor.proxies.get_proxy_types()

    runtime_names = {rt.name for rt in runtime_types}

    for rt in runtime_types:
        assert ProxyType.is_valid(rt.name), (
            f"Runtime proxy type '{rt.name}' (index={rt.index}) not in SDK ProxyType enum"
        )

    for pt in ProxyType:
        assert pt.value in runtime_names, (
            f"SDK ProxyType.{pt.value} not found in runtime getProxyTypes response"
        )


def test_proxy_filter_returns_valid_data(subtensor, alice_wallet):
    """Tests that get_proxy_filter() returns valid filter data for all types."""
    filters = subtensor.proxies.get_proxy_filter()

    assert len(filters) > 0
    valid_modes = {"AllowAll", "DenyAll", "Allow", "Deny"}

    for f in filters:
        assert f.filter_mode in valid_modes, (
            f"Invalid filter_mode '{f.filter_mode}' for {f.name}"
        )
        if f.filter_mode in ("AllowAll", "DenyAll"):
            assert f.calls == [], f"{f.name}: {f.filter_mode} should have empty calls"
        else:
            assert len(f.calls) > 0, (
                f"{f.name}: {f.filter_mode} should have non-empty calls"
            )
