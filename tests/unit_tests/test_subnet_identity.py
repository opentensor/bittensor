import pytest

from bittensor.core.chain_data.subnet_identity import SubnetIdentity


def test_subnet_identity_x_handle_parsing():
    # 1. Legacy Data (with NO x_handle)
    legacy_data = {
        "subnet_name": "Old",
        "github_repo": "http://g",
        "subnet_contact": "c",
        "subnet_url": "http://u",
        "logo_url": "http://l",
        "discord": "http://d",
        "description": "d",
        "additional": "a",
    }
    identity = SubnetIdentity._from_dict(legacy_data)
    assert identity.x_handle is None

    # 2. New Data (with x_handle)
    new_data = legacy_data.copy()
    new_data["x_handle"] = "@test"
    identity = SubnetIdentity._from_dict(new_data)
    assert identity.x_handle == "@test"

    # 3. Explicit None
    none_data = legacy_data.copy()
    none_data["x_handle"] = None
    identity = SubnetIdentity._from_dict(none_data)
    assert identity.x_handle is None

    # 4. Empty string (with x_handle)
    new_data = legacy_data.copy()
    new_data["x_handle"] = ""
    identity = SubnetIdentity._from_dict(new_data)
    assert identity.x_handle == ""

    # 5. Extra fields (forward Compatibility)
    garbage_data = legacy_data.copy()
    garbage_data["x_handle"] = "@future_proof"
    garbage_data["y_handle"] = "ignore_me"
    identity = SubnetIdentity._from_dict(garbage_data)
    assert identity.x_handle == "@future_proof"

    # 6. Direct Initialization (constructor test to verify x_handle defaults to None)
    identity_manual = SubnetIdentity(
        subnet_name="Manual",
        github_repo=".",
        subnet_contact=".",
        subnet_url=".",
        logo_url=".",
        discord=".",
        description=".",
        additional=".",
    )
    assert identity_manual.x_handle is None
