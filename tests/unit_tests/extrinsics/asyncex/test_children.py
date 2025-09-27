import pytest

from bittensor.core.extrinsics.asyncex import children
from bittensor.core.types import ExtrinsicResponse


@pytest.mark.asyncio
async def test_set_children_extrinsic(subtensor, mocker, fake_wallet):
    """Test that set_children_extrinsic correctly constructs and submits the extrinsic."""
    # Preps
    hotkey = "fake hotkey"
    netuid = 123
    fake_children = [
        (
            1.0,
            "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
        ),
    ]

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    success, message = await children.set_children_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey=hotkey,
        netuid=netuid,
        children=fake_children,
    )

    # Asserts
    assert success is True
    assert "Success" in message

    mocked_compose_call.assert_awaited_once_with(
        call_module="SubtensorModule",
        call_function="set_children",
        call_params={
            "children": [
                (
                    18446744073709551615,
                    "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
                ),
            ],
            "hotkey": "fake hotkey",
            "netuid": netuid,
        },
    )

    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )


@pytest.mark.asyncio
async def test_root_set_pending_childkey_cooldown_extrinsic(
    subtensor, mocker, fake_wallet
):
    """Verify root_set_pending_childkey_cooldown_extrinsic extrinsic."""
    # Preps
    cooldown = 100

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    success, message = await children.root_set_pending_childkey_cooldown_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        cooldown=cooldown,
    )
    # Asserts

    assert mocked_compose_call.call_count == 2
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        period=None,
        nonce_key="hotkey",
        sign_with="coldkey",
        use_nonce=False,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success is True
    assert "Success" in message
