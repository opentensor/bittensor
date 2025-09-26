import pytest
from bittensor.core.extrinsics.asyncex import staking


@pytest.mark.parametrize(
    "res_success, res_message",
    [
        (True, ""),
        (False, "Error"),
    ],
)
@pytest.mark.asyncio
async def test_set_auto_stake_extrinsic(
    subtensor, fake_wallet, mocker, res_success, res_message
):
    """Verify that `set_auto_stake_extrinsic` function calls proper methods."""
    # Preps
    netuid = mocker.Mock()
    hotkey_ss58 = mocker.Mock()
    mocked_unlock_key = mocker.patch.object(
        staking, "unlock_key", return_value=mocker.Mock(success=True, message="True")
    )

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")

    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=(res_success, res_message)
    )

    # Call
    success, message = await staking.set_auto_stake_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
    )

    # Asserts
    mocked_unlock_key.assert_called_once_with(fake_wallet, raise_error=False)
    mocked_compose_call.assert_awaited_once_with(
        call_module="SubtensorModule",
        call_function="set_coldkey_auto_stake_hotkey",
        call_params={"netuid": netuid, "hotkey": hotkey_ss58},
    )
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert success is res_success
    assert message == res_message
