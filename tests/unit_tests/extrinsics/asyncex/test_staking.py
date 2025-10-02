import pytest

from bittensor.core.extrinsics.asyncex import staking
from bittensor.core.types import ExtrinsicResponse


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


    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")

    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=ExtrinsicResponse(res_success, res_message)
    )

    # Call
    success, message = await staking.set_auto_stake_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
    )

    # Asserts
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
