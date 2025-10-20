from bittensor_wallet import Wallet
from scalecodec.types import GenericCall
import pytest
from bittensor.core.extrinsics.asyncex import crowdloan
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import Balance


@pytest.mark.asyncio
async def test_contribute_crowdloan_extrinsic(subtensor, mocker):
    """Test that `contribute_crowdloan_extrinsic` correctly constructs and submits the extrinsic."""
    # Preps
    faked_wallet = mocker.Mock(spec=Wallet)
    fake_crowdloan_id = mocker.Mock(spec=int)
    fake_amount = mocker.MagicMock(spec=Balance, rao=mocker.Mock(spec=int))

    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    success, message = await crowdloan.contribute_crowdloan_extrinsic(
        subtensor=subtensor,
        wallet=faked_wallet,
        crowdloan_id=fake_crowdloan_id,
        amount=fake_amount,
    )

    # Assertions
    mocked_compose_call.assert_awaited_once_with(
        call_module="Crowdloan",
        call_function="contribute",
        call_params=crowdloan.CrowdloanParams.contribute(
            fake_crowdloan_id, fake_amount
        ),
    )

    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=faked_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )

    assert success is True
    assert "Success" in message


@pytest.mark.asyncio
async def test_create_crowdloan_extrinsic(subtensor, mocker):
    """Test that `create_crowdloan_extrinsic` correctly constructs and submits the extrinsic."""
    # Preps
    faked_wallet = mocker.Mock(spec=Wallet)
    fake_deposit = mocker.MagicMock(spec=Balance, rao=mocker.Mock(spec=int))
    fake_min_contribution = mocker.MagicMock(spec=Balance, rao=mocker.Mock(spec=int))
    fake_cap = mocker.MagicMock(spec=Balance, rao=mocker.Mock(spec=int))
    fake_end = mocker.MagicMock(spec=int)
    fake_call = mocker.MagicMock(spec=GenericCall)
    fake_target_address = mocker.MagicMock(spec=str)

    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    success, message = await crowdloan.create_crowdloan_extrinsic(
        subtensor=subtensor,
        wallet=faked_wallet,
        deposit=fake_deposit,
        min_contribution=fake_min_contribution,
        cap=fake_cap,
        end=fake_end,
        call=fake_call,
        target_address=fake_target_address,
    )

    # Assertions
    mocked_compose_call.assert_awaited_once_with(
        call_module="Crowdloan",
        call_function="create",
        call_params=crowdloan.CrowdloanParams.create(
            fake_deposit,
            fake_min_contribution,
            fake_cap,
            fake_end,
            fake_call,
            fake_target_address,
        ),
    )

    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=faked_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )

    assert success is True
    assert "Success" in message


@pytest.mark.parametrize(
    "extrinsic, subtensor_function",
    [
        ("dissolve_crowdloan_extrinsic", "dissolve"),
        ("finalize_crowdloan_extrinsic", "finalize"),
        ("refund_crowdloan_extrinsic", "refund"),
        ("withdraw_crowdloan_extrinsic", "withdraw"),
    ],
)
@pytest.mark.asyncio
async def test_same_params_extrinsics(subtensor, mocker, extrinsic, subtensor_function):
    """Tests extrinsic with same parameters."""
    # Preps
    faked_wallet = mocker.Mock(spec=Wallet)
    fake_crowdloan_id = mocker.Mock(spec=int)

    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    success, message = await getattr(crowdloan, extrinsic)(
        subtensor=subtensor,
        wallet=faked_wallet,
        crowdloan_id=fake_crowdloan_id,
    )

    # Assertions
    mocked_compose_call.assert_awaited_once_with(
        call_module="Crowdloan",
        call_function=subtensor_function,
        call_params=getattr(crowdloan.CrowdloanParams, subtensor_function)(
            fake_crowdloan_id
        ),
    )

    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=faked_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )

    assert success is True
    assert "Success" in message


@pytest.mark.asyncio
async def test_update_cap_crowdloan_extrinsic(subtensor, mocker):
    """Test that `update_cap_crowdloan_extrinsic` correctly constructs and submits the extrinsic."""
    # Preps
    faked_wallet = mocker.Mock(spec=Wallet)
    fake_crowdloan_id = mocker.Mock(spec=int)
    fake_new_cap = mocker.MagicMock(spec=Balance, rao=mocker.Mock(spec=int))

    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    success, message = await crowdloan.update_cap_crowdloan_extrinsic(
        subtensor=subtensor,
        wallet=faked_wallet,
        crowdloan_id=fake_crowdloan_id,
        new_cap=fake_new_cap,
    )

    # Assertions
    mocked_compose_call.assert_awaited_once_with(
        call_module="Crowdloan",
        call_function="update_cap",
        call_params=crowdloan.CrowdloanParams.update_cap(
            fake_crowdloan_id, fake_new_cap
        ),
    )

    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=faked_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )

    assert success is True
    assert "Success" in message


@pytest.mark.asyncio
async def test_update_end_crowdloan_extrinsic(subtensor, mocker):
    """Test that `update_end_crowdloan_extrinsic` correctly constructs and submits the extrinsic."""
    # Preps
    faked_wallet = mocker.Mock(spec=Wallet)
    fake_crowdloan_id = mocker.Mock(spec=int)
    fake_new_end = mocker.MagicMock(spec=int)

    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    success, message = await crowdloan.update_end_crowdloan_extrinsic(
        subtensor=subtensor,
        wallet=faked_wallet,
        crowdloan_id=fake_crowdloan_id,
        new_end=fake_new_end,
    )

    # Assertions
    mocked_compose_call.assert_awaited_once_with(
        call_module="Crowdloan",
        call_function="update_end",
        call_params=crowdloan.CrowdloanParams.update_end(
            fake_crowdloan_id, fake_new_end
        ),
    )

    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=faked_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )

    assert success is True
    assert "Success" in message


@pytest.mark.asyncio
async def test_update_min_contribution_crowdloan_extrinsic(subtensor, mocker):
    """Test that `update_min_contribution_crowdloan_extrinsic` correctly constructs and submits the extrinsic."""
    # Preps
    faked_wallet = mocker.Mock(spec=Wallet)
    fake_crowdloan_id = mocker.Mock(spec=int)
    fake_new_min_contribution = mocker.MagicMock(
        spec=Balance, rao=mocker.Mock(spec=int)
    )

    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    success, message = await crowdloan.update_min_contribution_crowdloan_extrinsic(
        subtensor=subtensor,
        wallet=faked_wallet,
        crowdloan_id=fake_crowdloan_id,
        new_min_contribution=fake_new_min_contribution,
    )

    # Assertions
    mocked_compose_call.assert_awaited_once_with(
        call_module="Crowdloan",
        call_function="update_min_contribution",
        call_params=crowdloan.CrowdloanParams.update_min_contribution(
            fake_crowdloan_id, fake_new_min_contribution
        ),
    )

    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=faked_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )

    assert success is True
    assert "Success" in message
