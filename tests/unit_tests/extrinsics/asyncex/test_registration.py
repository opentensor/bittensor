import pytest
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import Balance

from bittensor.core.extrinsics.asyncex import registration as async_registration


@pytest.mark.asyncio
async def test_set_subnet_identity_extrinsic_is_success(subtensor, fake_wallet, mocker):
    """Verify that set_subnet_identity_extrinsic calls the correct functions and returns the correct result."""
    # Preps
    netuid = 123
    subnet_name = "mock_subnet_name"
    github_repo = "mock_github_repo"
    subnet_contact = "mock_subnet_contact"
    subnet_url = "mock_subnet_url"
    logo_url = "mock_logo_url"
    discord = "mock_discord"
    description = "mock_description"
    additional = "mock_additional"

    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")

    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    result = await async_registration.set_subnet_identity_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        subnet_name=subnet_name,
        github_repo=github_repo,
        subnet_contact=subnet_contact,
        subnet_url=subnet_url,
        logo_url=logo_url,
        discord=discord,
        description=description,
        additional=additional,
    )

    # Asserts
    mocked_compose_call.assert_awaited_once_with(
        call_module="SubtensorModule",
        call_function="set_subnet_identity",
        call_params={
            "netuid": netuid,
            "subnet_name": subnet_name,
            "github_repo": github_repo,
            "subnet_contact": subnet_contact,
            "subnet_url": subnet_url,
            "logo_url": logo_url,
            "discord": discord,
            "description": description,
            "additional": additional,
        },
    )
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )

    assert result == mocked_sign_and_send_extrinsic.return_value


@pytest.mark.asyncio
async def test_set_subnet_identity_extrinsic_is_failed(subtensor, fake_wallet, mocker):
    """Verify that set_subnet_identity_extrinsic calls the correct functions and returns False with bad result."""
    # Preps
    netuid = 123
    subnet_name = "mock_subnet_name"
    github_repo = "mock_github_repo"
    subnet_contact = "mock_subnet_contact"
    subnet_url = "mock_subnet_url"
    logo_url = "mock_logo_url"
    discord = "mock_discord"
    description = "mock_description"
    additional = "mock_additional"
    fake_error_message = "error message"

    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")

    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
    )

    # Call
    result = await async_registration.set_subnet_identity_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        subnet_name=subnet_name,
        github_repo=github_repo,
        subnet_contact=subnet_contact,
        subnet_url=subnet_url,
        logo_url=logo_url,
        discord=discord,
        description=description,
        additional=additional,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_compose_call.assert_awaited_once_with(
        call_module="SubtensorModule",
        call_function="set_subnet_identity",
        call_params={
            "netuid": netuid,
            "subnet_name": subnet_name,
            "github_repo": github_repo,
            "subnet_contact": subnet_contact,
            "subnet_url": subnet_url,
            "logo_url": logo_url,
            "discord": discord,
            "description": description,
            "additional": additional,
        },
    )
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )

    assert result == mocked_sign_and_send_extrinsic.return_value


@pytest.mark.parametrize(
    "subnet_exists, neuron_is_null, recycle_success, is_registered, expected_result, test_id",
    [
        # Happy paths
        (True, False, None, None, True, "neuron-not-null"),
        (True, True, True, True, True, "happy-path-wallet-registered"),
        # Error paths
        (False, True, False, None, False, "subnet-non-existence"),
        (True, True, False, False, False, "error-path-recycling-failed"),
        (True, True, True, False, False, "error-path-not-registered"),
    ],
)
@pytest.mark.asyncio
async def test_register_limit_extrinsic(
    subtensor,
    fake_wallet,
    subnet_exists,
    neuron_is_null,
    recycle_success,
    is_registered,
    expected_result,
    test_id,
    mocker,
):
    # Arrange
    fake_wallet.hotkey.ss58_address = "hotkey_ss58"
    fake_wallet.coldkeypub.ss58_address = "coldkey_ss58"

    mocker.patch.object(subtensor, "subnet_exists", return_value=subnet_exists)
    mocker.patch.object(
        subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=mocker.Mock(is_null=neuron_is_null),
    )
    mocker.patch.object(subtensor, "get_balance", return_value=mocker.Mock())
    mocker.patch.object(subtensor, "recycle", return_value=mocker.Mock())
    mocker.patch.object(subtensor, "compose_call")
    mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(recycle_success, "Mock error message"),
    )
    mocker.patch.object(subtensor, "is_hotkey_registered", return_value=is_registered)

    # Act
    result = await async_registration.register_limit_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=123,
        limit_price=Balance.from_rao(1000000000),
    )
    # Assert
    assert result.success == expected_result, f"Test failed for test_id: {test_id}"
