import pytest
from bittensor.core.types import ExtrinsicResponse

from bittensor.core.extrinsics.asyncex import registration as async_registration


@pytest.mark.asyncio
async def test_register_extrinsic_success(subtensor, fake_wallet, mocker):
    """Tests successful registration."""
    # Preps
    fake_wallet.hotkey.ss58_address = "hotkey_ss58"
    fake_wallet.coldkey.ss58_address = "coldkey_ss58"

    mocked_subnet_exists = mocker.patch.object(
        subtensor, "subnet_exists", return_value=True
    )
    mocked_get_neuron = mocker.patch.object(
        subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=mocker.Mock(is_null=True),
    )
    mocked_create_pow = mocker.patch.object(
        async_registration,
        "create_pow_async",
        return_value=mocker.Mock(
            is_stale_async=mocker.AsyncMock(return_value=False), seal=[]
        ),
    )
    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=ExtrinsicResponse(True, "")
    )
    mocked_is_hotkey_registered = mocker.patch.object(
        subtensor, "is_hotkey_registered", return_value=True
    )

    # Call
    result = await async_registration.register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=1,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_subnet_exists.assert_called_once_with(
        1,
        block_hash=subtensor.substrate.get_chain_head.return_value,
    )
    mocked_get_neuron.assert_called_once_with(
        hotkey_ss58="hotkey_ss58",
        netuid=1,
        block_hash=subtensor.substrate.get_chain_head.return_value,
    )
    mocked_create_pow.assert_called_once()
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )
    mocked_is_hotkey_registered.assert_called_once_with(
        netuid=1, hotkey_ss58="hotkey_ss58"
    )

    assert result[0]


@pytest.mark.asyncio
async def test_register_extrinsic_success_with_cuda(subtensor, fake_wallet, mocker):
    """Tests successful registration with CUDA enabled."""
    # Preps
    fake_wallet.hotkey.ss58_address = "hotkey_ss58"
    fake_wallet.coldkey.ss58_address = "coldkey_ss58"

    mocked_subnet_exists = mocker.patch.object(
        subtensor, "subnet_exists", return_value=True
    )
    mocked_get_neuron = mocker.patch.object(
        subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=mocker.Mock(is_null=True),
    )
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocked_create_pow = mocker.patch.object(
        async_registration,
        "create_pow_async",
        return_value=mocker.Mock(
            is_stale_async=mocker.AsyncMock(return_value=False), seal=[]
        ),
    )
    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=ExtrinsicResponse(True, "")
    )
    mocked_is_hotkey_registered = mocker.patch.object(
        subtensor, "is_hotkey_registered", return_value=True
    )

    # Call
    result = await async_registration.register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=1,
        cuda=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_subnet_exists.assert_called_once_with(
        1,
        block_hash=subtensor.substrate.get_chain_head.return_value,
    )
    mocked_get_neuron.assert_called_once_with(
        hotkey_ss58="hotkey_ss58",
        netuid=1,
        block_hash=subtensor.substrate.get_chain_head.return_value,
    )
    mocked_create_pow.assert_called_once()
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )
    mocked_is_hotkey_registered.assert_called_once_with(
        netuid=1, hotkey_ss58="hotkey_ss58"
    )
    assert result[0]


@pytest.mark.asyncio
async def test_register_extrinsic_failed_with_cuda(subtensor, fake_wallet, mocker):
    """Tests failed registration with CUDA enabled."""
    # Preps
    fake_wallet.hotkey.ss58_address = "hotkey_ss58"
    fake_wallet.coldkey.ss58_address = "coldkey_ss58"

    mocked_subnet_exists = mocker.patch.object(
        subtensor, "subnet_exists", return_value=True
    )
    mocked_get_neuron = mocker.patch.object(
        subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=mocker.Mock(is_null=True),
    )
    mocker.patch("torch.cuda.is_available", return_value=False)

    # Call
    result = await async_registration.register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=1,
        cuda=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_subnet_exists.assert_called_once_with(
        1,
        block_hash=subtensor.substrate.get_chain_head.return_value,
    )
    mocked_get_neuron.assert_called_once_with(
        hotkey_ss58="hotkey_ss58",
        netuid=1,
        block_hash=subtensor.substrate.get_chain_head.return_value,
    )
    assert result == ExtrinsicResponse(
        False,
        "CUDA not available.",
        extrinsic_function="register_extrinsic",
    )


@pytest.mark.asyncio
async def test_register_extrinsic_subnet_not_exists(subtensor, fake_wallet, mocker):
    """Tests registration when subnet does not exist."""
    # Preps
    netuid = 14
    mocked_subnet_exists = mocker.patch.object(
        subtensor, "subnet_exists", return_value=False
    )

    # Call
    result = await async_registration.register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
    )

    # Asserts
    mocked_subnet_exists.assert_called_once_with(
        netuid,
        block_hash=subtensor.substrate.get_chain_head.return_value,
    )
    assert result == ExtrinsicResponse(
        False,
        f"Subnet {netuid} does not exist.",
        extrinsic_function="register_extrinsic",
    )


@pytest.mark.asyncio
async def test_register_extrinsic_already_registered(subtensor, fake_wallet, mocker):
    """Tests registration when the key is already registered."""
    # Preps
    netuid = 14
    mocked_get_neuron = mocker.patch.object(
        subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=mocker.Mock(is_null=False),
    )

    # Call
    success, message = await async_registration.register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
    )

    # Asserts
    mocked_get_neuron.assert_called_once_with(
        hotkey_ss58=fake_wallet.hotkey.ss58_address,
        netuid=netuid,
        block_hash=subtensor.substrate.get_chain_head.return_value,
    )
    assert success is True
    assert message == f"Already registered."


@pytest.mark.asyncio
async def test_register_extrinsic_max_attempts_reached(subtensor, fake_wallet, mocker):
    # Preps
    fake_wallet.hotkey.ss58_address = "hotkey_ss58"
    fake_wallet.coldkey.ss58_address = "coldkey_ss58"

    stale_responses = iter([False, False, False, True])

    async def is_stale_side_effect(*_, **__):
        return next(stale_responses, True)

    fake_pow_result = mocker.Mock()
    fake_pow_result.is_stale_async = mocker.AsyncMock(side_effect=is_stale_side_effect)
    fake_pow_result.seal = []

    mocked_subnet_exists = mocker.patch.object(
        subtensor, "subnet_exists", return_value=True
    )
    mocked_get_neuron = mocker.patch.object(
        subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=mocker.Mock(is_null=True),
    )
    mocked_create_pow = mocker.patch.object(
        async_registration,
        "create_pow_async",
        return_value=fake_pow_result,
    )
    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(False, "Test Error"),
    )

    # Call
    result = await async_registration.register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=1,
        max_allowed_attempts=3,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_subnet_exists.assert_called_once_with(
        1,
        block_hash=subtensor.substrate.get_chain_head.return_value,
    )
    mocked_get_neuron.assert_called_once_with(
        hotkey_ss58="hotkey_ss58",
        netuid=1,
        block_hash=subtensor.substrate.get_chain_head.return_value,
    )
    assert mocked_create_pow.call_count == 3
    assert mocked_sign_and_send_extrinsic.call_count == 3
    mocked_sign_and_send_extrinsic.assert_called_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )
    assert result[0] is False
    assert result[1] == "No more attempts."


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
            "hotkey": fake_wallet.hotkey.ss58_address,
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
            "hotkey": fake_wallet.hotkey.ss58_address,
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
