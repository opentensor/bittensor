from bittensor_wallet import Wallet
from scalecodec.types import GenericCall

from bittensor.core.extrinsics import proxy
from bittensor.core.settings import DEFAULT_MEV_PROTECTION
from bittensor.core.types import ExtrinsicResponse


def test_add_proxy_extrinsic(subtensor, mocker):
    """Verify that sync `add_proxy_extrinsic` method calls proper async method."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    delegate_ss58 = mocker.MagicMock(spec=str)
    proxy_type = mocker.MagicMock(spec=proxy.ProxyType)
    delay = mocker.MagicMock(spec=int)

    mocked_normalize = mocker.patch.object(proxy.ProxyType, "normalize")
    mocked_pallet_call = mocker.patch.object(proxy.Proxy, "add_proxy")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    response = proxy.add_proxy_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        delegate_ss58=delegate_ss58,
        proxy_type=proxy_type,
        delay=delay,
        mev_protection=DEFAULT_MEV_PROTECTION,
    )

    # Asserts
    mocked_normalize.assert_called_once_with(proxy_type)
    mocked_pallet_call.assert_called_once_with(
        delegate=delegate_ss58,
        proxy_type=mocked_normalize.return_value,
        delay=delay,
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_pallet_call.return_value,
        wallet=wallet,
        raise_error=False,
        period=None,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value


def test_remove_proxy_extrinsic(subtensor, mocker):
    """Verify that sync `remove_proxy_extrinsic` method calls proper methods."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    delegate_ss58 = mocker.MagicMock(spec=str)
    proxy_type = mocker.MagicMock(spec=proxy.ProxyType)
    delay = mocker.MagicMock(spec=int)

    mocked_normalize = mocker.patch.object(proxy.ProxyType, "normalize")
    mocked_pallet_call = mocker.patch.object(proxy.Proxy, "remove_proxy")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    response = proxy.remove_proxy_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        delegate_ss58=delegate_ss58,
        proxy_type=proxy_type,
        delay=delay,
        mev_protection=DEFAULT_MEV_PROTECTION,
    )

    # Asserts
    mocked_normalize.assert_called_once_with(proxy_type)
    mocked_pallet_call.assert_called_once_with(
        delegate=delegate_ss58,
        proxy_type=mocked_normalize.return_value,
        delay=delay,
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_pallet_call.return_value,
        wallet=wallet,
        raise_error=False,
        period=None,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value


def test_remove_proxies_extrinsic(subtensor, mocker):
    """Verify that sync `remove_proxies_extrinsic` method calls proper methods."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)

    mocked_pallet_call = mocker.patch.object(proxy.Proxy, "remove_proxies")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    response = proxy.remove_proxies_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        mev_protection=DEFAULT_MEV_PROTECTION,
    )

    # Asserts
    mocked_pallet_call.assert_called_once_with()
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_pallet_call.return_value,
        wallet=wallet,
        raise_error=False,
        period=None,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value


def test_create_pure_proxy_extrinsic(subtensor, mocker):
    """Verify that sync `create_pure_proxy_extrinsic` method calls proper methods."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    proxy_type = mocker.MagicMock(spec=proxy.ProxyType)
    delay = mocker.MagicMock(spec=int)
    index = mocker.MagicMock(spec=int)

    mocked_normalize = mocker.patch.object(proxy.ProxyType, "normalize")
    mocked_pallet_call = mocker.patch.object(proxy.Proxy, "create_pure")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Mock response with events
    mock_response = mocker.MagicMock(spec=ExtrinsicResponse)
    mock_response.success = True
    mock_response.extrinsic_receipt = mocker.MagicMock()
    mock_response.extrinsic_receipt.triggered_events = [
        {
            "event_id": "PureCreated",
            "attributes": {
                "pure": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                "who": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "proxy_type": "Any",
                "disambiguation_index": 0,
            },
        }
    ]
    mock_response.extrinsic_receipt.block_hash = "NOT NONE BLOCK HASH"
    mock_response.extrinsic_receipt.extrinsic_idx = 1
    mocked_sign_and_send_extrinsic.return_value = mock_response

    mocked_get_block_number = mocker.patch.object(
        subtensor.substrate, "get_block_number"
    )

    # Call
    response = proxy.create_pure_proxy_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        proxy_type=proxy_type,
        delay=delay,
        index=index,
        mev_protection=DEFAULT_MEV_PROTECTION,
    )

    # Asserts
    mocked_normalize.assert_called_once_with(proxy_type)
    mocked_pallet_call.assert_called_once_with(
        proxy_type=mocked_normalize.return_value,
        delay=delay,
        index=index,
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_pallet_call.return_value,
        wallet=wallet,
        raise_error=False,
        period=None,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    mocked_get_block_number.assert_called_once()
    assert response == mock_response
    assert (
        response.data["pure_account"]
        == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    )
    assert (
        response.data["spawner"] == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    )
    assert response.data["height"] == mocked_get_block_number.return_value
    assert response.data["ext_index"] == 1


def test_kill_pure_proxy_extrinsic(subtensor, mocker):
    """Verify that sync `kill_pure_proxy_extrinsic` method calls proper methods."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    wallet.coldkey.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    pure_proxy_ss58 = mocker.MagicMock(spec=str)
    spawner = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    proxy_type = mocker.MagicMock(spec=proxy.ProxyType)
    index = mocker.MagicMock(spec=int)
    height = mocker.MagicMock(spec=int)
    ext_index = mocker.MagicMock(spec=int)

    mocked_normalize = mocker.patch.object(proxy.ProxyType, "normalize")
    mocked_pallet_call = mocker.patch.object(proxy.Proxy, "kill_pure")
    mocked_proxy_extrinsic = mocker.patch.object(proxy, "proxy_extrinsic")

    # Call
    response = proxy.kill_pure_proxy_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        pure_proxy_ss58=pure_proxy_ss58,
        spawner=spawner,
        proxy_type=proxy_type,
        index=index,
        height=height,
        ext_index=ext_index,
        mev_protection=DEFAULT_MEV_PROTECTION,
    )

    # Asserts
    mocked_normalize.assert_called_once_with(proxy_type)
    mocked_pallet_call.assert_called_once_with(
        spawner=spawner,
        proxy_type=mocked_normalize.return_value,
        index=index,
        height=height,
        ext_index=ext_index,
    )
    mocked_proxy_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        real_account_ss58=pure_proxy_ss58,
        force_proxy_type=proxy.ProxyType.Any,
        call=mocked_pallet_call.return_value,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )
    assert response == mocked_proxy_extrinsic.return_value


def test_proxy_extrinsic(subtensor, mocker):
    """Verify that sync `proxy_extrinsic` method calls proper methods."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    real_account_ss58 = mocker.MagicMock(spec=str)
    force_proxy_type = mocker.MagicMock(spec=str)
    call = mocker.MagicMock(spec=GenericCall)

    mocked_normalize = mocker.patch.object(proxy.ProxyType, "normalize")
    mocked_pallet_call = mocker.patch.object(proxy.Proxy, "proxy")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    response = proxy.proxy_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        real_account_ss58=real_account_ss58,
        force_proxy_type=force_proxy_type,
        call=call,
        mev_protection=DEFAULT_MEV_PROTECTION,
    )

    # Asserts
    mocked_normalize.assert_called_once_with(force_proxy_type)
    mocked_pallet_call.assert_called_once_with(
        real=real_account_ss58,
        force_proxy_type=mocked_normalize.return_value,
        call=call,
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_pallet_call.return_value,
        wallet=wallet,
        raise_error=False,
        period=None,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value


def test_proxy_extrinsic_with_none_force_proxy_type(subtensor, mocker):
    """Verify that sync `proxy_extrinsic` method handles None force_proxy_type."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    real_account_ss58 = mocker.MagicMock(spec=str)
    force_proxy_type = None
    call = mocker.MagicMock(spec=GenericCall)

    mocked_pallet_call = mocker.patch.object(proxy.Proxy, "proxy")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    response = proxy.proxy_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        real_account_ss58=real_account_ss58,
        force_proxy_type=force_proxy_type,
        call=call,
        mev_protection=DEFAULT_MEV_PROTECTION,
    )

    # Asserts
    mocked_pallet_call.assert_called_once_with(
        real=real_account_ss58,
        force_proxy_type=None,
        call=call,
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_pallet_call.return_value,
        wallet=wallet,
        raise_error=False,
        period=None,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value


def test_proxy_announced_extrinsic(subtensor, mocker):
    """Verify that sync `proxy_announced_extrinsic` method calls proper methods."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    delegate_ss58 = mocker.MagicMock(spec=str)
    real_account_ss58 = mocker.MagicMock(spec=str)
    force_proxy_type = mocker.MagicMock(spec=str)
    call = mocker.MagicMock(spec=GenericCall)

    mocked_normalize = mocker.patch.object(proxy.ProxyType, "normalize")
    mocked_pallet_call = mocker.patch.object(proxy.Proxy, "proxy_announced")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    response = proxy.proxy_announced_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        delegate_ss58=delegate_ss58,
        real_account_ss58=real_account_ss58,
        force_proxy_type=force_proxy_type,
        call=call,
        mev_protection=DEFAULT_MEV_PROTECTION,
    )

    # Asserts
    mocked_normalize.assert_called_once_with(force_proxy_type)
    mocked_pallet_call.assert_called_once_with(
        delegate=delegate_ss58,
        real=real_account_ss58,
        force_proxy_type=mocked_normalize.return_value,
        call=call,
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_pallet_call.return_value,
        wallet=wallet,
        raise_error=False,
        period=None,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value


def test_proxy_announced_extrinsic_with_none_force_proxy_type(subtensor, mocker):
    """Verify that sync `proxy_announced_extrinsic` method handles None force_proxy_type."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    delegate_ss58 = mocker.MagicMock(spec=str)
    real_account_ss58 = mocker.MagicMock(spec=str)
    force_proxy_type = None
    call = mocker.MagicMock(spec=GenericCall)

    mocked_pallet_call = mocker.patch.object(proxy.Proxy, "proxy_announced")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    response = proxy.proxy_announced_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        delegate_ss58=delegate_ss58,
        real_account_ss58=real_account_ss58,
        force_proxy_type=force_proxy_type,
        call=call,
        mev_protection=DEFAULT_MEV_PROTECTION,
    )

    # Asserts
    mocked_pallet_call.assert_called_once_with(
        delegate=delegate_ss58,
        real=real_account_ss58,
        force_proxy_type=None,
        call=call,
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_pallet_call.return_value,
        wallet=wallet,
        raise_error=False,
        period=None,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value


def test_announce_extrinsic(subtensor, mocker):
    """Verify that sync `announce_extrinsic` method calls proper methods."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    real_account_ss58 = mocker.MagicMock(spec=str)
    call_hash = mocker.MagicMock(spec=str)

    mocked_pallet_call = mocker.patch.object(proxy.Proxy, "announce")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    response = proxy.announce_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        real_account_ss58=real_account_ss58,
        call_hash=call_hash,
        mev_protection=DEFAULT_MEV_PROTECTION,
    )

    # Asserts
    mocked_pallet_call.assert_called_once_with(
        real=real_account_ss58,
        call_hash=call_hash.lstrip().__radd__(),
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_pallet_call.return_value,
        wallet=wallet,
        raise_error=False,
        period=None,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value


def test_reject_announcement_extrinsic(subtensor, mocker):
    """Verify that sync `reject_announcement_extrinsic` method calls proper methods."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    delegate_ss58 = mocker.MagicMock(spec=str)
    call_hash = mocker.MagicMock(spec=str)

    mocked_pallet_call = mocker.patch.object(proxy.Proxy, "reject_announcement")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    response = proxy.reject_announcement_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        delegate_ss58=delegate_ss58,
        call_hash=call_hash,
        mev_protection=DEFAULT_MEV_PROTECTION,
    )

    # Asserts
    mocked_pallet_call.assert_called_once_with(
        delegate=delegate_ss58,
        call_hash=call_hash.lstrip().__radd__(),
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_pallet_call.return_value,
        wallet=wallet,
        raise_error=False,
        period=None,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value


def test_remove_announcement_extrinsic(subtensor, mocker):
    """Verify that sync `remove_announcement_extrinsic` method calls proper methods."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)
    real_account_ss58 = mocker.MagicMock(spec=str)
    call_hash = mocker.MagicMock(spec=str)

    mocked_pallet_call = mocker.patch.object(proxy.Proxy, "remove_announcement")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    response = proxy.remove_announcement_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        real_account_ss58=real_account_ss58,
        call_hash=call_hash,
        mev_protection=DEFAULT_MEV_PROTECTION,
    )

    # Asserts
    mocked_pallet_call.assert_called_once_with(
        real=real_account_ss58,
        call_hash=call_hash.lstrip().__radd__(),
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_pallet_call.return_value,
        wallet=wallet,
        raise_error=False,
        period=None,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value


def test_poke_deposit_extrinsic(subtensor, mocker):
    """Verify that sync `poke_deposit_extrinsic` method calls proper methods."""
    # Preps
    wallet = mocker.MagicMock(spec=Wallet)

    mocked_pallet_call = mocker.patch.object(proxy.Proxy, "poke_deposit")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    response = proxy.poke_deposit_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        mev_protection=DEFAULT_MEV_PROTECTION,
    )

    # Asserts
    mocked_pallet_call.assert_called_once_with()
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_pallet_call.return_value,
        wallet=wallet,
        raise_error=False,
        period=None,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_sign_and_send_extrinsic.return_value
