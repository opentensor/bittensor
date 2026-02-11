import pytest

from bittensor.core.extrinsics.asyncex import staking
from bittensor.core.settings import DEFAULT_MEV_PROTECTION
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import Balance


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

    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")

    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(res_success, res_message),
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


@pytest.mark.asyncio
async def test_subnet_buyback_extrinsic(fake_wallet, mocker):
    """Verify that async `subnet_buyback_extrinsic` method calls proper methods."""
    # Preps
    fake_substrate = mocker.AsyncMock(**{"get_chain_head.return_value": "0xhead"})
    fake_subtensor = mocker.AsyncMock(
        **{
            "get_balance.return_value": Balance.from_tao(10),
            "get_existential_deposit.return_value": Balance.from_tao(1),
            "get_stake.return_value": Balance.from_tao(0),
            "get_block_hash.return_value": "0xblock",
            "sign_and_send_extrinsic.return_value": ExtrinsicResponse(True, "Success"),
            "substrate": fake_substrate,
        }
    )
    fake_wallet.coldkeypub.ss58_address = "coldkey"
    hotkey_ss58 = "hotkey"
    netuid = 1
    amount = Balance.from_tao(2)

    mocked_pallet_compose_call = mocker.AsyncMock()
    mocker.patch.object(
        staking.SubtensorModule, "add_stake_burn", new=mocked_pallet_compose_call
    )
    fake_subtensor.sim_swap = mocker.AsyncMock(
        return_value=mocker.Mock(tao_fee=Balance.from_rao(1), alpha_fee=mocker.Mock())
    )

    # Call
    result = await staking.add_stake_burn_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
        amount=amount,
        mev_protection=DEFAULT_MEV_PROTECTION,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )

    # Asserts
    assert result.success is True
    mocked_pallet_compose_call.assert_awaited_once_with(
        netuid=netuid,
        hotkey=hotkey_ss58,
        amount=amount.rao,
        limit=None,
    )
    fake_subtensor.sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_pallet_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        nonce_key="coldkeypub",
        use_nonce=True,
        period=None,
        raise_error=False,
    )


@pytest.mark.asyncio
async def test_subnet_buyback_extrinsic_with_limit(fake_wallet, mocker):
    """Verify that async `subnet_buyback_extrinsic` passes limit price."""
    # Preps
    fake_substrate = mocker.AsyncMock(**{"get_chain_head.return_value": "0xhead"})
    fake_subtensor = mocker.AsyncMock(
        **{
            "get_balance.return_value": Balance.from_tao(10),
            "get_existential_deposit.return_value": Balance.from_tao(1),
            "get_stake.return_value": Balance.from_tao(0),
            "get_block_hash.return_value": "0xblock",
            "sign_and_send_extrinsic.return_value": ExtrinsicResponse(True, "Success"),
            "substrate": fake_substrate,
        }
    )
    fake_wallet.coldkeypub.ss58_address = "coldkey"
    hotkey_ss58 = "hotkey"
    netuid = 1
    amount = Balance.from_tao(2)
    limit_price = Balance.from_tao(2)

    mocked_pallet_compose_call = mocker.AsyncMock()
    mocker.patch.object(
        staking.SubtensorModule, "add_stake_burn", new=mocked_pallet_compose_call
    )
    fake_subtensor.sim_swap = mocker.AsyncMock(
        return_value=mocker.Mock(tao_fee=Balance.from_rao(1), alpha_fee=mocker.Mock())
    )

    # Call
    result = await staking.add_stake_burn_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
        amount=amount,
        limit_price=limit_price,
        mev_protection=DEFAULT_MEV_PROTECTION,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )

    # Asserts
    assert result.success is True
    mocked_pallet_compose_call.assert_awaited_once_with(
        netuid=netuid,
        hotkey=hotkey_ss58,
        amount=amount.rao,
        limit=limit_price.rao,
    )
