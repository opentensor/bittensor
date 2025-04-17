import pytest
from async_substrate_interface.errors import SubstrateRequestException

from bittensor import logging
from tests.e2e_tests.utils.chain_interactions import sudo_set_admin_utils
from tests.e2e_tests.utils.e2e_test_utils import wait_to_start_call

logging.set_trace()


def test_commitment(local_chain, subtensor, alice_wallet, dave_wallet):
    dave_subnet_netuid = 2
    assert subtensor.register_subnet(dave_wallet, True, True)
    assert subtensor.subnet_exists(dave_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert wait_to_start_call(subtensor, dave_wallet, dave_subnet_netuid, 10)

    with pytest.raises(SubstrateRequestException, match="AccountNotAllowedCommit"):
        subtensor.set_commitment(
            alice_wallet,
            netuid=dave_subnet_netuid,
            data="Hello World!",
        )

    assert subtensor.burned_register(
        alice_wallet,
        netuid=dave_subnet_netuid,
    )

    uid = subtensor.get_uid_for_hotkey_on_subnet(
        alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    assert uid is not None

    assert "" == subtensor.get_commitment(
        netuid=dave_subnet_netuid,
        uid=uid,
    )

    assert subtensor.set_commitment(
        alice_wallet,
        netuid=dave_subnet_netuid,
        data="Hello World!",
    )

    status, error = sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_module="Commitments",
        call_function="set_max_space",
        call_params={
            "netuid": dave_subnet_netuid,
            "new_limit": len("Hello World!"),
        },
    )

    assert status is True, error

    with pytest.raises(
        SubstrateRequestException,
        match="SpaceLimitExceeded",
    ):
        subtensor.set_commitment(
            alice_wallet,
            netuid=dave_subnet_netuid,
            data="Hello World!1",
        )

    assert "Hello World!" == subtensor.get_commitment(
        netuid=dave_subnet_netuid,
        uid=uid,
    )

    assert (
        subtensor.get_all_commitments(netuid=dave_subnet_netuid)[
            alice_wallet.hotkey.ss58_address
        ]
        == "Hello World!"
    )


@pytest.mark.asyncio
async def test_commitment_async(
    local_chain, async_subtensor, alice_wallet, dave_wallet
):
    dave_subnet_netuid = 2
    assert await async_subtensor.register_subnet(dave_wallet)
    assert await async_subtensor.subnet_exists(dave_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    await async_subtensor.wait_for_block(await async_subtensor.block + 20)
    status, message = await async_subtensor.start_call(
        dave_wallet, dave_subnet_netuid, True, True
    )
    assert status, message

    async with async_subtensor as sub:
        with pytest.raises(SubstrateRequestException, match="AccountNotAllowedCommit"):
            await sub.set_commitment(
                alice_wallet,
                netuid=dave_subnet_netuid,
                data="Hello World!",
            )

        assert await sub.burned_register(
            alice_wallet,
            netuid=dave_subnet_netuid,
        )

        uid = await sub.get_uid_for_hotkey_on_subnet(
            alice_wallet.hotkey.ss58_address,
            netuid=dave_subnet_netuid,
        )

        assert uid is not None

        assert "" == await sub.get_commitment(
            netuid=dave_subnet_netuid,
            uid=uid,
        )

        assert await sub.set_commitment(
            alice_wallet,
            netuid=dave_subnet_netuid,
            data="Hello World!",
        )

        status, error = sudo_set_admin_utils(
            local_chain,
            alice_wallet,
            call_module="Commitments",
            call_function="set_max_space",
            call_params={
                "netuid": dave_subnet_netuid,
                "new_limit": len("Hello World!"),
            },
        )

        assert status is True, error

        with pytest.raises(
            SubstrateRequestException,
            match="SpaceLimitExceeded",
        ):
            await sub.set_commitment(
                alice_wallet,
                netuid=dave_subnet_netuid,
                data="Hello World!1",
            )

        assert "Hello World!" == await sub.get_commitment(
            netuid=dave_subnet_netuid,
            uid=uid,
        )

        assert (await sub.get_all_commitments(netuid=dave_subnet_netuid))[
            alice_wallet.hotkey.ss58_address
        ] == "Hello World!"
