import pytest
from async_substrate_interface.errors import SubstrateRequestException

from bittensor import logging
from tests.e2e_tests.utils.chain_interactions import (
    async_sudo_set_admin_utils,
    sudo_set_admin_utils,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    wait_to_start_call,
    async_wait_to_start_call,
)

logging.set_trace()


def test_commitment(subtensor, alice_wallet, dave_wallet):
    dave_subnet_netuid = 2
    assert subtensor.subnets.register_subnet(dave_wallet)
    assert subtensor.subnets.subnet_exists(dave_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert wait_to_start_call(subtensor, dave_wallet, dave_subnet_netuid)

    with pytest.raises(SubstrateRequestException, match="AccountNotAllowedCommit"):
        subtensor.commitments.set_commitment(
            alice_wallet,
            netuid=dave_subnet_netuid,
            data="Hello World!",
        )

    assert subtensor.subnets.burned_register(
        alice_wallet,
        netuid=dave_subnet_netuid,
    )

    uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(
        alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    assert uid is not None

    assert "" == subtensor.commitments.get_commitment(
        netuid=dave_subnet_netuid,
        uid=uid,
    )

    assert subtensor.commitments.set_commitment(
        wallet=alice_wallet,
        netuid=dave_subnet_netuid,
        data="Hello World!",
    )

    status, error = sudo_set_admin_utils(
        substrate=subtensor.substrate,
        wallet=alice_wallet,
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
        subtensor.commitments.set_commitment(
            wallet=alice_wallet,
            netuid=dave_subnet_netuid,
            data="Hello World!1",
        )

    assert "Hello World!" == subtensor.commitments.get_commitment(
        netuid=dave_subnet_netuid,
        uid=uid,
    )

    assert (
        subtensor.commitments.get_all_commitments(netuid=dave_subnet_netuid)[
            alice_wallet.hotkey.ss58_address
        ]
        == "Hello World!"
    )


@pytest.mark.asyncio
async def test_commitment_async(async_subtensor, alice_wallet, dave_wallet):
    dave_subnet_netuid = 2
    assert await async_subtensor.subnets.register_subnet(dave_wallet)
    assert await async_subtensor.subnets.subnet_exists(dave_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert await async_wait_to_start_call(
        async_subtensor, dave_wallet, dave_subnet_netuid
    )

    async with async_subtensor as sub:
        with pytest.raises(SubstrateRequestException, match="AccountNotAllowedCommit"):
            await sub.commitments.set_commitment(
                alice_wallet,
                netuid=dave_subnet_netuid,
                data="Hello World!",
            )

        assert await sub.subnets.burned_register(
            alice_wallet,
            netuid=dave_subnet_netuid,
        )

        uid = await sub.subnets.get_uid_for_hotkey_on_subnet(
            alice_wallet.hotkey.ss58_address,
            netuid=dave_subnet_netuid,
        )

        assert uid is not None

        assert "" == await sub.commitments.get_commitment(
            netuid=dave_subnet_netuid,
            uid=uid,
        )

        assert await sub.commitments.set_commitment(
            alice_wallet,
            netuid=dave_subnet_netuid,
            data="Hello World!",
        )

        status, error = await async_sudo_set_admin_utils(
            substrate=async_subtensor.substrate,
            wallet=alice_wallet,
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
            await sub.commitments.set_commitment(
                alice_wallet,
                netuid=dave_subnet_netuid,
                data="Hello World!1",
            )

        assert "Hello World!" == await sub.commitments.get_commitment(
            netuid=dave_subnet_netuid,
            uid=uid,
        )

        assert (await sub.commitments.get_all_commitments(netuid=dave_subnet_netuid))[
            alice_wallet.hotkey.ss58_address
        ] == "Hello World!"
