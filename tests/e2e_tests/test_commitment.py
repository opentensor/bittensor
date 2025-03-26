import pytest
from async_substrate_interface.errors import SubstrateRequestException

from bittensor import logging
from tests.e2e_tests.utils.chain_interactions import sudo_set_admin_utils

logging.set_trace()


def test_commitment(local_chain, subtensor, alice_wallet):
    with pytest.raises(SubstrateRequestException, match="AccountNotAllowedCommit"):
        subtensor.set_commitment(
            alice_wallet,
            netuid=1,
            data="Hello World!",
        )

    assert subtensor.burned_register(
        alice_wallet,
        netuid=1,
    )

    uid = subtensor.get_uid_for_hotkey_on_subnet(
        alice_wallet.hotkey.ss58_address,
        netuid=1,
    )

    assert uid is not None

    assert "" == subtensor.get_commitment(
        netuid=1,
        uid=uid,
    )

    assert subtensor.set_commitment(
        alice_wallet,
        netuid=1,
        data="Hello World!",
    )

    status, error = sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_module="Commitments",
        call_function="set_max_space",
        call_params={
            "netuid": 1,
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
            netuid=1,
            data="Hello World!1",
        )

    assert "Hello World!" == subtensor.get_commitment(
        netuid=1,
        uid=uid,
    )

    assert (
        subtensor.get_all_commitments(netuid=1)[alice_wallet.hotkey.ss58_address]
        == "Hello World!"
    )


@pytest.mark.asyncio
async def test_commitment_async(local_chain, async_subtensor, alice_wallet):
    async with async_subtensor as sub:
        with pytest.raises(SubstrateRequestException, match="AccountNotAllowedCommit"):
            await sub.set_commitment(
                alice_wallet,
                netuid=1,
                data="Hello World!",
            )

        assert await sub.burned_register(
            alice_wallet,
            netuid=1,
        )

        uid = await sub.get_uid_for_hotkey_on_subnet(
            alice_wallet.hotkey.ss58_address,
            netuid=1,
        )

        assert uid is not None

        assert "" == await sub.get_commitment(
            netuid=1,
            uid=uid,
        )

        assert await sub.set_commitment(
            alice_wallet,
            netuid=1,
            data="Hello World!",
        )

        status, error = sudo_set_admin_utils(
            local_chain,
            alice_wallet,
            call_module="Commitments",
            call_function="set_max_space",
            call_params={
                "netuid": 1,
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
                netuid=1,
                data="Hello World!1",
            )

        assert "Hello World!" == await sub.get_commitment(
            netuid=1,
            uid=uid,
        )

        assert (await sub.get_all_commitments(netuid=1))[
            alice_wallet.hotkey.ss58_address
        ] == "Hello World!"
