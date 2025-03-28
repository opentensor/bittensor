import pytest
from async_substrate_interface.errors import SubstrateRequestException

from bittensor import logging
from tests.e2e_tests.utils.chain_interactions import sudo_set_admin_utils

logging.set_trace()


@pytest.mark.asyncio
async def test_commitment(local_chain, subtensor, alice_wallet):
    with pytest.raises(SubstrateRequestException, match="AccountNotAllowedCommit"):
        await subtensor.set_commitment(
            alice_wallet,
            netuid=1,
            data="Hello World!",
        )

    assert await subtensor.burned_register(
        alice_wallet,
        netuid=1,
    )

    uid = await subtensor.get_uid_for_hotkey_on_subnet(
        alice_wallet.hotkey.ss58_address,
        netuid=1,
    )

    assert uid is not None

    assert "" == await subtensor.get_commitment(
        netuid=1,
        uid=uid,
    )

    assert await subtensor.set_commitment(
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
        await subtensor.set_commitment(
            alice_wallet,
            netuid=1,
            data="Hello World!1",
        )

    assert "Hello World!" == await subtensor.get_commitment(
        netuid=1,
        uid=uid,
    )

    assert (await subtensor.get_all_commitments(netuid=1))[
        alice_wallet.hotkey.ss58_address
    ] == "Hello World!"
