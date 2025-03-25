import pytest

from bittensor import logging
from async_substrate_interface.errors import SubstrateRequestException

logging.set_trace()


@pytest.mark.asyncio
async def test_commitment(subtensor, alice_wallet):
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

    with pytest.raises(
        SubstrateRequestException,
        match="CommitmentSetRateLimitExceeded",
    ):
        await subtensor.set_commitment(
            alice_wallet,
            netuid=1,
            data="Hello World!",
        )

    assert "Hello World!" == await subtensor.get_commitment(
        netuid=1,
        uid=uid,
    )

    assert (await subtensor.get_all_commitments(netuid=1))[
        alice_wallet.hotkey.ss58_address
    ] == "Hello World!"
