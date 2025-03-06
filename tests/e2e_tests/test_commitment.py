import pytest

from bittensor import logging
from async_substrate_interface.errors import SubstrateRequestException

logging.set_trace()


def test_commitment(subtensor, alice_wallet):
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

    with pytest.raises(
        SubstrateRequestException,
        match="CommitmentSetRateLimitExceeded",
    ):
        subtensor.set_commitment(
            alice_wallet,
            netuid=1,
            data="Hello World!",
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
async def test_commitment_async(async_subtensor, alice_wallet):
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

        with pytest.raises(
            SubstrateRequestException,
            match="CommitmentSetRateLimitExceeded",
        ):
            await sub.set_commitment(
                alice_wallet,
                netuid=1,
                data="Hello World!",
            )

        assert "Hello World!" == await sub.get_commitment(
            netuid=1,
            uid=uid,
        )

        assert (await sub.get_all_commitments(netuid=1))[
            alice_wallet.hotkey.ss58_address
        ] == "Hello World!"
