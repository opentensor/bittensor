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
