from collections import namedtuple

import pytest
from async_substrate_interface.errors import SubstrateRequestException

from tests.e2e_tests.utils import (
    TestSubnet,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
    REGISTER_NEURON,
)

SET_MAX_SPACE = namedtuple("SET_MAX_SPACE", ["wallet", "pallet", "sudo", "new_limit"])
COMMITMENT_MESSAGE = "Hello World!"


def test_commitment(subtensor, alice_wallet, dave_wallet):
    """Tests commitment extrinsic."""
    # Create and prepare subnet
    dave_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(dave_wallet),
        ACTIVATE_SUBNET(dave_wallet),
    ]
    dave_sn.execute_steps(steps)

    with pytest.raises(SubstrateRequestException, match="AccountNotAllowedCommit"):
        subtensor.commitments.set_commitment(
            wallet=alice_wallet,
            netuid=dave_sn.netuid,
            data=COMMITMENT_MESSAGE,
            raise_error=True,
        )

    dave_sn.execute_steps([REGISTER_NEURON(alice_wallet)])

    uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
    )
    assert uid is not None
    assert "" == subtensor.commitments.get_commitment(
        netuid=dave_sn.netuid,
        uid=uid,
    )

    assert subtensor.commitments.set_commitment(
        wallet=alice_wallet,
        netuid=dave_sn.netuid,
        data=COMMITMENT_MESSAGE,
    )

    response = dave_sn.execute_one(
        SET_MAX_SPACE(alice_wallet, "Commitments", True, len(COMMITMENT_MESSAGE))
    )
    assert response.success, response.message

    with pytest.raises(
        SubstrateRequestException,
        match="SpaceLimitExceeded",
    ):
        subtensor.commitments.set_commitment(
            wallet=alice_wallet,
            netuid=dave_sn.netuid,
            data=COMMITMENT_MESSAGE + "longer",
            raise_error=True,
        )

    assert COMMITMENT_MESSAGE == subtensor.commitments.get_commitment(
        netuid=dave_sn.netuid,
        uid=uid,
    )

    assert (
        subtensor.commitments.get_all_commitments(netuid=dave_sn.netuid)[
            alice_wallet.hotkey.ss58_address
        ]
        == COMMITMENT_MESSAGE
    )


@pytest.mark.asyncio
async def test_commitment_async(async_subtensor, alice_wallet, dave_wallet):
    # Create and prepare subnet
    dave_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(dave_wallet),
        ACTIVATE_SUBNET(dave_wallet),
    ]
    await dave_sn.async_execute_steps(steps)

    with pytest.raises(SubstrateRequestException, match="AccountNotAllowedCommit"):
        await async_subtensor.commitments.set_commitment(
            wallet=alice_wallet,
            netuid=dave_sn.netuid,
            data=COMMITMENT_MESSAGE,
            raise_error=True,
        )

    await dave_sn.async_execute_steps([REGISTER_NEURON(alice_wallet)])

    uid = await async_subtensor.subnets.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
    )
    assert uid is not None
    assert "" == await async_subtensor.commitments.get_commitment(
        netuid=dave_sn.netuid,
        uid=uid,
    )

    assert await async_subtensor.commitments.set_commitment(
        wallet=alice_wallet,
        netuid=dave_sn.netuid,
        data=COMMITMENT_MESSAGE,
    )

    response = await dave_sn.async_execute_one(
        SET_MAX_SPACE(alice_wallet, "Commitments", True, len(COMMITMENT_MESSAGE))
    )
    assert response.success, response.message

    with pytest.raises(
        SubstrateRequestException,
        match="SpaceLimitExceeded",
    ):
        await async_subtensor.commitments.set_commitment(
            wallet=alice_wallet,
            netuid=dave_sn.netuid,
            data=COMMITMENT_MESSAGE + "longer",
            raise_error=True,
        )

    assert COMMITMENT_MESSAGE == await async_subtensor.commitments.get_commitment(
        netuid=dave_sn.netuid,
        uid=uid,
    )

    assert (
        await async_subtensor.commitments.get_all_commitments(netuid=dave_sn.netuid)
    )[alice_wallet.hotkey.ss58_address] == COMMITMENT_MESSAGE
