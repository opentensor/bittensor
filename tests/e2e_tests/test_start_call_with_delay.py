import pytest

from tests.e2e_tests.utils import (
    AdminUtils,
    TestSubnet,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
    SUDO_SET_START_CALL_DELAY,
)


def test_start_call_with_delay(subtensor, alice_wallet, eve_wallet):
    """Test for start call with delay.

    Steps:
    - Prepare root subnet and eve subnet
    - Check initial start call delay value
    - Set new start call delay via sudo call
    - Verify the new start call delay is applied
    - Register and activate eve subnet to verify it works with a new delay
    """
    # Preps
    sn_root = TestSubnet(subtensor, netuid=0)
    eve_sn = TestSubnet(subtensor)

    # Set a new start call delay
    new_start_call_delay = 20

    # Check the initial start call delay
    initial_start_call = subtensor.inner_subtensor.query_subtensor(
        name="StartCallDelay"
    )
    assert initial_start_call == subtensor.chain.get_start_call_delay()

    # Set a new start call delay via sudo call
    sn_root.execute_one(
        SUDO_SET_START_CALL_DELAY(alice_wallet, AdminUtils, True, new_start_call_delay)
    )

    # Check a new start call delay is set
    assert subtensor.chain.get_start_call_delay() == new_start_call_delay

    # Verify eve subnet can be activated with a new start call delay
    eve_sn.execute_steps(
        [
            REGISTER_SUBNET(eve_wallet),
            ACTIVATE_SUBNET(eve_wallet),
        ]
    )


@pytest.mark.asyncio
async def test_start_call_with_delay_async(async_subtensor, alice_wallet, eve_wallet):
    """Async test for start call with delay.

    Steps:
    - Prepare root subnet and eve subnet
    - Check initial start call delay value
    - Set new start call delay via sudo call
    - Verify the new start call delay is applied
    - Register and activate eve subnet to verify it works with a new delay
    """
    # Preps
    sn_root = TestSubnet(async_subtensor, netuid=0)
    eve_sn = TestSubnet(async_subtensor)

    # Set a new start call delay
    new_start_call_delay = 20

    # Check the initial start call delay
    initial_start_call = await async_subtensor.inner_subtensor.query_subtensor(
        name="StartCallDelay"
    )
    assert initial_start_call == await async_subtensor.chain.get_start_call_delay()

    # Set a new start call delay via sudo call
    await sn_root.async_execute_one(
        SUDO_SET_START_CALL_DELAY(alice_wallet, AdminUtils, True, new_start_call_delay)
    )

    # Check a new start call delay is set
    assert await async_subtensor.chain.get_start_call_delay() == new_start_call_delay

    # Verify eve subnet can be activated with a new start call delay
    await eve_sn.async_execute_steps(
        [
            REGISTER_SUBNET(eve_wallet),
            ACTIVATE_SUBNET(eve_wallet),
        ]
    )
