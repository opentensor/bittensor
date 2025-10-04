import time

import pytest

from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils import (
    TestSubnet,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
    REGISTER_NEURON,
)


def test_set_reveal_commitment(subtensor, alice_wallet, bob_wallet):
    """
    Tests the set/reveal commitments with TLE (time-locked encrypted commitments) mechanism.

    Steps:
        1. Register a subnet through Alice
        2. Register Bob's neuron and add stake
        3. Set commitment from Alice hotkey
        4. Set commitment from Bob hotkey
        5. Wait until commitment is revealed.
        5. Verify commitment is revealed by Alice and Bob and available via mutual call.
        6. Verify commitment is revealed by Alice and Bob and available via separate calls.
    Raises:
        AssertionError: If any of the checks or verifications fail

    Note: Actually we can run this tests in fast block mode. For this we need to set `BLOCK_TIME` to 0.25 and replace
    `False` to `True` in `pytest.mark.parametrize` decorator.
    """
    BLOCK_TIME, BLOCKS_UNTIL_REVEAL = (
        (0.25, 10) if subtensor.chain.is_fast_blocks() else (12.0, 5)
    )

    alice_sn = TestSubnet(subtensor)
    alice_sn.execute_steps(
        [
            REGISTER_SUBNET(alice_wallet),
            ACTIVATE_SUBNET(alice_wallet),
            REGISTER_NEURON(bob_wallet),
        ]
    )

    # Set commitment from Alice hotkey
    message_alice = f"This is test message with time {time.time()} from Alice."

    response = subtensor.commitments.set_reveal_commitment(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        data=message_alice,
        blocks_until_reveal=BLOCKS_UNTIL_REVEAL,
        block_time=BLOCK_TIME,
    )
    assert response.success, response.message

    # Set commitment from Bob's hotkey
    message_bob = f"This is test message with time {time.time()} from Bob."

    response = subtensor.commitments.set_reveal_commitment(
        wallet=bob_wallet,
        netuid=alice_sn.netuid,
        data=message_bob,
        blocks_until_reveal=BLOCKS_UNTIL_REVEAL,
        block_time=BLOCK_TIME,
    )
    assert response.success, response.message

    target_reveal_round = response.data.get("reveal_round")

    # Sometimes the chain doesn't update the repository right away and the commit doesn't appear in the expected
    # `last_drand_round`. In this case need to wait a bit.
    logging.console.info(f"Waiting for reveal round {target_reveal_round}")
    chain_offset = 1 if subtensor.chain.is_fast_blocks() else 24

    last_drand_round = -1
    while last_drand_round <= target_reveal_round + chain_offset:
        # wait one drand period (3 sec)
        last_drand_round = subtensor.chain.last_drand_round()
        logging.console.info(f"Current last reveled drand round {last_drand_round}")
        time.sleep(3)

    actual_all = subtensor.commitments.get_all_revealed_commitments(alice_sn.netuid)

    alice_result = actual_all.get(alice_wallet.hotkey.ss58_address)
    assert alice_result is not None, "Alice's commitment was not received."

    bob_result = actual_all.get(bob_wallet.hotkey.ss58_address)
    assert bob_result is not None, "Bob's commitment was not received."

    alice_actual_block, alice_actual_message = alice_result[0]
    bob_actual_block, bob_actual_message = bob_result[0]

    # We do not check the release block because it is a dynamic number. It depends on the load of the chain, the number
    # of commits in the chain and the computing power.
    assert message_alice == alice_actual_message
    assert message_bob == bob_actual_message

    # Assertions for get_revealed_commitment (based of hotkey)
    actual_alice_block, actual_alice_message = (
        subtensor.commitments.get_revealed_commitment(alice_sn.netuid, 0)[0]
    )
    actual_bob_block, actual_bob_message = (
        subtensor.commitments.get_revealed_commitment(alice_sn.netuid, 1)[0]
    )

    assert message_alice == actual_alice_message
    assert message_bob == actual_bob_message


@pytest.mark.asyncio
async def test_set_reveal_commitment_async(async_subtensor, alice_wallet, bob_wallet):
    """
    Tests the set/reveal commitments with TLE (time-locked encrypted commitments) mechanism.

    Steps:
        1. Register a subnet through Alice
        2. Register Bob's neuron and add stake
        3. Set commitment from Alice hotkey
        4. Set commitment from Bob hotkey
        5. Wait until commitment is revealed.
        5. Verify commitment is revealed by Alice and Bob and available via mutual call.
        6. Verify commitment is revealed by Alice and Bob and available via separate calls.
    Raises:
        AssertionError: If any of the checks or verifications fail

    Note: Actually we can run this tests in fast block mode. For this we need to set `BLOCK_TIME` to 0.25 and replace
    `False` to `True` in `pytest.mark.parametrize` decorator.
    """
    BLOCK_TIME, BLOCKS_UNTIL_REVEAL = (
        (0.25, 10) if await async_subtensor.chain.is_fast_blocks() else (12.0, 5)
    )

    alice_sn = TestSubnet(async_subtensor)
    await alice_sn.async_execute_steps(
        [
            REGISTER_SUBNET(alice_wallet),
            ACTIVATE_SUBNET(alice_wallet),
            REGISTER_NEURON(bob_wallet),
        ]
    )

    # Set commitment from Alice hotkey
    message_alice = f"This is test message with time {time.time()} from Alice."

    response = await async_subtensor.commitments.set_reveal_commitment(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        data=message_alice,
        blocks_until_reveal=BLOCKS_UNTIL_REVEAL,
        block_time=BLOCK_TIME,
    )
    assert response.success, response.message

    # Set commitment from Bob's hotkey
    message_bob = f"This is test message with time {time.time()} from Bob."

    response = await async_subtensor.commitments.set_reveal_commitment(
        wallet=bob_wallet,
        netuid=alice_sn.netuid,
        data=message_bob,
        blocks_until_reveal=BLOCKS_UNTIL_REVEAL,
        block_time=BLOCK_TIME,
    )
    assert response.success, response.message

    target_reveal_round = response.data.get("reveal_round")

    # Sometimes the chain doesn't update the repository right away and the commit doesn't appear in the expected
    # `last_drand_round`. In this case need to wait a bit.
    logging.console.info(f"Waiting for reveal round {target_reveal_round}")
    chain_offset = 1 if await async_subtensor.chain.is_fast_blocks() else 24

    last_drand_round = -1
    while last_drand_round <= target_reveal_round + chain_offset:
        # wait one drand period (3 sec)
        last_drand_round = await async_subtensor.chain.last_drand_round()
        logging.console.info(f"Current last reveled drand round {last_drand_round}")
        time.sleep(3)

    actual_all = await async_subtensor.commitments.get_all_revealed_commitments(
        alice_sn.netuid
    )

    alice_result = actual_all.get(alice_wallet.hotkey.ss58_address)
    assert alice_result is not None, "Alice's commitment was not received."

    bob_result = actual_all.get(bob_wallet.hotkey.ss58_address)
    assert bob_result is not None, "Bob's commitment was not received."

    alice_actual_block, alice_actual_message = alice_result[0]
    bob_actual_block, bob_actual_message = bob_result[0]

    # We do not check the release block because it is a dynamic number. It depends on the load of the chain, the number
    # of commits in the chain and the computing power.
    assert message_alice == alice_actual_message
    assert message_bob == bob_actual_message

    # Assertions for get_revealed_commitment (based of hotkey)
    actual_alice_block, actual_alice_message = (
        await async_subtensor.commitments.get_revealed_commitment(alice_sn.netuid, 0)
    )[0]
    actual_bob_block, actual_bob_message = (
        await async_subtensor.commitments.get_revealed_commitment(alice_sn.netuid, 1)
    )[0]

    assert message_alice == actual_alice_message
    assert message_bob == actual_bob_message
