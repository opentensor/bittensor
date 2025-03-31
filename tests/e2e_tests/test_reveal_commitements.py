import time

import pytest

from bittensor.utils.btlogging import logging


@pytest.mark.parametrize("local_chain", [False], indirect=True)
@pytest.mark.asyncio
async def test_set_reveal_commitment(local_chain, subtensor, alice_wallet, bob_wallet):
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
    BLOCK_TIME = 12  # 12 for non-fast-block, 0.25 for fast block
    BLOCKS_UNTIL_REVEAL = 15

    NETUID = 2

    logging.console.info("Testing Drand encrypted commitments.")

    # Register subnet as Alice
    assert subtensor.register_subnet(
        alice_wallet, True, True
    ), "Unable to register the subnet"

    # Register Bob's neuron
    assert subtensor.burned_register(
        bob_wallet, NETUID, True, True
    ), "Bob's neuron was not register."

    # Verify subnet 2 created successfully
    assert subtensor.subnet_exists(NETUID), "Subnet wasn't created successfully"

    # Set commitment from Alice hotkey
    message_alice = f"This is test message with time {time.time()} from Alice."

    response = subtensor.set_reveal_commitment(
        alice_wallet, NETUID, message_alice, BLOCKS_UNTIL_REVEAL, BLOCK_TIME
    )
    assert response[0] is True

    # Set commitment from Bob's hotkey
    message_bob = f"This is test message with time {time.time()} from Bob."

    response = subtensor.set_reveal_commitment(
        bob_wallet, NETUID, message_bob, BLOCKS_UNTIL_REVEAL, block_time=BLOCK_TIME
    )
    assert response[0] is True

    target_reveal_round = response[1]

    # Sometimes the chain doesn't update the repository right away and the commit doesn't appear in the expected
    # `last_drand_round`. In this case need to wait a bit.
    while subtensor.last_drand_round() < target_reveal_round + 1:
        # wait one drand period (3 sec)
        time.sleep(3)

    actual_all = subtensor.get_all_revealed_commitments(NETUID)

    alice_result = actual_all.get(alice_wallet.hotkey.ss58_address)
    assert alice_result is not None, "Alice's commitment was not received."

    bob_result = actual_all.get(bob_wallet.hotkey.ss58_address)
    assert bob_result is not None, "Bob's commitment was not received."

    alice_actual_block, alice_actual_message = alice_result
    bob_actual_block, bob_actual_message = bob_result

    # We do not check the release block because it is a dynamic number. It depends on the load of the chain, the number
    # of commits in the chain and the computing power.
    assert message_alice == alice_actual_message
    assert message_bob == bob_actual_message

    # Assertions for get_revealed_commitment (based of hotkey)
    actual_alice_block, actual_alice_message = subtensor.get_revealed_commitment(
        NETUID, alice_wallet.hotkey.ss58_address
    )
    actual_bob_block, actual_bob_message = subtensor.get_revealed_commitment(
        NETUID, bob_wallet.hotkey.ss58_address
    )

    assert message_alice == actual_alice_message
    assert message_bob == actual_bob_message
