import re


def test_blocks(subtensor):
    """
    Tests:
    - Get current block
    - Get block hash
    - Wait for block
    """

    block = subtensor.get_current_block()

    assert block == subtensor.block

    block_hash = subtensor.get_block_hash(block)

    assert re.match("0x[a-z0-9]{64}", block_hash)

    subtensor.wait_for_block(block + 10)

    assert subtensor.get_current_block() == block + 10
