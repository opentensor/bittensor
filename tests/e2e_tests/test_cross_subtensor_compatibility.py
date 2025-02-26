from datetime import datetime
import pytest


@pytest.mark.asyncio
async def test_get_timestamp(subtensor, async_subtensor, local_chain):
    with subtensor:
        block_number = subtensor.get_current_block()
        assert isinstance(
            subtensor.get_timestamp(), datetime
        )  # verify it works with no block number specified
        sync_result = subtensor.get_timestamp(
            block=block_number
        )  # verify it works with block number specified
    async with async_subtensor:
        assert isinstance(await async_subtensor.get_timestamp(), datetime)
        async_result = await async_subtensor.get_timestamp(block=block_number)
    assert sync_result == async_result
