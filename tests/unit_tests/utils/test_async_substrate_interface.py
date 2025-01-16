import pytest
import asyncio
from async_substrate_interface import substrate_interface as async_substrate_interface
from typing import Any


@pytest.mark.asyncio
async def test_wait_for_block_invalid_result_handler():
    chain_interface = async_substrate_interface.AsyncSubstrateInterface(
        "dummy_endpoint"
    )

    with pytest.raises(ValueError):

        async def dummy_handler(
            block_data: dict[str, Any], extra_arg
        ):  # extra argument
            return block_data.get("header", {}).get("number", -1) == 2

        await chain_interface.wait_for_block(
            block=2, result_handler=dummy_handler, task_return=False
        )


@pytest.mark.asyncio
async def test_wait_for_block_async_return():
    chain_interface = async_substrate_interface.AsyncSubstrateInterface(
        "dummy_endpoint"
    )

    async def dummy_handler(block_data: dict[str, Any]) -> bool:
        return block_data.get("header", {}).get("number", -1) == 2

    result = await chain_interface.wait_for_block(
        block=2, result_handler=dummy_handler, task_return=True
    )

    assert isinstance(result, asyncio.Task)
