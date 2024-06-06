"""Async test module."""

import asyncio
import time

from bittensor.asyncbt import subtensor_module


subtensor = subtensor_module.async_subtensor()


async def get_delegates():
    return await subtensor.get_delegates()


async def get_all_subnets_info():
    return await subtensor.get_all_subnets_info()


async def get_block_hash():
    return await subtensor.get_block_hash(1000)


async def get_current_block():
    return await subtensor.get_current_block()


async def run():
    tasks = [
        get_delegates(),
        get_all_subnets_info(),
        get_block_hash(),
        get_current_block()
    ]
    return await asyncio.gather(*tasks)


if __name__ == "__main__":
    now = time.time()
    res = asyncio.run(run())
    print(res)
    print('>>> time', time.time() - now)
