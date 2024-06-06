"""Async test module."""

import asyncio

from bittensor.asyncbt import subtensor_module


async def main():
    subtensor = subtensor_module.async_subtensor()
    await subtensor.get_delegates()


if __name__ == "__main__":
    asyncio.run(main())
