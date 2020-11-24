import asyncio
from loguru import logger
import time
import concurrent

async def task1():
    logger.info("Task 1")
    await asyncio.sleep(1)
    await task1()


async def task2():
    logger.info("Task 2")
    await asyncio.sleep(2)
    await task2()


def blocking_task():
    logger.info("This task is blocking")
    time.sleep(10)
    logger.info("Blocking task complete")


loop = asyncio.get_event_loop()
loop.create_task(task1())
loop.create_task(task2())
loop.create_task(blocking_task())


quit()

# Create a limited thread pool.
executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=3,
)

loop.run_in_executor(executor, blocking_task)
loop.run_forever()
