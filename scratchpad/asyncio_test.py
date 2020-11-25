import asyncio
import time
from loguru import logger
from  bittensor.utils.asyncio import Asyncio


def mother(bleh):

    logger.info("Entered blocking task")


    # future = asyncio.run_coroutine_threadsafe(do_gather(), Asyncio.loop)

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(do_gather(loop))
    loop.stop()

    logger.info(result)

    logger.info("DONE")





async def do_gather(loop):
    result = await asyncio.gather(
        blocking_task(loop, 1,1),
        blocking_task(loop, 1,2),
        blocking_task(loop, 4,3),
        blocking_task(loop, 10,4),
        blocking_task(loop, 5,5)

    )

    print(result)

    return result




async def blocking_task(loop, delay, retval):
    return await loop.run_in_executor(None, blocking_sync_task, delay, retval)



def blocking_sync_task(delay, retval=1):
    logger.info("Executing blocking sync task: {} {}", delay, retval)
    time.sleep(delay)
    return retval




async def task_1():
    logger.info("task_1")


async def task_2():
    logger.info("task_2")


async def task_from_thread():
    logger.info("Entered async task from thread")


if __name__ == '__main__':

    Asyncio.init()


    Asyncio.start_in_thread(mother, Asyncio.loop)
    Asyncio.add_task(task_1())
    Asyncio.add_task(task_2())


    Asyncio.run_forever()

