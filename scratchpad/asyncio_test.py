import asyncio
import time
from loguru import logger
from  bittensor.utils.asyncio import Asyncio


def mother(bleh):

    logger.info("Entered blocking task")
    time.sleep(1)




    # future = asyncio.run_coroutine_threadsafe(do_gather(), Asyncio.loop)

    loop = asyncio.new_event_loop()
    loop.run_until_complete(do_gather())
    loop.stop()

    logger.info("DONE")





async def do_gather():
    result = await asyncio.gather(
        blocking_task(1,1),
        blocking_task(1,2),
        blocking_task(4,3),
        blocking_task(10,4),
        blocking_task(5,5)

    )

    print(result)



async def blocking_task(delay, retval):
    return await asyncio.sleep(delay, retval)


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

