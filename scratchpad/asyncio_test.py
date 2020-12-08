import asyncio
from loguru import logger


class Test:
    def __enter__(self):
        def handler(loop, context):
            logger.error("Exception")

        loop = asyncio.get_event_loop()
        # loop.set_exception_handler(handler)
        loop.run_until_complete(task_until_complete())

    def __exit__(self, exc_type, exc_value, exc_traceback):
        logger.info("RTEST")


def handler(loop, context):
    logger.error("Exception")


async def error_task():
    raise Exception

async def task_until_complete():
    loop = asyncio.get_event_loop()
    loop.create_task(error_task())


with Test():
    print("TEST")






