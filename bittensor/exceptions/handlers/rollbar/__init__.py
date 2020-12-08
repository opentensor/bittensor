import os
from loguru import logger
import asyncio
from bittensor.exceptions.handlers import asyncio_exception_handler
import rollbar

def init():
    token = os.environ.get("ROLLBAR_TOKEN", False)
    if not token:
        return

    env = os.environ.get("BT_ENV", "production")
    logger.info("Error reporting enabled using {}:{}", token, env)

    loop = asyncio.get_event_loop()
    loop.set_exception_handler(asyncio_exception_handler)


def is_enabled():
    return os.environ.get("ROLLBAR_TOKEN", False)

def run(func):
    try:
        func
    except BaseException as e:
        logger.debug("Sending exception to rollbar")
        rollbar.report_exc_info()
        raise e
