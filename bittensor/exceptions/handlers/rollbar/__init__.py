import os
from loguru import logger
import asyncio
from bittensor.exceptions.handlers import asyncio_exception_handler
import rollbar

token = None

def init():
    token = os.environ.get("ROLLBAR_TOKEN", False)
    if not token:
        return

    env = os.environ.get("BT_ENV", "production")
    logger.info("Error reporting enabled using {}:{}", token, env)

    loop = asyncio.get_event_loop()
    loop.set_exception_handler(asyncio_exception_handler)


def is_enabled():
    return True if token else False

def run(func):
    try:
        func
    except Exception as e:
        rollbar.report_exc_info()
        raise e
