import os
from loguru import logger
import asyncio
from bittensor.exceptions.handlers import asyncio_exception_handler
import rollbar
from pathlib import Path

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
        set_runtime_status("OK")
        func
    except:
        logger.debug("Sending exception to rollbar")
        rollbar.report_exc_info()

        set_runtime_status("ERR")

        raise


def set_runtime_status(status):
    file = Path('/tmp/bt_runstate')
    with file.open("w") as file:
        file.write("%s\n" % status)