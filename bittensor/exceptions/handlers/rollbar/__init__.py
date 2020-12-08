import os
from loguru import logger
import asyncio
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
        logger.info("Sending exception to rollbar")
        rollbar.report_exc_info()

        set_runtime_status("ERR")

        raise


def set_runtime_status(status):
    file = Path('/tmp/bt_runstate')
    with file.open("w") as file:
        file.write("%s\n" % status)


def asyncio_exception_handler(loop, context):
    logger.debug("asyncio exception has occured")

    exception: BaseException
    exception = context['exception']

    exc_info = __get_exc_info(exception)
    extra_data = __get_extra_data(context)

    rollbar.report_exc_info(exc_info=exc_info, extra_data=extra_data)
    logger.error(context)

    set_runtime_status("ERR")

def __get_exc_info(exception):
    _type = type(exception)
    value = exception
    tb = exception.__traceback__
    return _type, value, tb

def __get_extra_data(context):
    frames = []
    for elem in context['source_traceback']:
        frames.append(str(elem))
    extra_data = "\n".join(frames)
    return extra_data

