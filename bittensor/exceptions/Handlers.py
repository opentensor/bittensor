from loguru import logger
import rollbar
import os
import asyncio
from pathlib import Path

def asyncio_exception_handler(loop, context):
    logger.debug("asyncio exception has occured")

    exception: BaseException
    exception = context['exception']

    exc_info = get_exc_info(exception)
    extra_data = get_extra_data(context)

    rollbar.report_exc_info(exc_info=exc_info, extra_data=extra_data)
    logger.error(context)

    file = Path('/tmp/bt_runstate')
    with file.open("w") as file:
        file.write("ERR\n")


def get_exc_info(exception):
    _type = type(exception)
    value = exception
    tb = exception.__traceback__
    return _type, value, tb


def get_extra_data(context):
    frames = []
    for elem in context['source_traceback']:
        frames.append(str(elem))
    extra_data = "\n".join(frames)
    return extra_data


class Rollbar():
    def __init__(self):
        self.__token = os.environ.get("ROLLBAR_TOKEN", False)

    def is_enabled(self):
        return True if self.__token else False

    def run(self, func):
        env = os.environ.get("BT_ENV", "production")

        logger.info("Error reporting enabled using {}:{}", self.__token, env)
        rollbar.init(self.__token, env)

        loop = asyncio.get_event_loop()
        loop.set_exception_handler(asyncio_exception_handler)

        try:
            func()
        except Exception as e:
            rollbar.report_exc_info()
            raise e








