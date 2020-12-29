import os
from loguru import logger
import asyncio
import rollbar
from pathlib import Path
from loguru._handler import Message


class RollbarHandler():
    def __init__(self):
        self.token = os.environ.get("ROLLBAR_TOKEN", False)

    def write(self, message):
        if not self.token:
            return

        record = message.record
        level = record['level'].name

        if level == "WARNING":
            rollbar.report_message(message, 'warning')
        elif level == "ERROR":
            rollbar.report_message(message, 'error')
        else:
            pass




def init():
    token = os.environ.get("ROLLBAR_TOKEN", False)
    if not token:
        return

    env = os.environ.get("BT_ENV", "production")
    logger.info("Error reporting enabled using {}:{}", token, env)
    rollbar.init(token, env)
    set_runtime_status("OK")
    
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(asyncio_exception_handler)

    logger.add(sink=RollbarHandler(), level='WARNING')


def is_enabled():
    return os.environ.get("ROLLBAR_TOKEN", False)

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

    send_exception(exc_info, extra_data)
    logger.error(context)

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

def send_exception(info = None, extra_data = None):
    if not is_enabled():
        return

    logger.info("Sending exception to rollbar")
    rollbar.report_exc_info(exc_info=info, extra_data=extra_data)
    set_runtime_status("ERR")

