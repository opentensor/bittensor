from loguru import logger
import rollbar
from bittensor.exceptions.handlers.rollbar import set_runtime_status

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

