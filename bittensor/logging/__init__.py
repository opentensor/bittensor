import os
import sys
import time
import logging
import argparse
from typing import Optional, NamedTuple

import bittensor.config
from bittensor.logging.format import BtStreamFormatter
from bittensor.logging.helpers import all_loggers


BASE_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
BITTENSOR_LOGGER_NAME = "bittensor"


class LoggingConfig(NamedTuple):
    debug: bool
    trace: bool
    logging_dir: str
    record_log: bool


def __enable_logger_and_set_bt_format(
    logger: logging.Logger, 
    log_level: int=logging.INFO,
    logfile: str="bittensor.log"
):
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(BtStreamFormatter(BASE_LOG_FORMAT, DATE_FORMAT))
    handler.setLevel(log_level)
    handler.formatter.set_trace(True)

    logger.addHandler(handler)
    logger.setLevel(log_level)


def getLogger(name: str = BITTENSOR_LOGGER_NAME):
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(BtStreamFormatter(BASE_LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


def set_trace(on: bool=True):
    bt_logger = logging.getLogger(BITTENSOR_LOGGER_NAME)
    if on:
        for logger in all_loggers():
            __enable_logger_and_set_bt_format(logger, log_level=logging.TRACE)
    else:
        __set_default_logging(bt_logger)
        for logger in all_loggers():
            if logger.name is BITTENSOR_LOGGER_NAME:
                __set_default_logging(logger)
            __disable_logger(logger)


def set_debug(on: bool=True):
    if on:
        for logger in all_loggers():
            __enable_logger(level=logging.DEBUG)
    else:
        for logger in all_loggers():
            if logger.name is BITTENSOR_LOGGER_NAME:
                __set_default_logging(logger)
            __disable_logger(logger)


def __disable_logger(logger: logging.Logger):
    logger.setLevel(logging.CRITICAL)
    for handler in logger.handlers:
        if isinstance(handler.formatter, BtStreamFormatter):
            handler.formatter.set_trace(False)
        else:
            logger.removeHandler(handler)


def __enable_debug(logger: logging.Logger):
    pass


def __set_default_logging_global():
    for logger in all_loggers():
        if logger.name == BITTENSOR_LOGGER_NAME:
            __set_default_logging(logger)
        else:
            __disable_logger(logger)


def __disable_third_party_loggers():
    for logger in all_loggers():
        if logger.name == BITTENSOR_LOGGER_NAME:
            continue
        __disable_logger(logger)


def __set_default_logging(logger: logging.Logger):
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        if isinstance(handler.formatter, BtStreamFormatter):
            handler.setLevel(logging.INFO)
            handler.formatter.set_trace(False)
    

def on():
    logger = logging.getLogger(BITTENSOR_LOGGER_NAME)
    __enable_logger_and_set_bt_format(logger, logging.INFO)


def off():
    for logger in all_loggers():
        __disable_logger(logger)


def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None):
    """Accept specific arguments fro parser"""
    prefix_str = "" if prefix == None else prefix + "."
    try:
        default_logging_debug = os.getenv("BT_LOGGING_DEBUG") or False
        default_logging_trace = os.getenv("BT_LOGGING_TRACE") or False
        default_logging_record_log = os.getenv("BT_LOGGING_RECORD_LOG") or False
        default_logging_logging_dir = (
            os.getenv("BT_LOGGING_LOGGING_DIR") or "~/.bittensor/miners"
        )
        parser.add_argument(
            "--" + prefix_str + "logging.debug",
            action="store_true",
            help="""Turn on bittensor debugging information""",
            default=default_logging_debug,
        )
        parser.add_argument(
            "--" + prefix_str + "logging.trace",
            action="store_true",
            help="""Turn on bittensor trace level information""",
            default=default_logging_trace,
        )
        parser.add_argument(
            "--" + prefix_str + "logging.record_log",
            action="store_true",
            help="""Turns on logging to file.""",
            default=default_logging_record_log,
        )
        parser.add_argument(
            "--" + prefix_str + "logging.logging_dir",
            type=str,
            help="Logging default root directory.",
            default=default_logging_logging_dir,
        )
    except argparse.ArgumentError:
        # re-parsing arguments.
        pass


def config(cls):
    """Get config from the argument parser.

    Return:
        bittensor.config object
    """
    parser = argparse.ArgumentParser()
    add_args(parser)
    return bittensor.config(parser, args=[])



logger = getLogger(BITTENSOR_LOGGER_NAME)


# class btlogging:
#     @classmethod
#     def on(cls):
#         """Turn on logging output by re-adding the sinks."""
#         if cls.__off__:
#             cls.__off__ = False

#             cls.__std_sink__ = logging.StreamHandler(
#                 sys.stdout,
#                 colorize=True,
#                 enqueue=True,
#                 backtrace=True,
#                 diagnose=True,
#                 format=cls.log_formatter,                
#             )
#             cls.__std_sink__.setLevel(logging.NOTSET)
            
#             logger.addHandler(cls.__std_sink__)

#             # Check if logging to file was originally enabled and re-add the file sink
#             if cls.__file_sink__ is not None:
#                 config = cls.config()
#                 filepath = config.logging.logging_dir + "/logs.log"
#                 cls.__file_sink__ = logging.FileHandler(
#                     filepath,
#                     enqueue=True,
#                     backtrace=True,
#                     diagnose=True,
#                     format=cls.log_save_formatter,
#                     rotation="25 MB",
#                     retention="10 days",
#                 )
#                 logging.addHandler(cls.__file_sink__)

#             cls.set_debug(cls.__debug_on__)
#             cls.set_trace(cls.__trace_on__)



