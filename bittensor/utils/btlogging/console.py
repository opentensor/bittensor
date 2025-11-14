"""
BittensorConsole class gives the ability to log messages to the terminal without changing Bittensor logging level.

Example:
    from bittensor import logging

    # will be logged
    logging.console.info("info message")
    logging.console.error("error message")
    logging.console.success("success message")
    logging.console.warning("warning message")
    logging.console.critical("critical message")

    # will not be logged
    logging.info("test info")
"""

from functools import wraps
from typing import TYPE_CHECKING, Callable

from .helpers import all_loggers

if TYPE_CHECKING:
    from .loggingmachine import LoggingMachine


def _print_wrapper(func: "Callable"):
    @wraps(func)
    def wrapper(self: "BittensorConsole", *args, **kwargs):
        """A wrapper function to temporarily set the logger level to debug."""
        old_logger_level = self.logger.get_level()
        self.logger.set_console()
        func(self, *args, **kwargs)

        for logger in all_loggers():
            logger.setLevel(old_logger_level)

    return wrapper


class BittensorConsole:
    def __init__(self, logger: "LoggingMachine"):
        self.logger = logger

    @_print_wrapper
    def debug(self, message: str):
        """Logs a DEBUG message to the console."""
        self.logger.debug(message, stacklevel=3)

    @_print_wrapper
    def info(self, message: str):
        """Logs a INFO message to the console."""
        self.logger.info(message, stacklevel=3)

    @_print_wrapper
    def success(self, message: str):
        """Logs a SUCCESS message to the console."""
        self.logger.success(message, stacklevel=3)

    @_print_wrapper
    def warning(self, message: str):
        """Logs a WARNING message to the console."""
        self.logger.warning(message, stacklevel=3)

    @_print_wrapper
    def error(self, message: str):
        """Logs a ERROR message to the console."""
        self.logger.error(message, stacklevel=3)

    @_print_wrapper
    def critical(self, message: str):
        """Logs a CRITICAL message to the console."""
        self.logger.critical(message, stacklevel=3)
