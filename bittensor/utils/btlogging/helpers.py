# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
btlogging.helpers module provides helper functions for the Bittensor logging system.
"""

import logging
from typing import Generator


def all_loggers() -> Generator["logging.Logger", None, None]:
    """Generator that yields all logger instances in the application.

    Iterates through the logging root manager's logger dictionary and yields all active `Logger` instances. It skips
    placeholders and other types that are not instances of `Logger`.

    Yields:
        logger (logging.Logger): An active logger instance.
    """
    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.PlaceHolder):
            continue
        # In some versions of Python, the values in loggerDict might be
        # LoggerAdapter instances instead of Logger instances.
        # We check for Logger instances specifically.
        if isinstance(logger, logging.Logger):
            yield logger
        else:
            # If it's not a Logger instance, it could be a LoggerAdapter or
            # another form that doesn't directly offer logging methods.
            # This branch can be extended to handle such cases as needed.
            pass


def all_logger_names() -> Generator[str, None, None]:
    """
    Generate the names of all active loggers.

    This function iterates through the logging root manager's logger dictionary and yields the names of all active
    `Logger` instances. It skips placeholders and other types that are not instances of `Logger`.

    Yields:
        name (str): The name of an active logger.
    """
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.PlaceHolder):
            continue
        # In some versions of Python, the values in loggerDict might be
        # LoggerAdapter instances instead of Logger instances.
        # We check for Logger instances specifically.
        if isinstance(logger, logging.Logger):
            yield name
        else:
            # If it's not a Logger instance, it could be a LoggerAdapter or
            # another form that doesn't directly offer logging methods.
            # This branch can be extended to handle such cases as needed.
            pass


def get_max_logger_name_length() -> int:
    """
    Calculate and return the length of the longest logger name.

    This function iterates through all active logger names and determines the length of the longest name.

    Returns:
        max_length (int): The length of the longest logger name.
    """
    max_length = 0
    for name in all_logger_names():
        if len(name) > max_length:
            max_length = len(name)
    return max_length
