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
Module provides a logging framework for Bittensor, managing both Bittensor-specific and third-party logging states.
It leverages the StateMachine from the statemachine package to transition between different logging states such as
Default, Debug, Trace, and Disabled.
"""

import argparse
import atexit
import copy
import logging as stdlogging
import multiprocessing as mp
import os
import sys
from logging import Logger
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from typing import NamedTuple

from statemachine import State, StateMachine

from bittensor.core.config import Config
from .defines import (
    BITTENSOR_LOGGER_NAME,
    DATE_FORMAT,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_FILE_NAME,
    DEFAULT_MAX_ROTATING_LOG_FILE_SIZE,
    TRACE_LOG_FORMAT,
)
from .format import BtFileFormatter, BtStreamFormatter
from .helpers import all_loggers


def _concat_message(msg="", prefix="", suffix=""):
    """Concatenates a message with optional prefix and suffix."""
    msg = f"{f'{prefix} - ' if prefix else ''}{msg}{f' - {suffix}' if suffix else ''}"
    return msg


class LoggingConfig(NamedTuple):
    """Named tuple to hold the logging configuration."""

    debug: bool
    trace: bool
    record_log: bool
    logging_dir: str


class LoggingMachine(StateMachine, Logger):
    """Handles logger states for bittensor and 3rd party libraries."""

    Default = State(initial=True)
    Debug = State()
    Trace = State()
    Disabled = State()
    Warning = State()

    enable_default = (
        Debug.to(Default)
        | Trace.to(Default)
        | Disabled.to(Default)
        | Default.to(Default)
        | Warning.to(Default)
    )

    enable_info = enable_default

    enable_trace = (
        Default.to(Trace)
        | Debug.to(Trace)
        | Disabled.to(Trace)
        | Trace.to(Trace)
        | Warning.to(Trace)
    )

    enable_debug = (
        Default.to(Debug)
        | Trace.to(Debug)
        | Disabled.to(Debug)
        | Debug.to(Debug)
        | Warning.to(Debug)
    )

    enable_warning = (
        Default.to(Warning)
        | Trace.to(Warning)
        | Disabled.to(Warning)
        | Debug.to(Warning)
        | Warning.to(Warning)
    )

    disable_trace = Trace.to(Default)

    disable_debug = Debug.to(Default)

    disable_warning = Warning.to(Default)

    disable_logging = (
        Trace.to(Disabled)
        | Debug.to(Disabled)
        | Default.to(Disabled)
        | Disabled.to(Disabled)
    )

    def __init__(self, config: "Config", name: str = BITTENSOR_LOGGER_NAME):
        # basics
        super(LoggingMachine, self).__init__()
        self._queue = mp.Queue(-1)
        self._primary_loggers = {name}
        self._config = self._extract_logging_config(config)

        # Formatters
        #
        # In the future, this may be expanded to a dictionary mapping handler
        # types to their respective formatters.
        self._stream_formatter = BtStreamFormatter()
        self._file_formatter = BtFileFormatter(TRACE_LOG_FORMAT, DATE_FORMAT)

        # start with handlers for the QueueListener.
        #
        # In the future, we may want to add options to introduce other handlers
        # for things like log aggregation by external services.
        self._handlers = self._configure_handlers(self._config)

        # configure and start the queue listener
        self._listener = self._create_and_start_listener(self._handlers)

        # set up all the loggers
        self._logger = self._initialize_bt_logger(name)
        self.disable_third_party_loggers()
        self._enable_initial_state(self._config)

    def _enable_initial_state(self, config):
        """Set correct state action on initializing"""
        if config.trace:
            self.enable_trace()
        elif config.debug:
            self.enable_debug()
        else:
            self.enable_default()

    def _extract_logging_config(self, config: "Config") -> dict:
        """Extract btlogging's config from bittensor config

        Args:
            config (bittensor.core.config.Config): Bittensor config instance.

        Returns:
            (dict): btlogging's config from Bittensor config or Bittensor config.
        """
        if hasattr(config, "logging"):
            return config.logging
        else:
            return config

    def _configure_handlers(self, config) -> list[stdlogging.Handler]:
        handlers = list()

        # stream handler, a given
        stream_handler = stdlogging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(self._stream_formatter)
        handlers.append(stream_handler)

        # file handler, maybe
        if config.record_log and config.logging_dir:
            logfile = os.path.abspath(
                os.path.join(config.logging_dir, DEFAULT_LOG_FILE_NAME)
            )
            file_handler = self._create_file_handler(logfile)
            handlers.append(file_handler)
        return handlers

    def get_config(self):
        return self._config

    def set_config(self, config: "Config"):
        """Set config after initialization, if desired.

        Args:
            config (bittensor.core.config.Config): Bittensor config instance.
        """
        self._config = config
        if config.logging_dir and config.record_log:
            expanded_dir = os.path.expanduser(config.logging_dir)
            logfile = os.path.abspath(os.path.join(expanded_dir, DEFAULT_LOG_FILE_NAME))
            self._enable_file_logging(logfile)
        if config.trace:
            self.enable_trace()
        elif config.debug:
            self.enable_debug()

    def _create_and_start_listener(self, handlers):
        """
        A listener to receive and publish log records.

        This listener receives records from a queue populated by the main bittensor logger, as well as 3rd party loggers
        """

        listener = QueueListener(self._queue, *handlers, respect_handler_level=True)
        listener.start()
        atexit.register(listener.stop)
        return listener

    def get_queue(self):
        """
        Get the queue the QueueListener is publishing from.

        To set up logging in a separate process, a QueueHandler must be added to all the desired loggers.
        """
        return self._queue

    def _initialize_bt_logger(self, name: str):
        """
        Initialize logging for bittensor.

        Since the initial state is Default, logging level for the module logger is INFO, and all third-party loggers are
        silenced. Subsequent state transitions will handle all logger outputs.
        """
        logger = stdlogging.getLogger(name)
        queue_handler = QueueHandler(self._queue)
        logger.addHandler(queue_handler)
        return logger

    def _deinitialize_bt_logger(self, name: str):
        """Find the logger by name and remove the queue handler associated with it."""
        logger = stdlogging.getLogger(name)
        for handler in list(logger.handlers):
            if isinstance(handler, QueueHandler):
                logger.removeHandler(handler)
        return logger

    def _create_file_handler(self, logfile: str):
        file_handler = RotatingFileHandler(
            logfile,
            maxBytes=DEFAULT_MAX_ROTATING_LOG_FILE_SIZE,
            backupCount=DEFAULT_LOG_BACKUP_COUNT,
        )
        file_handler.setFormatter(self._file_formatter)
        file_handler.setLevel(stdlogging.TRACE)
        return file_handler

    def register_primary_logger(self, name: str):
        """
        Register a logger as primary logger

        This adds a logger to the _primary_loggers set to ensure
        it doesn't get disabled when disabling third-party loggers.
        A queue handler is also associated with it.

        Args:
            name (str): the name for primary logger.
        """
        self._primary_loggers.add(name)
        self._initialize_bt_logger(name)

    def deregister_primary_logger(self, name: str):
        """
        De-registers a primary logger

        This function removes the logger from the _primary_loggers
        set and deinitializes its queue handler

        Args:
            name (str): the name of primary logger.
        """
        self._primary_loggers.remove(name)
        self._deinitialize_bt_logger(name)

    def enable_third_party_loggers(self):
        """Enables logging for third-party loggers by adding a queue handler to each."""
        for logger in all_loggers():
            if logger.name in self._primary_loggers:
                continue
            queue_handler = QueueHandler(self._queue)
            logger.addHandler(queue_handler)
            logger.setLevel(self._logger.level)

    def disable_third_party_loggers(self):
        """Disables logging for third-party loggers by removing all their handlers."""
        # remove all handlers
        for logger in all_loggers():
            if logger.name in self._primary_loggers:
                continue
            for handler in logger.handlers:
                logger.removeHandler(handler)

    def _enable_file_logging(self, logfile: str):
        # preserve idempotency; do not create extra filehandlers
        # if one already exists
        if any(
            [isinstance(handler, RotatingFileHandler) for handler in self._handlers]
        ):
            return
        file_handler = self._create_file_handler(logfile)
        self._handlers.append(file_handler)
        self._listener.handlers = tuple(self._handlers)

    # state transitions
    def before_transition(self, event, state):
        """Stops listener after transition."""
        self._listener.stop()

    def after_transition(self, event, state):
        """Starts listener after transition."""
        self._listener.start()

    # Default Logging
    def before_enable_default(self):
        """Logs status before enable Default."""
        self._logger.info("Enabling default logging.")
        self._logger.setLevel(stdlogging.WARNING)
        for logger in all_loggers():
            if logger.name in self._primary_loggers:
                continue
            logger.setLevel(stdlogging.CRITICAL)

    def before_enable_info(self):
        """Logs status before enable Default."""
        self._logger.info("Enabling default logging.")
        self._logger.setLevel(stdlogging.INFO)
        for logger in all_loggers():
            if logger.name in self._primary_loggers:
                continue
            logger.setLevel(stdlogging.CRITICAL)

    def after_enable_default(self):
        pass

    def before_enable_warning(self):
        """Logs status before enable Warning."""
        self._logger.info("Enabling warning.")
        self._stream_formatter.set_trace(True)
        for logger in all_loggers():
            logger.setLevel(stdlogging.WARNING)

    def after_enable_warning(self):
        """Logs status after enable Warning."""
        self._logger.info("Warning enabled.")

    # Trace
    def before_enable_trace(self):
        """Logs status before enable Trace."""
        self._logger.info("Enabling trace.")
        self._stream_formatter.set_trace(True)
        for logger in all_loggers():
            logger.setLevel(stdlogging.TRACE)

    def after_enable_trace(self):
        """Logs status after enable Trace."""
        self._logger.info("Trace enabled.")

    def before_disable_trace(self):
        """Logs status before disable Trace."""
        self._logger.info(f"Disabling trace.")
        self._stream_formatter.set_trace(False)
        self.enable_default()

    def after_disable_trace(self):
        """Logs status after disable Trace."""
        self._logger.info("Trace disabled.")

    # Debug
    def before_enable_debug(self):
        """Logs status before enable Debug."""
        self._logger.info("Enabling debug.")
        self._stream_formatter.set_trace(True)
        for logger in all_loggers():
            logger.setLevel(stdlogging.DEBUG)

    def after_enable_debug(self):
        """Logs status after enable Debug."""
        self._logger.info("Debug enabled.")

    def before_disable_debug(self):
        """Logs status before disable Debug."""
        self._logger.info("Disabling debug.")
        self._stream_formatter.set_trace(False)
        self.enable_default()

    def after_disable_debug(self):
        """Logs status after disable Debug."""
        self._logger.info("Debug disabled.")

    # Disable Logging
    def before_disable_logging(self):
        """
        Prepares the logging system for disabling.

        This method performs the following actions:
        1. Logs an informational message indicating that logging is being disabled.
        2. Disables trace mode in the stream formatter.
        3. Sets the logging level to CRITICAL for all loggers.

        This ensures that only critical messages will be logged after this method is called.
        """
        self._logger.info("Disabling logging.")
        self._stream_formatter.set_trace(False)

        for logger in all_loggers():
            logger.setLevel(stdlogging.CRITICAL)

    # Required API support log commands for API backwards compatibility.
    @property
    def __trace_on__(self) -> bool:
        """
        Checks if the current state is in "Trace" mode.

        Returns:
            bool: True if the current state is "Trace", otherwise False.
        """
        return self.current_state_value == "Trace"

    def trace(self, msg="", prefix="", suffix="", *args, **kwargs):
        """Wraps trace message with prefix and suffix."""
        msg = _concat_message(msg, prefix, suffix)
        self._logger.trace(msg, *args, **kwargs)

    def debug(self, msg="", prefix="", suffix="", *args, **kwargs):
        """Wraps debug message with prefix and suffix."""
        msg = _concat_message(msg, prefix, suffix)
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg="", prefix="", suffix="", *args, **kwargs):
        """Wraps info message with prefix and suffix."""
        msg = _concat_message(msg, prefix, suffix)
        self._logger.info(msg, *args, **kwargs)

    def success(self, msg="", prefix="", suffix="", *args, **kwargs):
        """Wraps success message with prefix and suffix."""
        msg = f"{prefix} - {msg} - {suffix}"
        self._logger.success(msg, *args, **kwargs)

    def warning(self, msg="", prefix="", suffix="", *args, **kwargs):
        """Wraps warning message with prefix and suffix."""
        msg = f"{prefix} - {msg} - {suffix}"
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg="", prefix="", suffix="", *args, **kwargs):
        """Wraps error message with prefix and suffix."""
        msg = f"{prefix} - {msg} - {suffix}"
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg="", prefix="", suffix="", *args, **kwargs):
        """Wraps critical message with prefix and suffix."""
        msg = f"{prefix} - {msg} - {suffix}"
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg="", prefix="", suffix="", *args, **kwargs):
        """Wraps exception message with prefix and suffix."""
        msg = f"{prefix} - {msg} - {suffix}"
        self._logger.exception(msg, *args, **kwargs)

    def on(self):
        """Enable default state."""
        self._logger.info("Logging enabled.")
        self.enable_default()

    def off(self):
        """Disables all states."""
        self.disable_logging()

    def set_debug(self, on: bool = True):
        """Sets Debug state."""
        if on and not self.current_state_value == "Debug":
            self.enable_debug()
        elif not on:
            if self.current_state_value == "Debug":
                self.disable_debug()

    def set_trace(self, on: bool = True):
        """Sets Trace state."""
        if on and not self.current_state_value == "Trace":
            self.enable_trace()
        elif not on:
            if self.current_state_value == "Trace":
                self.disable_trace()
    
    def set_warning(self, on: bool = True):
        """Sets Warning state."""
        if on and not self.current_state_value == "Warning":
            self.enable_warning()
        elif not on:
            if self.current_state_value == "Warning":
                self.disable_warning()

    def set_default(self):
        """Sets Default state."""
        if not self.current_state_value == "Default":
            self.enable_default()

    # as an option to be more obvious. `bittensor.logging.set_info()` is the same `bittensor.logging.set_default()`
    def set_info(self):
        """Sets Default state."""
        if not self.current_state_value == "Default":
            self.enable_info()
d
    def get_level(self) -> int:
        """Returns Logging level."""
        return self._logger.level

    def check_config(self, config: "Config"):
        assert config.logging

    def help(self):
        pass

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None):
        """Accept specific arguments fro parser"""
        prefix_str = "" if prefix is None else prefix + "."
        try:
            default_logging_debug = os.getenv("BT_LOGGING_DEBUG") or False
            default_logging_trace = os.getenv("BT_LOGGING_TRACE") or False
            default_logging_record_log = os.getenv("BT_LOGGING_RECORD_LOG") or False
            default_logging_logging_dir = os.getenv(
                "BT_LOGGING_LOGGING_DIR"
            ) or os.path.join("~", ".bittensor", "miners")
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

    @classmethod
    def config(cls) -> "Config":
        """Get config from the argument parser.

        Return:
            config (bittensor.core.config.Config): config object
        """
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        return Config(parser, args=[])

    def __call__(
        self,
        config: "Config" = None,
        debug: bool = None,
        trace: bool = None,
        record_log: bool = None,
        logging_dir: str = None,
    ):
        if config is not None:
            cfg = copy.deepcopy(config)
            if debug is not None:
                cfg.debug = debug
            elif trace is not None:
                cfg.trace = trace
            if record_log is not None:
                cfg.record_log = record_log
            if logging_dir is not None:
                cfg.logging_dir = logging_dir
        else:
            cfg = LoggingConfig(
                debug=debug, trace=trace, record_log=record_log, logging_dir=logging_dir
            )
        self.set_config(cfg)
