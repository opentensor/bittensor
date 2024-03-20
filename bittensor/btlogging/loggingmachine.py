import os
import sys
import atexit
import inspect
import threading
import multiprocessing as mp
import logging as stdlogging
from typing import NamedTuple
from statemachine import StateMachine, State
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

from bittensor.btlogging.format import BtStreamFormatter, BtFileFormatter
from bittensor.btlogging.helpers import all_loggers
from bittensor.btlogging.defines import (
    BASE_LOG_FORMAT,
    TRACE_LOG_FORMAT,
    DATE_FORMAT,
    BITTENSOR_LOGGER_NAME,
    DEFAULT_LOG_FILE_NAME,
    DEFAULT_MAX_ROTATING_LOG_FILE_SIZE,
    DEFAULT_LOG_BACKUP_COUNT
)


class LoggingConfig(NamedTuple):
    debug: bool
    trace: bool
    record_log: bool
    logging_dir: str


class LoggingMachine(StateMachine):
    """
    Handles logger states for bittensor and 3rd party libraries
    """

    Default = State(initial=True)
    Debug = State()
    Trace = State()
    Disabled = State()

    enable_default = (
        Debug.to(Default)
        | Trace.to(Default)
        | Disabled.to(Default)
        | Default.to(Default)
    )

    enable_trace = (
        Default.to(Trace)
        | Debug.to(Trace)
        | Disabled.to(Trace)
        | Trace.to(Trace)
    )

    enable_debug = (
        Default.to(Debug)
        | Trace.to(Debug)
        | Disabled.to(Debug)
        | Debug.to(Debug)
    )

    disable_trace = (
        Trace.to(Default)
    )

    disable_debug = (
        Debug.to(Default)
    )

    disable_logging = (
        Trace.to(Disabled)
        | Debug.to(Disabled)
        | Default.to(Disabled)
        | Disabled.to(Disabled)
    )

    def __init__(self, config: "bittensor.config", name: str=BITTENSOR_LOGGER_NAME):
        # set initial state based on config
        super(LoggingMachine, self).__init__()

        # basics
        self._queue = mp.Queue(-1) # all 
        self._name = name
        # self._lock = threading.Lock()
        self._state_change_event = threading.Event()
        self._state_change_event.set()

        # handlers
        # TODO: change this to a list of handlers, so that self.listener.handlers
        #   can be updated to modify sinks
        # self._handlers = list()

        self._stream_formatter = BtStreamFormatter()
        self._file_formatter = BtFileFormatter(TRACE_LOG_FORMAT, DATE_FORMAT)

        self.set_config(config)

        self._logger = self._initialize_bt_logger(name, config)
        self._initialize_external_loggers(config)
        
    
    def get_config(self):
        return self._config

    def set_config(self, config):
        self._config = config
        if config.logging_dir and config.record_log:
            logfile = os.path.abspath(os.path.join(config.logging_dir, DEFAULT_LOG_FILE_NAME))
            # self.info(f"Enabling file logging to: {logfile}")
            self._enable_file_logging(logfile)
        if config.trace:
            self.enable_trace()
        elif config.debug:
            self.enable_debug()

    def _create_listener(self, handlers):
        """
        A listener to receive and publish log records.
        
        This listener receives records from a queue populated by the main bittensor
        logger, as well as 3rd party loggers. The output sinks 
        """

        listener = QueueListener(
            self._queue, 
            *handlers,
            respect_handler_level=True
        )
        return listener

    def get_queue(self):
        if hasattr(self, "_queue"):
            return self._queue
        else:
            raise AttributeError("File logging is not enabled, no queue available.")

    def _initialize_bt_logger(self, name, config):
        """
        Initialize logging for bittensor.

        Since the initial state is Default, logging level for the module logger 
        is INFO, and all third-party loggers are silenced. Subsequent state 
        transitions will handle all logger outputs.
        """
        logger = stdlogging.getLogger(name)
        queue_handler = QueueHandler(self._queue)
        logger.addHandler(queue_handler)
        handlers = list()

        # main handler
        stream_handler = stdlogging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(self._stream_formatter)
        handlers.append(stream_handler)

        if config.record_log and config.logging_dir:
            # logfile = os.path.abspath(os.path.join(config.logging_dir, DEFAULT_LOG_FILE_NAME))
            # handlers.append(self._create_file_handler(logfile))
            pass

        self._handlers = handlers
        self._listener = self._create_listener(handlers)
        self._listener.start()
        atexit.register(self._listener.stop)
        
        return logger

    def _create_file_handler(self, logfile: str):
        file_handler = RotatingFileHandler(
                logfile, 
                maxBytes=DEFAULT_MAX_ROTATING_LOG_FILE_SIZE,
                backupCount=DEFAULT_LOG_BACKUP_COUNT
            )
        file_handler.setFormatter(self._file_formatter)
        file_handler.setLevel(stdlogging.TRACE)
        return file_handler

    def _initialize_external_loggers(self, config):
        # remove all handlers
        for logger in all_loggers():
            if logger.name == self._name:
                continue
            for handler in logger.handlers:
                logger.removeHandler(handler)
            queue_handler = QueueHandler(self._queue)
            logger.addHandler(queue_handler)
            logger.setLevel(stdlogging.CRITICAL)    

    def _enable_file_logging(self, logfile:str):
        # preserve idempotency; do not create extra filehandlers
        # if one already exists
        # if any([isinstance(handler, RotatingFileHandler) for handler in self._handlers]):
        #     return
        file_handler = self._create_file_handler(logfile)
        self._handlers.append(file_handler)
        self._listener.stop()
        self._listener.handlers = tuple(self._handlers)
        self._listener.start()

    # state transitions
    # Default Logging
    def before_enable_default(self):
        # with self._lock:
        self._logger.info(f"Enabling default logging.")
        self._state_change_event.clear()
        self._logger.setLevel(stdlogging.INFO)
        self._stream_formatter.set_trace(False)
        for logger in all_loggers():
            if logger.name == self._name:
                continue
            logger.setLevel(stdlogging.CRITICAL)
        self._state_change_event.set()

    def after_enable_default(self):
        pass

    # Trace
    def before_enable_trace(self):
        # with self._lock:
        self._logger.info("Enabling trace.")
        self._state_change_event.clear()
        self._stream_formatter.set_trace(True)
        for logger in all_loggers():
            logger.setLevel(stdlogging.TRACE)
        self._state_change_event.set()

    def after_enable_trace(self):
        self._logger.info("Trace enabled.")

    def before_disable_trace(self):
        # with self._lock:
        self._logger.info(f"Disabling trace.")
        self._state_change_event.clear()
        self._stream_formatter.set_trace(False)
        self.enable_default()
        self._state_change_event.set()
    
    def after_disable_trace(self):
        self._logger.info("Trace disabled.")

    # Debug
    def before_enable_debug(self):
        # with self._lock:
        self._logger.info("Enabling debug.")
        self._state_change_event.clear()
        self._stream_formatter.set_trace(True)
        self._state_change_event.set()
        for logger in all_loggers():
            logger.setLevel(stdlogging.DEBUG)

    def after_enable_debug(self):
        self._logger.info("Debug enabled.")

    def before_disable_debug(self):
        # with self._lock:
        self._logger.info("Disabling debug.")
        self._state_change_event.clear()
        self._stream_formatter.set_trace(False)
        self.enable_default()
        self._state_change_event.set()

    def after_disable_debug(self):
        self._logger.info("Debug disabled.")
    
    # Disable Logging
    def before_disable_logging(self):
        # with self._lock:
        self._logger.info("Disabling logging.")        
        self._state_change_event.set()
        self._stream_formatter.set_trace(False)
        self._state_change_event.clear()
        
        for logger in all_loggers():
            logger.setLevel(stdlogging.CRITICAL)
    
    # Required API
    # support log commands for API backwards compatibility 
    def trace(self, msg, *args, **kwargs):
        self._state_change_event.wait()
        self._logger.trace(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._state_change_event.wait()
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._state_change_event.wait()
        self._logger.info(msg, *args, **kwargs)

    def success(self, msg, *args, **kwargs):
        self._state_change_event.wait()
        self._logger.success(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._state_change_event.wait()
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._state_change_event.wait()
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._state_change_event.wait()
        self._logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg, *args, **kwargs):
        self._state_change_event.wait()
        self._logger.exception(msg, *args, **kwargs)

    def on(self):
        self._logger.info("Logging enabled.")
        self.enable_default()

    def off(self):
        self.disable_logging()

    def set_debug(self, on: bool=True):
        if on and not self.current_state_value == "Debug":
            self.enable_debug()
        elif not on:
            if self.current_state_value == "Debug":
                self.disable_debug()
            

    def set_trace(self, on: bool=True):
        if on and not self.current_state_value == "Trace":
            self.enable_trace()
        elif not on:
            if self.current_state_value == "Trace":
                self.disable_trace()
    
    def get_level(self):
        return self._logger.level

    def help(self):
        pass