import sys
import logging as stdlogging
from statemachine import StateMachine, State

from bittensor.btlogging.format import BtStreamFormatter
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

        self._name = name
        self._stream_formatter = BtStreamFormatter()
        self._file_formatter = stdlogging.Formatter(BASE_LOG_FORMAT, DATE_FORMAT)

        self._logger = self._initialize_bt_logger(name, config)
        self._initialize_external_loggers(config)
        self.config = config

        # set file handler logic
        if config.trace:
            self.enable_trace()
        elif config.debug:
            self.enable_debug()
    
    def _initialize_bt_logger(self, name, config):
        """
        Initialize logging for bittensor.

        Since the initial state is Default, logging level for the module logger 
        is INFO, and all third-party loggers are silenced. Subsequent state 
        transitions will handle all logger outputs.
        """
        logger = stdlogging.getLogger(name)

        # set stream handler
        sh = stdlogging.StreamHandler(sys.stdout)
        sh.setFormatter(self._stream_formatter)
        logger.addHandler(sh)
        logger.setLevel(stdlogging.INFO)

        # set file handler, if applicable
        if config.logging_dir and config.record_log:
            file_formatter = stdlogging.Formatter(BASE_LOG_FORMAT, DATE_FORMAT)
            filepath = os.path.join(config.logging_dir, DEFAULT_LOG_FILE_NAME)
            fh = stdlogging.handlers.RotatingFileHandler(
                filepath, 
                maxBytes=DEFAULT_MAX_ROTATING_LOG_FILE_SIZE,
                backupCount=DEFAULT_LOG_BACKUP_COUNT
            )
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)
            
        
        return logger

    def _initialize_external_loggers(self, config):
        # remove all handlers
        for logger in all_loggers():
            if logger.name == self._name:
                continue
            for handler in logger.handlers:
                logger.removeHandler(handler)
            
            # stream handling
            sh = stdlogging.StreamHandler(sys.stdout)
            sh.setFormatter(self._stream_formatter)
            logger.addHandler(sh)
            # initial state is default, hence we silence all loggers
            logger.setLevel(stdlogging.CRITICAL)

            if config.logging_dir and config.record_log:
                # TODO: handle multiple loggers logging to the same file.
                pass
                # file_formatter = logging.Formatter(BASE_LOG_FORMAT, DATE_FORMAT)
                # filepath = os.path.join(config.logging_dir, DEFAULT_LOG_FILE_NAME)
                # fh = logging.handlers.RotatingFileHandler(
                #     filepath, 
                #     maxBytes=DEFAULT_MAX_ROTATING_LOG_FILE_SIZE,
                #     backupCount=DEFAULT_LOG_BACKUP_COUNT
                # )
                # fh.setFormatter(file_formatter)
                # self._logger.addHandler(fh)
    
    def get_logger(self, name: str) -> stdlogging.Logger:
        return self._logger

    # state transitions
    # Default Logging
    def before_enable_default(self):
        if self.current_state_value == "Default":
            return
        self._logger.setLevel(stdlogging.INFO)
        self._stream_formatter.set_trace(False)
        for logger in all_loggers():
            if logger.name == self._name:
                continue
            logger.setLevel(stdlogging.CRITICAL)

    def after_enable_default(self):
        pass
            
    # Trace
    def before_enable_trace(self):
        if self.current_state_value == "Trace":
            return 
        self._logger.info("Enabling trace.")
        self._stream_formatter.set_trace(True)
        for logger in all_loggers():
            logger.setLevel(stdlogging.TRACE)

    def after_enable_trace(self):
        self._logger.info("Trace enabled.")

    def before_disable_trace(self):
        self._logger.info("Disabling trace.")
        self._stream_formatter.set_trace(False)
        self.enable_default()
    
    def after_disable_trace(self):
        self._logger.info("Trace disabled.")

    # Debug
    def before_enable_debug(self):
        if self.current_state_value == "Debug":
            return
        self._logger.info("Enabling debug.")
        self._stream_formatter.set_trace(True)
        for logger in all_loggers():
            logger.setLevel(stdlogging.DEBUG)

    def after_enable_debug(self):
        self._logger.info("Debug enabled.")

    def before_disable_debug(self):
        self._logger.info("Disabling debug.")
        self._stream_formatter.set_trace(False)
        self.enable_default()
    
    def after_disable_debug(self):
        self._logger.info("Debug disabled.")
    
    # Disable Logging
    def before_disable_logging(self):
        self._logger.info("Disabling logging.")
        self._stream_formatter.set_trace(False)
        for logger in all_loggers():
            logger.setLevel(stdlogging.CRITICAL)
    
    # Required API
    # support log commands for API backwards compatibility 
    def trace(self, msg, *args, **kwargs):
        self._logger.trace(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def success(self, msg, *args, **kwargs):
        self._logger.success(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg, *args, **kwargs):
        self._logger.exception(msg, *args, **kwargs)

    def on(self):
        self.enable_default()

    def off(self):
        self.disable_logging()

    def set_debug(self, on: bool=True):
        if on:
            self.enable_debug()
        else:
            if self.current_state_value == "Debug":
                self.disable_debug()
            

    def set_trace(self, on: bool=True):
        if on:
            self.enable_trace()
        else:
            if self.current_state_value == "Trace":
                self.disable_trace()
    
    def get_level(self):
        return self._logger.level

    def help(self):
        pass