import logging
import pytest

from bittensor.logging import (
    getLogger, 
    trace, 
    __enable_non_bt_loggers, 
    __disable_non_bt_loggers,
    __enable_trace, 
    __disable_logger, 
    __set_default_logging,
    __set_bt_format_logger, 
    BITTENSOR_LOGGER_NAME
)
from bittensor.logging.format import BtStreamFormatter


# Mock for all_loggers function
@pytest.fixture
def all_loggers_mock(mocker):
    mock = mocker.patch('your_module.all_loggers')
    mock.return_value = [logging.getLogger(name) for name in ['test_logger1', 'test_logger2']]
    return mock

def test_getLogger(mock_stdout):
    logger = getLogger('testLogger')
    

def test_trace_enable(mocker, mock_stdout):
    bt_logger = logging.getLogger(BITTENSOR_LOGGER_NAME)
    trace()
    # This assumes TRACE is a custom level you've set somewhere as logging.TRACE might not be a default level
    assert bt_logger.level == logging.TRACE  # Use DEBUG for illustration

def test_enable_non_bt_loggers(all_loggers_mock, mock_stdout):
    __enable_non_bt_loggers()
    for logger in all_loggers():
        if logger.name != BITTENSOR_LOGGER_NAME:
            assert logger.level == logging.INFO  # Assuming INFO is the default
            assert any([handler.formatter is BtStreamFormatter for handler in logger.handlers])

def test_disable_non_bt_loggers(all_loggers_mock, mock_stdout):
    __disable_non_bt_loggers()
    for logger in all_loggers_mock.return_value:
        if logger.name != BITTENSOR_LOGGER_NAME:
            assert logger.level == logging.CRITICAL

def test_enable_trace(mock_stdout):
    logger = logging.getLogger('test_trace')
    __enable_trace(logger)
    # Assuming TRACE level and checking if formatter's trace is enabled
    for handler in logger.handlers:
        assert isinstance(handler.formatter, BtStreamFormatter)
        assert handler.formatter.trace_enabled == True

def test_disable_logger(mock_stdout):
    logger = logging.getLogger('test_disable')
    __disable_logger(logger)
    # Assuming disabling sets level to CRITICAL and trace to False
    assert logger.level == logging.CRITICAL
    for handler in logger.handlers:
        assert isinstance(handler.formatter, BtStreamFormatter)
        assert handler.formatter.trace_enabled == False

# Further tests would continue in this manner, focusing on the behavior expected from each function.

