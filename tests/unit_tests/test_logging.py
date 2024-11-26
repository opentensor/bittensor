import logging as stdlogging
import multiprocessing
from unittest.mock import MagicMock, patch

import pytest

from bittensor.utils.btlogging import LoggingMachine
from bittensor.utils.btlogging.defines import (
    DEFAULT_LOG_FILE_NAME,
    BITTENSOR_LOGGER_NAME,
)
from bittensor.utils.btlogging.loggingmachine import LoggingConfig, _concat_message


@pytest.fixture(autouse=True, scope="session")
def disable_stdout_streaming():
    # Backup original handlers
    original_handlers = stdlogging.root.handlers[:]

    # Remove all handlers that stream to stdout
    stdlogging.root.handlers = [
        h
        for h in stdlogging.root.handlers
        if not isinstance(h, stdlogging.StreamHandler)
    ]

    yield  # Yield control to the test or fixture setup

    # Restore original handlers after the test
    stdlogging.root.handlers = original_handlers


@pytest.fixture
def mock_config(tmp_path):
    # Using pytest's tmp_path fixture to generate a temporary directory
    log_dir = tmp_path / "logs"
    log_dir.mkdir()  # Create the temporary directory
    log_file_path = log_dir / DEFAULT_LOG_FILE_NAME

    mock_config = LoggingConfig(
        debug=False, trace=False, record_log=True, logging_dir=str(log_dir)
    )

    yield mock_config, log_file_path
    # Cleanup: No need to explicitly delete the log file or directory, tmp_path does it automatically


@pytest.fixture
def logging_machine(mock_config):
    config, _ = mock_config
    logging_machine = LoggingMachine(config=config)
    return logging_machine


def test_initialization(logging_machine, mock_config):
    """
    Test initialization of LoggingMachine.
    """
    config, log_file_path = mock_config  # Unpack to get the log_file_path

    assert logging_machine.get_queue() is not None
    assert isinstance(logging_machine.get_queue(), multiprocessing.queues.Queue)
    assert logging_machine.get_config() == config

    # Ensure that handlers are set up correctly
    assert any(
        isinstance(handler, stdlogging.StreamHandler)
        for handler in logging_machine._handlers
    )
    if config.record_log and config.logging_dir:
        assert any(
            isinstance(handler, stdlogging.FileHandler)
            for handler in logging_machine._handlers
        )
        assert log_file_path.exists()  # Check if log file is created


def test_state_transitions(logging_machine, mock_config):
    """
    Test state transitions and the associated logging level changes.
    """
    config, log_file_path = mock_config
    with patch(
        "bittensor.utils.btlogging.loggingmachine.all_loggers"
    ) as mocked_all_loggers:
        # mock the main bittensor logger, identified by its `name` field
        mocked_bt_logger = MagicMock()
        mocked_bt_logger.name = BITTENSOR_LOGGER_NAME
        # third party loggers are treated differently and silenced under default
        # logging settings
        mocked_third_party_logger = MagicMock()
        logging_machine._logger = mocked_bt_logger
        mocked_all_loggers.return_value = [mocked_third_party_logger, mocked_bt_logger]

        # Enable/Disable Debug
        # from default
        assert logging_machine.current_state_value == "Default"
        logging_machine.enable_debug()
        assert logging_machine.current_state_value == "Debug"
        # check log levels
        mocked_bt_logger.setLevel.assert_called_with(stdlogging.DEBUG)
        mocked_third_party_logger.setLevel.assert_called_with(stdlogging.DEBUG)

        logging_machine.disable_debug()

        # Enable/Disable Trace
        assert logging_machine.current_state_value == "Default"
        logging_machine.enable_trace()
        assert logging_machine.current_state_value == "Trace"
        # check log levels
        mocked_bt_logger.setLevel.assert_called_with(stdlogging.TRACE)
        mocked_third_party_logger.setLevel.assert_called_with(stdlogging.TRACE)
        logging_machine.disable_trace()
        assert logging_machine.current_state_value == "Default"

        # Enable Default
        logging_machine.enable_debug()
        assert logging_machine.current_state_value == "Debug"
        logging_machine.enable_default()
        assert logging_machine.current_state_value == "Default"
        # main logger set to INFO
        mocked_bt_logger.setLevel.assert_called_with(stdlogging.WARNING)
        # 3rd party loggers should be disabled by setting to CRITICAL
        mocked_third_party_logger.setLevel.assert_called_with(stdlogging.CRITICAL)

        # Disable Logging
        # from default
        logging_machine.disable_logging()
        assert logging_machine.current_state_value == "Disabled"
        mocked_bt_logger.setLevel.assert_called_with(stdlogging.CRITICAL)
        mocked_third_party_logger.setLevel.assert_called_with(stdlogging.CRITICAL)


def test_enable_file_logging_with_new_config(tmp_path):
    """
    Test enabling file logging by setting a new config.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()  # Create the temporary directory
    log_file_path = log_dir / DEFAULT_LOG_FILE_NAME

    # check no file handler is created
    config = LoggingConfig(debug=False, trace=False, record_log=True, logging_dir=None)
    lm = LoggingMachine(config)
    assert not any(
        isinstance(handler, stdlogging.FileHandler) for handler in lm._handlers
    )

    # check file handler now exists
    new_config = LoggingConfig(
        debug=False, trace=False, record_log=True, logging_dir=str(log_dir)
    )
    lm.set_config(new_config)
    assert any(isinstance(handler, stdlogging.FileHandler) for handler in lm._handlers)


def test_all_log_levels_output(logging_machine, caplog):
    """
    Test that all log levels are captured.
    """
    logging_machine.set_trace()

    logging_machine.trace("Test trace")
    logging_machine.debug("Test debug")
    logging_machine.info("Test info")
    logging_machine.success("Test success")
    logging_machine.warning("Test warning")
    logging_machine.error("Test error")
    logging_machine.critical("Test critical")

    assert "Test trace" in caplog.text
    assert "Test debug" in caplog.text
    assert "Test info" in caplog.text
    assert "Test success" in caplog.text
    assert "Test warning" in caplog.text
    assert "Test error" in caplog.text
    assert "Test critical" in caplog.text

    records = [(r.module, r.getMessage()) for r in caplog.records]

    assert records == [
        ("loggingmachine", "Trace enabled."),
        ("test_logging", "Test trace"),
        ("test_logging", "Test debug"),
        ("test_logging", "Test info"),
        ("test_logging", "Test success"),
        ("test_logging", "Test warning"),
        ("test_logging", "Test error"),
        ("test_logging", "Test critical"),
    ]


@pytest.mark.parametrize(
    "msg, prefix, suffix, expected_result",
    [
        ("msg", "", "", "msg"),
        ("msg", None, None, "msg"),
        ("msg", "prefix", None, "prefix - msg"),
        ("msg", None, "suffix", "msg - suffix"),
        ("msg", "prefix", "suffix", "prefix - msg - suffix"),
    ],
    ids=[
        "message, no prefix (str), no suffix (str)",
        "message, no prefix (None), no suffix (None)",
        "message and prefix only",
        "message and suffix only",
        "message, prefix, and suffix",
    ],
)
def test_concat(msg, prefix, suffix, expected_result):
    """Test different options of message concatenation with prefix and suffix."""
    assert _concat_message(msg, prefix, suffix) == expected_result
