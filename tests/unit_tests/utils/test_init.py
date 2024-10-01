import pytest

from bittensor import warnings, __getattr__, version_split, logging, trace, debug


def test_getattr_version_split():
    """Test that __getattr__ for 'version_split' issues a deprecation warning and returns the correct value."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert __getattr__("version_split") == version_split
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "version_split is deprecated" in str(w[-1].message)


@pytest.mark.parametrize("test_input, expected", [(True, "Trace"), (False, "Default")])
def test_trace(test_input, expected):
    """Test the trace function turns tracing on|off."""
    trace(test_input)
    assert logging.current_state_value == expected


@pytest.mark.parametrize("test_input, expected", [(True, "Debug"), (False, "Default")])
def test_debug(test_input, expected):
    """Test the debug function turns tracing on|off."""
    debug(test_input)
    assert logging.current_state_value == expected
