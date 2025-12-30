import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from bittensor.utils.retry import retry_call, retry_async

# Create custom exception for testing
class NetworkError(Exception):
    pass

class NonRetryableError(Exception):
    pass

@pytest.fixture
def mock_sleep():
    with patch("time.sleep") as m:
        yield m

@pytest.fixture
def mock_async_sleep():
    with patch("asyncio.sleep", new_callable=AsyncMock) as m:
        yield m

@pytest.fixture
def enable_retries():
    # Patch environment variables
    with patch.dict("os.environ", {"BT_RETRY_ENABLED": "True"}):
        yield

@pytest.fixture
def disable_retries():
    # Patch environment variables
    with patch.dict("os.environ", {"BT_RETRY_ENABLED": "False"}):
        yield

# --- Sync Tests ---

def test_sync_retry_success(enable_retries):
    mock_func = Mock(return_value="success")
    result = retry_call(mock_func, retry_exceptions=(NetworkError,), max_attempts=3)
    assert result == "success"
    assert mock_func.call_count == 1

def test_sync_retry_eventual_success(enable_retries, mock_sleep):
    mock_func = Mock(side_effect=[NetworkError("Fail 1"), NetworkError("Fail 2"), "success"])
    result = retry_call(mock_func, retry_exceptions=(NetworkError,), max_attempts=3)
    assert result == "success"
    assert mock_func.call_count == 3

def test_sync_retry_exhaustion(enable_retries, mock_sleep):
    mock_func = Mock(side_effect=NetworkError("Persistent Fail"))
    with pytest.raises(NetworkError, match="Persistent Fail"):
        retry_call(mock_func, retry_exceptions=(NetworkError,), max_attempts=3)
    assert mock_func.call_count == 3

def test_sync_no_retry_on_wrong_exception(enable_retries):
    mock_func = Mock(side_effect=NonRetryableError("Fatal"))
    with pytest.raises(NonRetryableError, match="Fatal"):
        retry_call(mock_func, retry_exceptions=(NetworkError,), max_attempts=3)
    assert mock_func.call_count == 1

def test_sync_disabled_retries_executes_once(disable_retries):
    mock_func = Mock(side_effect=NetworkError("Fail"))
    with pytest.raises(NetworkError, match="Fail"):
        retry_call(mock_func, retry_exceptions=(NetworkError,), max_attempts=3)
    assert mock_func.call_count == 1

def test_sync_default_retry_exceptions_do_not_retry_non_network_error(enable_retries):
    mock_func = Mock(side_effect=ValueError("bad input"))
    with pytest.raises(ValueError, match="bad input"):
        # Should raise immediately because ValueError is not in (OSError, TimeoutError)
        retry_call(mock_func)
    assert mock_func.call_count == 1


# --- Async Tests ---

@pytest.mark.asyncio
async def test_async_retry_success(enable_retries):
    mock_func = AsyncMock(return_value="success")
    result = await retry_async(mock_func, retry_exceptions=(NetworkError,), max_attempts=3)
    assert result == "success"
    assert mock_func.call_count == 1

@pytest.mark.asyncio
async def test_async_retry_eventual_success(enable_retries, mock_async_sleep):
    mock_func = AsyncMock(side_effect=[NetworkError("Fail 1"), NetworkError("Fail 2"), "success"])
    result = await retry_async(mock_func, retry_exceptions=(NetworkError,), max_attempts=3)
    assert result == "success"
    assert mock_func.call_count == 3

@pytest.mark.asyncio
async def test_async_retry_exhaustion(enable_retries, mock_async_sleep):
    mock_func = AsyncMock(side_effect=NetworkError("Persistent Fail"))
    with pytest.raises(NetworkError, match="Persistent Fail"):
        await retry_async(mock_func, retry_exceptions=(NetworkError,), max_attempts=3)
    assert mock_func.call_count == 3

@pytest.mark.asyncio
async def test_async_no_retry_on_wrong_exception(enable_retries):
    mock_func = AsyncMock(side_effect=NonRetryableError("Fatal"))
    with pytest.raises(NonRetryableError, match="Fatal"):
        await retry_async(mock_func, retry_exceptions=(NetworkError,), max_attempts=3)
    assert mock_func.call_count == 1

@pytest.mark.asyncio
async def test_async_disabled_retries_executes_once(disable_retries):
    mock_func = AsyncMock(side_effect=NetworkError("Fail"))
    with pytest.raises(NetworkError, match="Fail"):
        await retry_async(mock_func, retry_exceptions=(NetworkError,), max_attempts=3)
    assert mock_func.call_count == 1

