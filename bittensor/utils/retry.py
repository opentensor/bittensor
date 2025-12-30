import asyncio
import os
import time
import random
import logging
from typing import Type, Tuple, Optional, Callable, Any, Union

logger = logging.getLogger("bittensor.utils.retry")

# Note: This utility is not used internally by the SDK.
# It is provided as an optional helper for users who wish
# to implement consistent retry behavior themselves.


# Helpers for runtime environment variable access
def _retry_enabled() -> bool:
    return os.environ.get("BT_RETRY_ENABLED", "False").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )


def _retry_max_attempts() -> int:
    """Get the maximum number of retry attempts from the environment, with validation."""
    default = 3
    raw = os.environ.get("BT_RETRY_MAX_ATTEMPTS")
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for BT_RETRY_MAX_ATTEMPTS=%r; falling back to default %d",
            raw,
            default,
        )
        return default


def _retry_base_delay() -> float:
    """Get the base delay (in seconds) for retries from the environment, with validation."""
    default = 1.0
    raw = os.environ.get("BT_RETRY_BASE_DELAY")
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for BT_RETRY_BASE_DELAY=%r; falling back to default %.2f",
            raw,
            default,
        )
        return default


def _retry_max_delay() -> float:
    """Get the maximum delay (in seconds) for retries from the environment, with validation."""
    default = 60.0
    raw = os.environ.get("BT_RETRY_MAX_DELAY")
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for BT_RETRY_MAX_DELAY=%r; falling back to default %.2f",
            raw,
            default,
        )
        return default


_RETRY_BACKOFF_FACTOR = 2.0


def _retry_backoff_factor() -> float:
    return float(os.environ.get("BT_RETRY_BACKOFF_FACTOR", _RETRY_BACKOFF_FACTOR))


def _get_backoff_time(attempt: int, base_delay: float, max_delay: float) -> float:
    """Calculates backoff time with exponential backoff and jitter."""
    delay = min(max_delay, base_delay * (_retry_backoff_factor() ** attempt))
    # Add jitter: random value between 0 and delay
    return delay * (0.5 + random.random())


def retry_call(
    func: Callable,
    *args,
    retry_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (
        OSError,
        TimeoutError,
    ),
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    **kwargs,
) -> Any:
    """
    Synchronous retry wrapper.

    If BT_RETRY_ENABLED is False, executes the function exactly once.
    """
    if not _retry_enabled():
        return func(*args, **kwargs)

    # Resolve configuration
    _max_attempts = max_attempts if max_attempts is not None else _retry_max_attempts()
    _base_delay = base_delay if base_delay is not None else _retry_base_delay()
    _max_delay = max_delay if max_delay is not None else _retry_max_delay()

    last_exception = None

    for attempt in range(1, _max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except retry_exceptions as e:
            last_exception = e
            if attempt == _max_attempts:
                logger.debug(
                    f"Retry exhausted after {_max_attempts} attempts. Last error: {e}"
                )
                raise e

            backoff = _get_backoff_time(attempt - 1, _base_delay, _max_delay)
            logger.debug(
                f"Retry attempt {attempt}/{_max_attempts} failed with {e}. Retrying in {backoff:.2f}s..."
            )
            time.sleep(backoff)

    if last_exception:
        raise last_exception
    return None  # Should not be reached


async def retry_async(
    func: Callable,
    *args,
    retry_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (
        OSError,
        TimeoutError,
    ),
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    **kwargs,
) -> Any:
    """
    Asynchronous retry wrapper.

    If BT_RETRY_ENABLED is False, executes the function exactly once.
    """
    if not _retry_enabled():
        return await func(*args, **kwargs)

    # Resolve configuration
    _max_attempts = max_attempts if max_attempts is not None else _retry_max_attempts()
    _base_delay = base_delay if base_delay is not None else _retry_base_delay()
    _max_delay = max_delay if max_delay is not None else _retry_max_delay()

    last_exception = None

    for attempt in range(1, _max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except retry_exceptions as e:
            last_exception = e
            if attempt == _max_attempts:
                logger.debug(
                    f"Retry exhausted after {_max_attempts} attempts. Last error: {e}"
                )
                raise e

            backoff = _get_backoff_time(attempt - 1, _base_delay, _max_delay)
            logger.debug(
                f"Retry attempt {attempt}/{_max_attempts} failed with {e}. Retrying in {backoff:.2f}s..."
            )
            await asyncio.sleep(backoff)

    if last_exception:
        raise last_exception
    return None  # Should not be reached
