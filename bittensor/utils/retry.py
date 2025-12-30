"""Retry utilities for handling transient failures with exponential backoff. 

This module provides optional retry wrappers for both synchronous and asynchronous
functions.  Retry behavior is controlled via environment variables and is disabled
by default. 

Environment Variables:
    BT_RETRY_ENABLED:  Enable retry behavior ("true", "1", "yes", "on")
    BT_RETRY_MAX_ATTEMPTS: Maximum retry attempts (default: 3)
    BT_RETRY_BASE_DELAY: Base delay in seconds (default: 1.0)
    BT_RETRY_MAX_DELAY: Maximum delay in seconds (default: 60.0)
    BT_RETRY_BACKOFF_FACTOR: Exponential backoff multiplier (default: 2.0)

Note: 
    This utility is not used internally by the SDK.  It is provided as an
    optional helper for users who wish to implement consistent retry behavior. 

For more information on retry strategies, see:
    https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
"""

import asyncio
import inspect
import os
import time
import random
import logging
from typing import Type, Tuple, Optional, Callable, Any, Union

logger = logging.getLogger("bittensor. utils.retry")


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
        value = int(raw)
        if value <= 0:
            logger.warning(
                "Invalid value for BT_RETRY_MAX_ATTEMPTS=%r (must be positive); falling back to default %d",
                raw,
                default,
            )
            return default
        return value
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
        value = float(raw)
        if value < 0:
            logger.warning(
                "Invalid value for BT_RETRY_BASE_DELAY=%r (must be non-negative); falling back to default %. 2f",
                raw,
                default,
            )
            return default
        return value
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
        value = float(raw)
        if value < 0:
            logger.warning(
                "Invalid value for BT_RETRY_MAX_DELAY=%r (must be non-negative); falling back to default %.2f",
                raw,
                default,
            )
            return default
        return value
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for BT_RETRY_MAX_DELAY=%r; falling back to default %.2f",
            raw,
            default,
        )
        return default


_RETRY_BACKOFF_FACTOR = 2.0


def _retry_backoff_factor() -> float:
    """Get the backoff factor for exponential backoff from the environment, with validation."""
    default = _RETRY_BACKOFF_FACTOR
    raw = os.environ.get("BT_RETRY_BACKOFF_FACTOR")
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
        if value <= 0:
            logger.warning(
                "Invalid value for BT_RETRY_BACKOFF_FACTOR=%r (must be positive); falling back to default %.2f",
                raw,
                default,
            )
            return default
        return value
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for BT_RETRY_BACKOFF_FACTOR=%r (must be positive); falling back to default %.2f",
            raw,
            default,
        )
        return default


def _get_backoff_time(attempt: int, base_delay: float, max_delay: float) -> float:
    """Calculates backoff time with exponential backoff and jitter."""
    delay = min(max_delay, base_delay * (_retry_backoff_factor() ** attempt))
    # Add jitter while ensuring the final backoff does not exceed max_delay
    return min(max_delay, delay * (0.5 + random.random()))


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
    """Synchronous retry wrapper with optional exponential backoff.

    Retries are only enabled when BT_RETRY_ENABLED is set to a truthy value. 
    When disabled, the function executes exactly once. 

    Args:
        func:  The callable to be executed and potentially retried.
        *args: Positional arguments forwarded to func.
        retry_exceptions: Exception type(s) that trigger a retry.  Any exception
            not matching these types is raised immediately.  Defaults to
            (OSError, TimeoutError).
        max_attempts: Maximum number of attempts. If None, uses
            BT_RETRY_MAX_ATTEMPTS environment variable (default: 3).
        base_delay: Base delay in seconds for exponential backoff. If None,
            uses BT_RETRY_BASE_DELAY environment variable (default: 1.0).
        max_delay: Maximum delay in seconds between attempts. If None, uses
            BT_RETRY_MAX_DELAY environment variable (default: 60.0).
        **kwargs:  Keyword arguments forwarded to func. 

    Returns:
        The return value from the first successful func execution. 

    Raises:
        TypeError: If func is an async function.  Use async_retry_call instead.
        Exception: Any exception raised by func when retries are disabled, or
            when the exception type doesn't match retry_exceptions, or after
            all retry attempts are exhausted.
    """
    # Validate that func is not async
    if inspect.iscoroutinefunction(func):
        raise TypeError(
            f"retry_call() cannot be used with async functions. "
            f"Use async_retry_call() instead for {func.__name__}."
        )

    if not _retry_enabled():
        return func(*args, **kwargs)

    # Resolve configuration
    _max_attempts = max_attempts if max_attempts is not None else _retry_max_attempts()
    _base_delay = base_delay if base_delay is not None else _retry_base_delay()
    _max_delay = max_delay if max_delay is not None else _retry_max_delay()

    for attempt in range(1, _max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except retry_exceptions as e:
            if attempt == _max_attempts:
                logger.debug(
                    f"Retry exhausted after {_max_attempts} attempts. Last error: {e}"
                )
                raise

            backoff = _get_backoff_time(attempt - 1, _base_delay, _max_delay)
            logger.debug(
                f"Retry attempt {attempt}/{_max_attempts} failed with {e}.  Retrying in {backoff:. 2f}s..."
            )
            time.sleep(backoff)

    # This should never be reached due to the logic above
    assert False, "Unreachable code"


async def async_retry_call(
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
    """Asynchronous retry wrapper with optional exponential backoff.

    Retries are only enabled when BT_RETRY_ENABLED is set to a truthy value.
    When disabled, the function executes exactly once.

    Args:
        func: The async callable to be executed and potentially retried.
        *args: Positional arguments forwarded to func on each attempt.
        retry_exceptions: Exception type(s) that trigger a retry. Any exception
            not matching these types is raised immediately. Defaults to
            (OSError, TimeoutError).
        max_attempts: Maximum number of attempts. If None, uses
            BT_RETRY_MAX_ATTEMPTS environment variable (default: 3).
        base_delay: Base delay in seconds for exponential backoff. If None,
            uses BT_RETRY_BASE_DELAY environment variable (default: 1.0).
        max_delay: Maximum delay in seconds between attempts. If None, uses
            BT_RETRY_MAX_DELAY environment variable (default: 60.0).
        **kwargs: Keyword arguments forwarded to func on each attempt.

    Returns:
        The result from the first successful func execution.

    Raises:
        TypeError: If func is not an async function. Use retry_call instead.
        Exception: Any exception raised by func when retries are disabled, or
            when the exception type doesn't match retry_exceptions, or after
            all retry attempts are exhausted.
    """
    # Validate that func is async
    if not inspect.iscoroutinefunction(func):
        raise TypeError(
            f"async_retry_call() requires an async function. "
            f"Use retry_call() instead for {func.__name__}."
        )

    if not _retry_enabled():
        return await func(*args, **kwargs)

    # Resolve configuration
    _max_attempts = max_attempts if max_attempts is not None else _retry_max_attempts()
    _base_delay = base_delay if base_delay is not None else _retry_base_delay()
    _max_delay = max_delay if max_delay is not None else _retry_max_delay()

    for attempt in range(1, _max_attempts + 1):
        try:
            return await func(*args, **kwargs) 
        except retry_exceptions as e: 
            if attempt == _max_attempts:
                logger.debug(
                    f"Retry exhausted after {_max_attempts} attempts. Last error: {e}"
                )
                raise

            backoff = _get_backoff_time(attempt - 1, _base_delay, _max_delay)
            logger.debug(
                f"Retry attempt {attempt}/{_max_attempts} failed with {e}. Retrying in {backoff:.2f}s..."
            )
            await asyncio.sleep(backoff)

    # This should never be reached due to the logic above
    assert False, "Unreachable code"
