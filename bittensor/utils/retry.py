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
    raw = os. environ.get("BT_RETRY_MAX_DELAY")
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
    return float(os.environ.get("BT_RETRY_BACKOFF_FACTOR", _RETRY_BACKOFF_FACTOR))


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
    """
    Synchronous retry wrapper around ``func`` with optional exponential backoff.

    When the environment variable ``BT_RETRY_ENABLED`` is set to a truthy value
    (e.g. ``"true"``, ``"1"``, ``"yes"``, ``"on"``), the call to ``func`` will be
    retried on failure up to ``max_attempts`` times using exponential backoff with
    jitter. When ``BT_RETRY_ENABLED`` is false or unset, ``func`` is executed
    exactly once and any exception it raises is propagated immediately.

    Parameters
    ----------
    func : Callable
        The callable to be executed and potentially retried.
    *args
        Positional arguments forwarded to ``func``.
    retry_exceptions : Exception type or tuple of Exception types, optional
        Exception type(s) that should trigger a retry. Any exception that is
        not an instance of these types is raised immediately without further
        retry attempts. Defaults to ``(OSError, TimeoutError)``.
    max_attempts : int, optional
        Maximum number of attempts (initial attempt + retries) that will be
        made before giving up. If ``None``, the value is taken from the
        ``BT_RETRY_MAX_ATTEMPTS`` environment variable (default ``3``).
    base_delay : float, optional
        Base delay, in seconds, used for exponential backoff before applying
        jitter. If ``None``, the value is taken from the
        ``BT_RETRY_BASE_DELAY`` environment variable (default ``1.0``).
    max_delay : float, optional
        Maximum delay, in seconds, between attempts. The computed backoff
        value will not exceed this. If ``None``, the value is taken from the
        ``BT_RETRY_MAX_DELAY`` environment variable (default ``60.0``).
    **kwargs
        Keyword arguments forwarded to ``func``.

    Returns
    -------
    Any
        The return value of ``func`` from the first successful attempt.

    Raises
    ------
    Exception
        Any exception raised by ``func`` when retries are disabled
        (``BT_RETRY_ENABLED`` is falsey), or when the exception type is not
        included in ``retry_exceptions``. When retries are enabled and all
        attempts fail with a ``retry_exceptions`` type, the last such
        exception is re-raised after the final attempt.

    Examples
    --------
    Basic usage with defaults::

        result = retry_call(do_network_request, url, timeout=5)

    Custom retry configuration::

        result = retry_call(
            do_network_request,
            url,
            timeout=5,
            max_attempts=5,
            base_delay=0.5,
            max_delay=10.0,
            retry_exceptions=(OSError, TimeoutError, ConnectionError),
        )

    To disable retries entirely, unset ``BT_RETRY_ENABLED`` or set it to a
    falsey value such as ``"false"`` or ``"0"``. In that case ``retry_call``
    simply calls ``func(*args, **kwargs)`` once and propagates any exception.
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
    Asynchronously call a function with optional retry logic and exponential backoff.

    When retries are **disabled** (``BT_RETRY_ENABLED`` is unset or set to a falsy
    value such as ``"false"`` or ``"0"``), this helper makes a single call:

        * ``await func(*args, **kwargs)`` is executed exactly once.
        * Any exception raised by ``func`` is propagated immediately.

    When retries are **enabled** (``BT_RETRY_ENABLED`` is set to a truthy value
    such as ``"true"`` or ``"1"``), the call will be retried on a configurable
    set of exceptions using exponential backoff with jitter:

        * The number of attempts defaults to ``BT_RETRY_MAX_ATTEMPTS`` (int, default 3)
          and can be overridden via ``max_attempts``.
        * The initial delay between attempts defaults to ``BT_RETRY_BASE_DELAY``
          (float seconds, default 1.0) and can be overridden via ``base_delay``.
        * The delay is multiplied by an internal backoff factor on each attempt and
          capped by ``BT_RETRY_MAX_DELAY`` (float seconds, default 60.0), which can
          be overridden via ``max_delay``.

    Parameters
    ----------
    func : Callable
        Asynchronous callable to execute. This must return an awaitable and is
        called as ``await func(*args, **kwargs)``.
    *args :
        Positional arguments forwarded to ``func`` on each attempt.
    retry_exceptions : Exception type or tuple of Exception types, optional
        Exception type(s) that trigger a retry when raised by ``func``.
        Any exception not matching ``retry_exceptions`` is propagated immediately
        without further retries. Defaults to ``(OSError, TimeoutError)``.
    max_attempts : int, optional
        Maximum number of attempts (including the first attempt). If ``None``,
        the value is taken from the ``BT_RETRY_MAX_ATTEMPTS`` environment
        variable (default 3 when unset).
    base_delay : float, optional
        Base delay in seconds before the first retry. If ``None``, the value is
        taken from the ``BT_RETRY_BASE_DELAY`` environment variable (default
        1.0 when unset).
    max_delay : float, optional
        Maximum delay in seconds between retries. If ``None``, the value is
        taken from the ``BT_RETRY_MAX_DELAY`` environment variable (default
        60.0 when unset).
    **kwargs :
        Keyword arguments forwarded to ``func`` on each attempt.

    Returns
    -------
    Any
        The result returned by ``func`` on the first successful attempt.

    Raises
    ------
    Exception
        Any exception raised by ``func`` when retries are disabled.
    retry_exceptions
        One of the configured ``retry_exceptions`` if all retry attempts are
        exhausted while retries are enabled.
    Exception
        Any exception not matching ``retry_exceptions`` is propagated
        immediately without retry.

    Examples
    --------
    Basic usage with environment-controlled configuration::

        async def fetch():
            ...

        result = await retry_async(fetch)

    Overriding retry configuration and the set of retryable exceptions::

        async def fetch_with_timeout():
            ...

        result = await retry_async(
            fetch_with_timeout,
            retry_exceptions=(OSError, TimeoutError, ConnectionError),
            max_attempts=5,
            base_delay=0.5,
            max_delay=10.0,
        )
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
