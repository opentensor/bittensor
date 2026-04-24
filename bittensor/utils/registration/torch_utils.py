"""Torch compatibility utilities for Bittensor."""

import functools
import os
from typing import TYPE_CHECKING

import numpy

from bittensor.utils.btlogging import logging


def use_torch() -> bool:
    """Force the use of torch over numpy for certain operations."""
    return True if os.getenv("USE_TORCH") == "1" else False


def legacy_torch_api_compat(func):
    """
    Convert function operating on numpy Input&Output to legacy torch Input&Output API if `use_torch()` is True.

    Parameters:
        func: Function with numpy Input/Output to be decorated.

    Returns:
        decorated: Decorated function.
    """

    @functools.wraps(func)
    def decorated(*args, **kwargs):
        if use_torch():
            args = [
                arg.cpu().numpy() if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ]
            kwargs = {
                key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
            }
        ret = func(*args, **kwargs)
        if use_torch():
            if isinstance(ret, numpy.ndarray):
                ret = torch.from_numpy(ret)
        return ret

    return decorated


@functools.cache
def _get_real_torch():
    try:
        import torch as _real_torch
    except ImportError:
        _real_torch = None
    return _real_torch


def log_no_torch_error():
    logging.error(
        "This command requires torch. You can install torch for bittensor"
        ' with `pip install bittensor[torch]` or `pip install ".[torch]"`'
        " if installing from source, and then run the command with USE_TORCH=1 {command}"
    )


class LazyLoadedTorch:
    """A lazy-loading proxy for the torch module."""

    def __bool__(self):
        return bool(_get_real_torch())

    def __getattr__(self, name):
        if real_torch := _get_real_torch():
            return getattr(real_torch, name)
        else:
            log_no_torch_error()
            raise ImportError("torch not installed")


if TYPE_CHECKING:
    import torch
else:
    torch = LazyLoadedTorch()
