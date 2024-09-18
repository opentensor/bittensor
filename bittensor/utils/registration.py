# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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

    Args:
        func (function): Function with numpy Input/Output to be decorated.

    Returns:
        decorated (function): Decorated function.
    """

    @functools.wraps(func)
    def decorated(*args, **kwargs):
        if use_torch():
            # if argument is a Torch tensor, convert it to numpy
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
            # if return value is a numpy array, convert it to Torch tensor
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
