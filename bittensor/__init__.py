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

import os
import warnings

from .core.settings import __version__, version_split, DEFAULTS
from .utils.btlogging import logging
from .utils.deprecated import *


def __getattr__(name):
    if name == "version_split":
        warnings.warn(
            "version_split is deprecated and will be removed in future versions. Use __version__ instead.",
            DeprecationWarning,
        )
        return version_split
    raise AttributeError(f"module {__name__} has no attribute {name}")


# Logging helpers.
def trace(on: bool = True):
    logging.set_trace(on)


def debug(on: bool = True):
    logging.set_debug(on)


def __apply_nest_asyncio():
    """
    Apply nest_asyncio if the environment variable NEST_ASYNCIO is set to "1" or not set.
    If not set, warn the user that the default will change in the future.
    """
    nest_asyncio_env = os.getenv("NEST_ASYNCIO")

    if nest_asyncio_env == "1" or nest_asyncio_env is None:
        if nest_asyncio_env is None:
            warnings.warn(
                """NEST_ASYNCIO implicitly set to '1'. In the future, the default value will be '0'.
                If you use `nest_asyncio`, make sure to add it explicitly to your project dependencies,
                as it will be removed from `bittensor` package dependencies in the future.
                To silence this warning, explicitly set the environment variable, e.g. `export NEST_ASYNCIO=0`.""",
                DeprecationWarning,
            )
        # Install and apply nest asyncio to allow the async functions to run in a .ipynb
        import nest_asyncio

        nest_asyncio.apply()


__apply_nest_asyncio()
