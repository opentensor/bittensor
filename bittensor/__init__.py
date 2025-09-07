import warnings

from .core.settings import __version__, version_split, DEFAULTS, DEFAULT_NETWORK
from .utils.btlogging import logging
from .utils.easy_imports import *
from .utils.runtime_browser import runtime_browser as runtime
from .utils.runtime_async_browser import runtime_async_browser as async_runtime


def __getattr__(name):
    if name == "version_split":
        warnings.warn(
            "version_split is deprecated and will be removed in future versions. Use __version__ instead.",
            DeprecationWarning,
        )
        return version_split
    raise AttributeError(f"module {__name__} has no attribute {name}")
