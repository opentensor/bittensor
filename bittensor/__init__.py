import warnings

from .core.settings import __version__, version_split, DEFAULTS, DEFAULT_NETWORK
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
