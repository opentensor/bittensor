from .core.settings import __version__, DEFAULTS, DEFAULT_NETWORK
from .utils.btlogging import logging
from .utils.async_substrate_interface_patch import apply_patch
# Apply the memory leak patch for AsyncSubstrateInterface
apply_patch()

from .utils.easy_imports import *
