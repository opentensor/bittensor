from .core.settings import __version__, DEFAULTS, DEFAULT_NETWORK
from .utils.btlogging import logging
from .utils.async_substrate_interface_patch import apply_patch
# Apply the memory leak patch for AsyncSubstrateInterface *before* importing anything
# that may create AsyncSubstrateInterface instances. In particular, easy_imports
# pulls in AsyncSubtensor, which uses AsyncSubstrateInterface, so it must be
# imported only after apply_patch() has been called. Do not reorder these imports.
apply_patch()

from .utils.easy_imports import *
