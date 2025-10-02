"""
The `addons` sub-package contains optional extensions and logic augmentations for the core functionality of the project.

Modules placed in this package may include experimental features, alternative implementations, developer tools, or
enhancements that extend or customize core behavior. These components are not always critical for the main application,
but can be enabled or imported as needed for advanced use cases, internal tooling, or feature expansion.

Use this package to keep optional, modular, or feature-gated logic separate from the primary codebase while maintaining
discoverability and structure.
"""

from bittensor.extras import timelock
from bittensor.extras.subtensor_api import SubtensorApi

__all__ = [
    "timelock",
    "SubtensorApi",
]
