import warnings

from .core.settings import __version__, version_split, DEFAULTS, DEFAULT_NETWORK
from .utils.btlogging import logging
from .utils.easy_imports import *
import scalecodec.types


def __getattr__(name):
    if name == "version_split":
        warnings.warn(
            "version_split is deprecated and will be removed in future versions. Use __version__ instead.",
            DeprecationWarning,
        )
        return version_split
    raise AttributeError(f"module {__name__} has no attribute {name}")


# the following patches the `scalecodec.types.Option.process` that allows for decoding certain extrinsics (specifically
# the ones used by crowdloan using Option<scale_info::227>. There is a PR up for this: https://github.com/JAMdotTech/py-scale-codec/pull/134
# and this patch will be removed when this is applied/released.


def patched_process(self):
    option_byte = self.get_next_bytes(1)

    if self.sub_type and option_byte != b"\x00":
        self.value_object = self.process_type(self.sub_type, metadata=self.metadata)
        return self.value_object.value

    return None


scalecodec.types.Option.process = patched_process
