from bittensor.utils import easy_imports
import bittensor

import pytest


@pytest.mark.parametrize(
    "attr",
    [
        a
        for a in dir(easy_imports)
        if (not a.startswith("__") and a not in ["sys", "importlib"])  # we don't care about systemwide pkgs
    ],
)
def test_easy_imports(attr):
    assert getattr(bittensor, attr), attr
