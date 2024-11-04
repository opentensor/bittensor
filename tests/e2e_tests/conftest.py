import os
import re
import shlex
import signal
import subprocess
import time

import pytest
from substrateinterface import SubstrateInterface

from bittensor import logging
from tests.e2e_tests.utils.e2e_test_utils import (
    clone_or_update_templates,
    install_templates,
    template_path,
    uninstall_templates,
)


# Fixture for setting up and tearing down a localnet.sh chain between tests
@pytest.fixture(scope="function")
def local_chain(request):
    

    # Run the test, passing in substrate interface
    yield SubstrateInterface(url="ws://127.0.0.1:9945")

    