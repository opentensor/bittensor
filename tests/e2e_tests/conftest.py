import os
import re
import shlex
import signal
import subprocess
import time
import threading

import pytest
from async_substrate_interface import SubstrateInterface
from bittensor.core.subtensor import Subtensor

from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.e2e_test_utils import (
    Templates,
    setup_wallet,
)


# Fixture for setting up and tearing down a localnet.sh chain between tests
@pytest.fixture(scope="function")
def local_chain(request):
    param = request.param if hasattr(request, "param") else None
    # Get the environment variable for the script path
    script_path = os.getenv("LOCALNET_SH_PATH")

    if not script_path:
        # Skip the test if the localhost.sh path is not set
        logging.warning("LOCALNET_SH_PATH env variable is not set, e2e test skipped.")
        pytest.skip("LOCALNET_SH_PATH environment variable is not set.")

    # Check if param is None, and handle it accordingly
    args = "" if param is None else f"{param}"

    # Compile commands to send to process
    cmds = shlex.split(f"{script_path} {args}")

    # Pattern match indicates node is compiled and ready
    pattern = re.compile(r"Imported #1")
    timestamp = int(time.time())

    def wait_for_node_start(process, pattern):
        while True:
            line = process.stdout.readline()
            if not line:
                break

            print(line.strip())
            # 10 min as timeout
            if int(time.time()) - timestamp > 10 * 60:
                print("Subtensor not started in time")
                raise TimeoutError
            if pattern.search(line):
                print("Node started!")
                break

        # Start a background reader after pattern is found
        # To prevent the buffer filling up
        def read_output():
            while True:
                line = process.stdout.readline()
                if not line:
                    break

        reader_thread = threading.Thread(target=read_output, daemon=True)
        reader_thread.start()

    with subprocess.Popen(
        cmds,
        start_new_session=True,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            wait_for_node_start(process, pattern)
        except TimeoutError:
            raise
        else:
            yield SubstrateInterface(url="ws://127.0.0.1:9944")
        finally:
            # Terminate the process group (includes all child processes)
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)

            try:
                process.wait(1)
            except subprocess.TimeoutExpired:
                # If the process is not terminated, send SIGKILL
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()


@pytest.fixture(scope="session")
def templates():
    with Templates() as templates:
        yield templates


@pytest.fixture
def subtensor(local_chain):
    return Subtensor(network="ws://localhost:9944")


@pytest.fixture
def alice_wallet():
    keypair, wallet = setup_wallet("//Alice")
    return wallet


@pytest.fixture
def bob_wallet():
    keypair, wallet = setup_wallet("//Bob")
    return wallet


@pytest.fixture
def dave_wallet():
    keypair, wallet = setup_wallet("//Dave")
    return wallet
