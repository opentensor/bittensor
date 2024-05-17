import os
import signal
from substrateinterface import SubstrateInterface
import pytest
import subprocess
import logging
import shlex
import re
import time

logging.basicConfig(level=logging.INFO)


# Fixture for setting up and tearing down a localnet.sh chain between tests
@pytest.fixture(scope="function")
def local_chain():
    # Get the environment variable for the script path
    script_path = os.getenv("LOCALNET_SH_PATH")

    if not script_path:
        # Skip the test if the localhost.sh path is not set
        logging.warning("LOCALNET_SH_PATH env variable is not set, e2e test skipped.")
        pytest.skip("LOCALNET_SH_PATH environment variable is not set.")

    # Start new node process
    cmds = shlex.split(script_path)
    process = subprocess.Popen(
        cmds, stdout=subprocess.PIPE, text=True, preexec_fn=os.setsid
    )

    # Pattern match indicates node is compiled and ready
    pattern = re.compile(r"Successfully ran block step\.")

    def wait_for_node_start(process, pattern):
        for line in process.stdout:
            print(line.strip())
            if pattern.search(line):
                print("Node started!")
                break

    wait_for_node_start(process, pattern)

    # Run the test, passing in substrate interface
    yield SubstrateInterface(url="ws://127.0.0.1:9945")

    # Terminate the process group (includes all child processes)
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    # Give some time for the process to terminate
    time.sleep(1)

    # If the process is not terminated, send SIGKILL
    if process.poll() is None:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)

    # Ensure the process has terminated
    process.wait()
