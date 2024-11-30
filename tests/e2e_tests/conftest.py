import os
import re
import shlex
import signal
import subprocess
import time

import pytest
from substrateinterface import SubstrateInterface

from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.e2e_test_utils import (
    clone_or_update_templates,
    install_templates,
    template_path,
    uninstall_templates,
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

    # Start new node process
    process = subprocess.Popen(
        cmds, stdout=subprocess.PIPE, text=True, preexec_fn=os.setsid
    )

    # Pattern match indicates node is compiled and ready
    pattern = re.compile(r"Imported #1")

    # install neuron templates
    logging.info("downloading and installing neuron templates from github")
    # commit with subnet-template-repo changes for rust wallet
    templates_dir = clone_or_update_templates()
    install_templates(templates_dir)

    timestamp = int(time.time())

    def wait_for_node_start(process, pattern):
        for line in process.stdout:
            print(line.strip())
            # 10 min as timeout
            if int(time.time()) - timestamp > 10 * 60:
                print("Subtensor not started in time")
                break
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

    # uninstall templates
    logging.info("uninstalling neuron templates")
    uninstall_templates(template_path)