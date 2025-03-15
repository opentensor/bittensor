import os
import re
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
from bittensor.utils.btlogging import logging

import pytest
from async_substrate_interface import SubstrateInterface

from bittensor.core.async_subtensor import AsyncSubtensor
from bittensor.core.subtensor import Subtensor
from tests.e2e_tests.utils.e2e_test_utils import (
    Templates,
    setup_wallet,
)


def wait_for_node_start(process, timestamp=None):
    """Waits for node to start in the docker."""
    while True:
        line = process.stdout.readline()
        if not line:
            break

        timestamp = timestamp or int(time.time())
        print(line.strip())
        # 10 min as timeout
        if int(time.time()) - timestamp > 20 * 30:
            print("Subtensor not started in time")
            raise TimeoutError

        pattern = re.compile(r"Imported #1")
        if pattern.search(line):
            print("Node started!")
            break

    # Start a background reader after pattern is found
    # To prevent the buffer filling up
    def read_output():
        while True:
            if not process.stdout.readline():
                break

    reader_thread = threading.Thread(target=read_output, daemon=True)
    reader_thread.start()


@pytest.fixture(scope="function")
def local_chain(request):
    """Determines whether to run the localnet.sh script in a subprocess or a Docker container."""
    args = request.param if hasattr(request, "param") else None
    params = "" if args is None else f"{args}"
    if shutil.which("docker"):
        yield from docker_runner(params)
        return

    if sys.platform.startswith("linux"):
        docker_commend = (
            "Install docker with command "
            "[blue]sudo apt-get update && sudo apt-get install docker.io -y[/blue]"
        )
    elif sys.platform == "darwin":
        docker_commend = "Install docker with command [blue]brew install docker[/blue]"
    else:
        docker_commend = "[blue]Unknown OS, install Docker manually: https://docs.docker.com/get-docker/[/blue]"

    logging.warning("Docker not found in the operating system!")
    logging.warning(docker_commend)
    logging.warning("Tests are run in legacy mode.")
    yield from legacy_runner(request)


def legacy_runner(params):
    """Runs the localnet.sh script in a subprocess and waits for it to start."""
    # Get the environment variable for the script path
    script_path = os.getenv("LOCALNET_SH_PATH")

    if not script_path:
        # Skip the test if the localhost.sh path is not set
        logging.warning("LOCALNET_SH_PATH env variable is not set, e2e test skipped.")
        pytest.skip("LOCALNET_SH_PATH environment variable is not set.")

    # Check if param is None, and handle it accordingly
    args = "" if params is None else f"{params}"

    # Compile commands to send to process
    cmds = shlex.split(f"{script_path} {args}")

    with subprocess.Popen(
        cmds,
        start_new_session=True,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            wait_for_node_start(process)
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


def docker_runner(params):
    """Starts a Docker container before tests and gracefully terminates it after."""

    container_name = f"test_local_chain_{str(time.time()).replace(".", "_")}"
    image_name = "ghcr.io/opentensor/subtensor-localnet:latest"

    # Command to start container
    cmds = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "-p",
        "9944:9944",
        "-p",
        "9945:9945",
        image_name,
        params,
    ]

    # Start container
    with subprocess.Popen(
        cmds,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            try:
                wait_for_node_start(process, int(time.time()))
            except TimeoutError:
                raise

            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={container_name}"],
                capture_output=True,
                text=True,
            )
            if not result.stdout.strip():
                raise RuntimeError("Docker container failed to start.")

            yield SubstrateInterface(url="ws://127.0.0.1:9944")

        finally:
            try:
                subprocess.run(["docker", "kill", container_name])
                process.wait()
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)


@pytest.fixture(scope="session")
def templates():
    with Templates() as templates:
        yield templates


@pytest.fixture
def subtensor(local_chain):
    return Subtensor(network="ws://localhost:9944")


@pytest.fixture
def async_subtensor(local_chain):
    return AsyncSubtensor(network="ws://localhost:9944")


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
