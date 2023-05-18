# The MIT License (MIT)
# Copyright © 2022-2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from substrateinterface import SubstrateInterface, Keypair
from substrateinterface.exceptions import SubstrateRequestException
from scalecodec import GenericCall
import psutil
import subprocess
from sys import platform
import bittensor
import time
import os
from typing import Optional, Tuple, Dict, Union, TypedDict
import requests
from urllib3.exceptions import LocationValueError
import numpy as np
from filelock import Timeout, FileLock
from retry import retry

from . import subtensor_impl

__type_registery__ = {
    "runtime_id": 2,
    "types": {
        "Balance": "u64",
        "NeuronMetadataOf": {
            "type": "struct",
            "type_mapping": [
                ["version", "u32"],
                ["ip", "u128"],
                ["port", "u16"],
                ["ip_type", "u8"],
                ["uid", "u32"],
                ["modality", "u8"],
                ["hotkey", "AccountId"],
                ["coldkey", "AccountId"],
                ["active", "bool"],
                ["last_update", "u64"],
                ["validator_permit", "bool"],
                ["stake", "u64"],
                ["rank", "u16"],
                ["trust", "u16"],
                ["consensus", "u16"],
                ["validator_trust", "u16"],
                ["incentive", "u16"],
                ["dividends", "u16"],
                ["emission", "u64"],
                ["bonds", "Vec<(u16, u16)>"],
                ["weights", "Vec<(u16, u16)>"],
            ],
        },
    },
}

GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME = "mock-node-subtensor"


class mock_subtensor:
    r"""Returns a subtensor connection interface to a mocked subtensor process running in the background.
    Optionall creates the background process if it does not exist.
    """

    @classmethod
    def mock(cls):
        _owned_mock_subtensor_process = None

        if not cls.global_mock_process_is_running():
            # Remove any old chain db
            if os.path.exists(f"{bittensor.__mock_chain_db__}_{os.getpid()}"):
                # Name mock chain db using pid to avoid conflicts while multiple processes are running.
                os.system(f"rm -rf {bittensor.__mock_chain_db__}_{os.getpid()}")
            _owned_mock_subtensor_process = cls.create_global_mock_process(os.getpid())

        endpoint: str = bittensor.__mock_entrypoint__
        url_root, ws_port = endpoint.split(":")

        if _owned_mock_subtensor_process is None:
            print("Mock subtensor already running.")
            # THen ws_port is set by the global process.
            ws_port = None

            # Wait for other process to finish setting up the mock subtensor.
            timeout = 35  # seconds
            time_elapsed = 0
            time_start = time.time()
            # Try to get ws_port
            while time_elapsed < timeout:
                try:
                    ws_port = cls.get_global_ws_port()
                    if ws_port is None:
                        continue

                    # Try to connect to the mock subtensor process.
                    errored = cls.try_connect_to_mock(ws_port, timeout=2)
                    connected = not errored

                    if connected:
                        break
                except FileNotFoundError:
                    time.sleep(0.1)
                    time_elapsed = time.time() - time_start
            else:
                raise TimeoutError(f"Could not get ws_port from file")

        substrate = SubstrateInterface(
            ss58_format=bittensor.__ss58_format__,
            type_registry_preset="substrate-node-template",
            type_registry=__type_registery__,
            url=f"ws://{url_root}:{ws_port}",
            use_remote_preset=True,
        )
        subtensor = Mock_Subtensor(
            substrate=substrate,
            network="mock",
            chain_endpoint=f"{url_root}:{ws_port}",
            # Is mocked, optionally has owned process for ref counting.
            _is_mocked=True,
            _owned_mock_subtensor_process=_owned_mock_subtensor_process,
        )
        return subtensor

    @staticmethod
    def global_mock_process_is_running() -> bool:
        r"""Check if the global mocked subtensor process is running on the machine.
        This means only one mock subtensor will run for ALL processes.
        """
        for p in psutil.process_iter():
            if (
                p.status() != psutil.STATUS_ZOMBIE
                and p.status() != psutil.STATUS_DEAD
                and p.name() == GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME
            ):
                print(
                    f"Found process with name {p.name()}, parent {p.parent().pid} status {p.status()} and pid {p.pid}"
                )
                return True
        return False

    @classmethod
    def kill_global_mock_process(cls):
        r"""Kills the global mocked subtensor process even if not owned."""
        for p in psutil.process_iter():
            if (
                p.name() == GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME
                and p.parent().pid == os.getpid()
            ):
                p.terminate()
                p.kill()
                cls.destroy_lock()  # Remove lock file.
        time.sleep(2)  # Buffer to ensure the processes actually die

    @staticmethod
    def try_connect_to_mock(ws_port: int, timeout: Optional[int] = None) -> bool:
        r"""Tries to connect to the mock subtensor process.
        Returns False if the connection fails.
        """
        time_elapsed = 0
        time_start = time.time()

        errored: bool = True
        while errored and (timeout is None or time_elapsed < timeout):
            errored = False
            try:
                _ = requests.get("http://localhost:{}".format(ws_port))
            except (requests.exceptions.ConnectionError, LocationValueError) as e:
                errored = True
                time.sleep(0.3)  # Wait for the process to start.
                time_elapsed = time.time() - time_start

        return errored

    @staticmethod
    def _get_ws_port_from_file(filename: str) -> int:
        r"""Gets the ws port from `filename`."""
        with open(filename, "r") as f:
            ws_port = int(f.read())
        return ws_port

    @staticmethod
    def _write_ws_port_to_file(ws_port: int, filename: str) -> None:
        r"""Writes the global ws port to `filename`."""
        with open(filename, "w") as f:
            f.write(str(ws_port))

    @classmethod
    def get_global_ws_port(cls) -> Optional[int]:
        r"""Gets the ws port from the global mock subtensor process."""
        filename = "./tests/mock_subtensor/ws_port.txt"
        if os.path.exists(filename):
            return cls._get_ws_port_from_file(filename)
        else:
            return None

    @classmethod
    def save_global_ws_port(cls, ws_port: int) -> None:
        r""" """
        root_path = "./tests/mock_subtensor"
        filename = f"{root_path}/ws_port.txt"
        if not os.path.exists(root_path):
            os.makedirs(root_path, exist_ok=True)

        cls._write_ws_port_to_file(ws_port, filename)

    _lock_filename = "./tests/mock_subtensor/lock.lock"

    @classmethod
    def make_lock(cls) -> FileLock:
        r"""Creates a file lock."""
        lock_file = cls._lock_filename
        lock = FileLock(lock_file, timeout=1)
        return lock

    @classmethod
    def destroy_lock(cls) -> None:
        r"""Destroys the file lock."""
        lock_file = cls._lock_filename
        if os.path.exists(lock_file):
            os.remove(lock_file)

    @classmethod
    def create_global_mock_process(
        cls, pid: int
    ) -> Optional["subprocess.Popen[bytes]"]:
        r"""Creates a global mocked subtensor process running in the backgroun with name GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME.
        Returns None if the process is already running.
        Raises:
            RuntimeError: If the process cannot be created.
        """
        try:
            # acquire file lock
            lock = cls.make_lock()
            lock.acquire(timeout=1)  # Wait for 1 seconds to acquire the lock.

            operating_system = "OSX" if platform == "darwin" else "Linux"
            path_root = "./tests/mock_subtensor"
            path = "{}/bin/{}/{}".format(
                path_root, operating_system, GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME
            )
            path_to_spec = "{}/specs/local_raw.json".format(path_root)

            ws_port = int(bittensor.__mock_entrypoint__.split(":")[1])
            print(f"MockSub ws_port: {ws_port}")

            command_args = (
                [path]
                + f"--chain {path_to_spec} --base-path {bittensor.__mock_chain_db__}_{pid} --execution native --ws-max-connections 1000 --no-mdns --rpc-cors all".split(
                    " "
                )
                + f"--port {int(bittensor.get_random_unused_port())} --rpc-port {int(bittensor.get_random_unused_port())} --ws-port {ws_port}".split(
                    " "
                )
                + "--validator --alice".split(" ")
            )

            print("Starting subtensor process with command: {}".format(command_args))

            _mock_subtensor_process = subprocess.Popen(
                command_args,
                close_fds=True,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # # Write the ws port to a file
            # cls.save_global_ws_port(ws_port)

            # Wait for the process to start. Check for errors.
            try:
                # Timeout is okay.
                error_code = _mock_subtensor_process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                error_code = None

            if error_code is not None:
                # Get the error message.
                error_message = _mock_subtensor_process.stderr.read().decode("utf-8")
                raise RuntimeError(
                    "Failed to start mocked subtensor process: {}".format(error_code),
                    error_message,
                )

            print(
                "Starting subtensor process with pid {} and name {}".format(
                    _mock_subtensor_process.pid, GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME
                )
            )

            # Wait for the process to start.
            errored = cls.try_connect_to_mock(ws_port)

            return _mock_subtensor_process
        except Timeout:
            return None  # Another process has the lock.
        except Exception as e:
            raise RuntimeError("Failed to start mocked subtensor process: {}".format(e))

class DecodedGenericExtrinsic(TypedDict):
    nonce: int

class TxError(Exception):
    pass

class PriorityTooLowError(TxError):
    pass

class InvalidNonceError(TxError):
    pass


class Mock_Subtensor(subtensor_impl.Subtensor):
    """
    Handles interactions with the subtensor chain.
    """

    sudo_keypair: Keypair = Keypair.create_from_uri(
        "//Alice"
    )  # Alice is the sudo keypair for the mock chain.

    def __init__(
        self,
        _is_mocked: bool,
        _owned_mock_subtensor_process: object,
        **kwargs,
    ):
        r"""Initializes a subtensor chain interface.
        Args:
            _owned_mock_subtensor_process (Used for testing):
                a subprocess where a mock chain is running.
        """
        super().__init__(**kwargs)
        # Exclusively used to mock a connection to our chain.
        self._owned_mock_subtensor_process = _owned_mock_subtensor_process
        self._is_mocked = _is_mocked

        print("---- MOCKED SUBTENSOR INITIALIZED ----")

    def __str__(self) -> str:
        if self._is_mocked == True and self._owned_mock_subtensor_process != None:
            # Mocked and owns background process.
            return "MockSubtensor({}, PID:{})".format(
                self.chain_endpoint, self._owned_mock_subtensor_process.pid
            )
        else:
            # Mocked but does not own process.
            return "MockSubtensor({})".format(self.chain_endpoint)

    def __exit__(self):
        pass

    def optionally_kill_owned_mock_instance(self):
        r"""If this subtensor instance owns the mock process, it kills the process."""
        if self._owned_mock_subtensor_process != None:
            try:
                self._owned_mock_subtensor_process.terminate()
                self._owned_mock_subtensor_process.kill()
                os.system("kill %i" % self._owned_mock_subtensor_process.pid)
                mock_subtensor.destroy_lock()  # Remove lock file.
                time.sleep(2)  # Buffer to ensure the processes actually die
            except Exception as e:
                print(f"failed to kill owned mock instance: {e}")
                # Occasionally
                pass

    def wrap_sudo(self, call: GenericCall) -> GenericCall:
        r"""Wraps a call in a sudo call."""
        return self.substrate.compose_call(
            call_module="Sudo", call_function="sudo", call_params={"call": call.value}
        )

    def sudo_force_set_balance(
        self,
        ss58_address: str,
        balance: Union["bittensor.Balance", int, float],
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        nonce: Optional[int] = None,
    ) -> Tuple[bool, Optional[str], int]:
        r"""Sets the balance of an account using the sudo key."""
        if isinstance(balance, bittensor.Balance):
            balance = balance.rao
        elif isinstance(balance, float):
            balance = int(balance * bittensor.utils.RAOPERTAO)
        elif isinstance(balance, int):
            pass
        else:
            raise ValueError("Invalid type for balance: {}".format(type(balance)))

        @retry(exceptions=(PriorityTooLowError, InvalidNonceError), delay=2, tries=3, backoff=2, max_delay=4)
        def make_call() -> Tuple[bool, Optional[str], int]:
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module="Balances",
                    call_function="set_balance",
                    call_params={
                        "who": ss58_address,
                        "new_free": balance,
                        "new_reserved": 0,
                    },
                )

                wrapped_call = self.wrap_sudo(call)

                try:
                    return self._submit_call(
                        substrate,
                        wrapped_call,
                        nonce,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
                    )
                except InvalidNonceError:
                    return self._submit_call(
                        substrate,
                        wrapped_call,
                        nonce = None,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
                    )

        return make_call()

    def sudo_set_serving_rate_limit(
        self,
        netuid: int,
        serving_rate_limit: int,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str], int]:
        r"""Sets the serving rate limit of the subnet in the mock chain using the sudo key."""
        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module="SubtensorModule",
                call_function="sudo_set_serving_rate_limit",
                call_params={
                    "netuid": netuid,
                    "serving_rate_limit": serving_rate_limit,
                },
            )

            wrapped_call = self.wrap_sudo(call)

            return self._submit_call(
                substrate,
                wrapped_call,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

    def sudo_set_tx_rate_limit(
        self,
        tx_rate_limit: int,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str], int]:
        r"""Sets the tx rate limit of the subnet in the mock chain using the sudo key."""
        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module="SubtensorModule",
                call_function="sudo_set_tx_rate_limit",
                call_params={"tx_rate_limit": tx_rate_limit},
            )

            wrapped_call = self.wrap_sudo(call)

            return self._submit_call(
                substrate,
                wrapped_call,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

    def get_tx_rate_limit(self) -> int:
        r"""Gets the tx rate limit of the subnet in the mock chain."""
        result = self.query_subtensor("TxRateLimit")

        return result.value

    def sudo_set_difficulty(
        self,
        netuid: int,
        difficulty: int,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str], int]:
        r"""Sets the difficulty of the mock chain using the sudo key."""
        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module="SubtensorModule",
                call_function="sudo_set_difficulty",
                call_params={"netuid": netuid, "difficulty": difficulty},
            )

            wrapped_call = self.wrap_sudo(call)

            return self._submit_call(
                substrate,
                wrapped_call,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

    def sudo_set_max_difficulty(
        self,
        netuid: int,
        max_difficulty: int,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str], int]:
        r"""Sets the max difficulty of the mock chain using the sudo key."""
        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module="SubtensorModule",
                call_function="sudo_set_max_difficulty",
                call_params={"netuid": netuid, "max_difficulty": max_difficulty},
            )

            wrapped_call = self.wrap_sudo(call)

            return self._submit_call(
                substrate,
                wrapped_call,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

    def sudo_set_min_difficulty(
        self,
        netuid: int,
        min_difficulty: int,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str], int]:
        r"""Sets the min difficulty of the mock chain using the sudo key."""
        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module="SubtensorModule",
                call_function="sudo_set_min_difficulty",
                call_params={"netuid": netuid, "min_difficulty": min_difficulty},
            )

            wrapped_call = self.wrap_sudo(call)

            return self._submit_call(
                substrate,
                wrapped_call,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

    def sudo_add_network(
        self,
        netuid: int,
        tempo: int = 0,
        modality: int = 0,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str], int]:
        r"""Adds a network to the mock chain using the sudo key."""
        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module="SubtensorModule",
                call_function="sudo_add_network",
                call_params={"netuid": netuid, "tempo": tempo, "modality": modality},
            )

            wrapped_call = self.wrap_sudo(call)

            return self._submit_call(
                substrate,
                wrapped_call,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
    
    def _submit_call(
        self,
        substrate: SubstrateInterface,
        wrapped_call: GenericCall,
        nonce: Optional[int] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str], int]:
        
        try: 
            extrinsic = substrate.create_signed_extrinsic(
                call=wrapped_call, keypair=self.sudo_keypair, nonce=nonce
            )

            decoded_extrinsic: DecodedGenericExtrinsic = extrinsic.decode()
            used_nonce = decoded_extrinsic["nonce"]
            
            response = substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if not wait_for_finalization:
                return True, None, used_nonce

            try:
                response.process_events()
            except Exception as e:
                print(f"Failed to process events: {e}")

            if response.is_success:
                return True, None, used_nonce
            else:
                return False, response.error_message, used_nonce
        except SubstrateRequestException as e:
            code = None
            if hasattr(e, 'args') and len(e.args) > 0:
                args = e.args[0]
                code = args['code']
            
            if code == 1010:
                raise InvalidNonceError()
            elif code == 1014:
                raise PriorityTooLowError()
            else:
                raise e
                


    def sudo_register(
        self,
        netuid: int,
        hotkey: str,
        coldkey: str,
        stake: int = 0,
        balance: int = 0,
        nonce: Optional[int] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str], int]:
        r"""Registers a neuron to the subnet using sudo.
        Returns:
            bool: True if the extrinsic was successful, False otherwise.
            Optional[str]: The error message if the extrinsic failed, None otherwise.
            int: The nonce used for the extrinsic.
        """
        @retry(exceptions=(PriorityTooLowError, InvalidNonceError), delay=2, tries=3, backoff=2, max_delay=4)
        def make_call(nonce: Optional[int]) -> Tuple[bool, Optional[str], int]:
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="sudo_register",
                    call_params={
                        "netuid": netuid,
                        "hotkey": hotkey,
                        "coldkey": coldkey,
                        "stake": stake,
                        "balance": balance,
                    },
                )

                wrapped_call = self.wrap_sudo(call)

                try:
                    return self._submit_call(
                        substrate,
                        wrapped_call,
                        nonce,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
                    )
                except InvalidNonceError:
                    return self._submit_call(
                        substrate,
                        wrapped_call,
                        nonce = None,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
                    )
        
        return make_call(nonce)
