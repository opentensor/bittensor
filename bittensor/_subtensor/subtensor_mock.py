# The MIT License (MIT)
# Copyright © 2022 Opentensor Foundation

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

from substrateinterface import SubstrateInterface
import psutil
import subprocess
from sys import platform   
import bittensor
import time
import os

from . import subtensor_impl
from bittensor.utils.test_utils import get_random_unused_port

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
                ["active", "u32"],
                ["last_update", "u64"],
                ["validator_permit", "bool"],
                ["priority", "u64"],
                ["stake", "u64"],
                ["rank", "u16"],
                ["trust", "u16"],
                ["consensus", "u16"],
                ["validator_trust", "u16"],
                ["weight_consensus", "u16"],
                ["incentive", "u16"],
                ["dividends", "u16"],
                ["emission", "u64"],
                ["bonds", "Vec<(u32, u64)>"],
                ["weights", "Vec<(u32, u32)>"]
            ]
        }
    }
}

GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME = "node-subtensor"

class mock_subtensor():
    r""" Returns a subtensor connection interface to a mocked subtensor process running in the background.
        Optionall creates the background process if it does not exist.
    """

    @classmethod
    def mock(cls):

        if not cls.global_mock_process_is_running():
            _owned_mock_subtensor_process = cls.create_global_mock_process()
            time.sleep(3)
        else:
            _owned_mock_subtensor_process = None
            print ('Mock subtensor already running.')

        endpoint = bittensor.__mock_entrypoint__
        port = int(endpoint.split(':')[1])
        substrate = SubstrateInterface(
            ss58_format = bittensor.__ss58_format__,
            type_registry_preset='substrate-node-template',
            type_registry = __type_registery__,
            url = "ws://{}".format('localhost:{}'.format(port)),
            use_remote_preset=True
        )
        subtensor = Mock_Subtensor( 
            substrate = substrate,
            network = 'mock',
            chain_endpoint = 'localhost:{}'.format(port),

            # Is mocked, optionally has owned process for ref counting.
            _is_mocked = True,
            _owned_mock_subtensor_process = _owned_mock_subtensor_process
        )
        return subtensor

    @classmethod
    def global_mock_process_is_running(cle) -> bool:
        r""" If subtensor is running a mock process this kills the mock.
        """
        for p in psutil.process_iter():
            if p.name() == GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME and p.parent().pid == os.getpid() and p.status() != psutil.STATUS_ZOMBIE and p.status() != psutil.STATUS_DEAD:
                print(f"Found process with name {p.name()}, parent {p.parent().pid} status {p.status()} and pid {p.pid}")
                return True
        return False

    @classmethod
    def kill_global_mock_process(self):
        r""" Kills the global mocked subtensor process even if not owned.
        """
        for p in psutil.process_iter():
            if p.name() == GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME and p.parent().pid == os.getpid() :
                p.terminate()
                p.kill()
        time.sleep(2) # Buffer to ensure the processes actually die

    @classmethod
    def create_global_mock_process(self):
        r""" Creates a global mocked subtensor process running in the backgroun with name GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME.
        """
        try:
            operating_system = "OSX" if platform == "darwin" else "Linux"
            path = "./bin/chain/{}/node-subtensor".format(operating_system)
            ws_port = int(bittensor.__mock_entrypoint__.split(':')[1])
            print(ws_port)
            print(os.getpid())
            baseport = get_random_unused_port()
            rpc = get_random_unused_port()
            subprocess.Popen([path, 'purge-chain', '--dev', '-y'], close_fds=True, shell=False)    
            _mock_subtensor_process = subprocess.Popen( [path, '--dev', '--port', str(baseport), '--ws-port', str(ws_port), '--rpc-port', str(rpc), '--tmp'], close_fds=True, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print ('Starting subtensor process with pid {} and name {}'.format(_mock_subtensor_process.pid, GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME))
            return _mock_subtensor_process
        except Exception as e:
            raise RuntimeError( 'Failed to start mocked subtensor process: {}'.format(e) )


class Mock_Subtensor(subtensor_impl.Subtensor):
    """
    Handles interactions with the subtensor chain.
    """
    def __init__( 
        self, 
        _is_mocked: bool,
        _owned_mock_subtensor_process: object,
        **kwargs,
    ):
        r""" Initializes a subtensor chain interface.
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
            return "MockSubtensor({}, PID:{})".format( self.chain_endpoint, self._owned_mock_subtensor_process.pid)
        else:
            # Mocked but does not own process.
            return "MockSubtensor({})".format( self.chain_endpoint)

    def __del__(self):
        self.optionally_kill_owned_mock_instance()
    
    def __exit__(self):
        self.__del__()

    def optionally_kill_owned_mock_instance(self):
        r""" If this subtensor instance owns the mock process, it kills the process.
        """
        if self._owned_mock_subtensor_process != None:
            try:
                self._owned_mock_subtensor_process.terminate()
                self._owned_mock_subtensor_process.kill()
                os.system("kill %i" % self._owned_mock_subtensor_process.pid)
                time.sleep(2) # Buffer to ensure the processes actually die
            except Exception as e:
                print(f"failed to kill owned mock instance: {e}")
                # Occasionally 
                pass
