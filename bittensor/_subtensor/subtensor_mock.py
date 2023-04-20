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

from substrateinterface import SubstrateInterface, Keypair
from scalecodec import GenericCall
import psutil
import subprocess
from sys import platform
import bittensor
import time
import os
from typing import Optional, Tuple, Dict, Union
import requests

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
                ["weights", "Vec<(u16, u16)>"]
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
            # Remove any old chain db
            if os.path.exists(f'{bittensor.__mock_chain_db__}_{os.getpid()}'):
                # Name mock chain db using pid to avoid conflicts while multiple processes are running.
                os.system(f'rm -rf {bittensor.__mock_chain_db__}_{os.getpid()}')
            _owned_mock_subtensor_process = cls.create_global_mock_process(os.getpid())
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
    def global_mock_process_is_running(cls) -> bool:
        r""" Check if the global mocked subtensor process is running under a process with the same name as this one.
        """
        this_process = psutil.Process(os.getpid())
        for p in psutil.process_iter():
            if p.name() == GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME and p.status() != psutil.STATUS_ZOMBIE and p.status() != psutil.STATUS_DEAD:
                if p.parent().name == this_process.name:
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
    def create_global_mock_process(self, pid: int) -> 'subprocess.Popen[bytes]':
        r""" Creates a global mocked subtensor process running in the backgroun with name GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME.
        """
        try:
            operating_system = "OSX" if platform == "darwin" else "Linux"
            path_root = "./tests/mock_subtensor"
            path = "{}/bin/{}/{}".format(path_root, operating_system, GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME)
            path_to_spec = "{}/specs/local_raw.json".format(path_root)
            
            ws_port = int(bittensor.__mock_entrypoint__.split(':')[1])
            print(f'MockSub ws_port: {ws_port}')
            
            command_args = [ path ] + f'--chain {path_to_spec} --base-path {bittensor.__mock_chain_db__}_{pid} --execution native --ws-max-connections 1000 --no-mdns --rpc-cors all'.split(' ') + \
                f'--port {int(bittensor.get_random_unused_port())} --rpc-port {int(bittensor.get_random_unused_port())} --ws-port {ws_port}'.split(' ') + \
                '--validator --alice'.split(' ')
            
            print ('Starting subtensor process with command: {}'.format(command_args))
            
            _mock_subtensor_process = subprocess.Popen(
                command_args,
                close_fds=True, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
            
            # Wait for the process to start. Check for errors.
            try:
                # Timeout is okay.
                error_code = _mock_subtensor_process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                error_code = None
            
            if error_code is not None:
                # Get the error message.
                error_message = _mock_subtensor_process.stderr.read().decode('utf-8')
                raise RuntimeError( 'Failed to start mocked subtensor process: {}'.format(error_code), error_message )

            print ('Starting subtensor process with pid {} and name {}'.format(_mock_subtensor_process.pid, GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME))

            errored: bool = True
            while errored:
                errored = False
                try:
                    _ = requests.get('http://localhost:{}'.format(ws_port))
                except requests.exceptions.ConnectionError as e:
                    errored = True
                    time.sleep(0.5) # Wait for the process to start.
            
            return _mock_subtensor_process
        except Exception as e:
            raise RuntimeError( 'Failed to start mocked subtensor process: {}'.format(e) )


class Mock_Subtensor(subtensor_impl.Subtensor):
    """
    Handles interactions with the subtensor chain.
    """
    sudo_keypair: Keypair = Keypair.create_from_uri('//Alice') # Alice is the sudo keypair for the mock chain.
    
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
        pass
    
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

    def wrap_sudo(self, call: GenericCall) -> GenericCall:
        r""" Wraps a call in a sudo call.
        """
        return self.substrate.compose_call(
            call_module='Sudo',
            call_function='sudo',
            call_params = {
                'call': call.value
            }
        )

    def sudo_force_set_balance(self, ss58_address: str, balance: Union['bittensor.Balance', int, float], ) -> Tuple[bool, Optional[str]]:
        r""" Sets the balance of an account using the sudo key.
        """
        if isinstance(balance, bittensor.Balance):
            balance = balance.rao
        elif isinstance(balance, float):
            balance = int(balance * bittensor.utils.RAOPERTAO)
        elif isinstance(balance, int):
            pass
        else:
            raise ValueError('Invalid type for balance: {}'.format(type(balance)))
        
        with self.substrate as substrate:
            call = substrate.compose_call(
                    call_module='Balances',
                    call_function='set_balance',
                    call_params = {
                        'who': ss58_address,
                        'new_free': balance,
                        'new_reserved': 0
                    }
                )

            wrapped_call = self.wrap_sudo(call)

            extrinsic = substrate.create_signed_extrinsic( call = wrapped_call, keypair = self.sudo_keypair )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = True, wait_for_finalization = True )

            response.process_events()
            if response.is_success:
                return True, None
            else:
                return False, response.error_message
            
    def sudo_set_tx_rate_limit(self, netuid: int, tx_rate_limit: int, wait_for_inclusion: bool = True, wait_for_finalization: bool = True ) -> Tuple[bool, Optional[str]]:
        r""" Sets the tx rate limit of the subnet in the mock chain using the sudo key.
        """
        with self.substrate as substrate:
            call = substrate.compose_call(
                    call_module='SubtensorModule',
                    call_function='sudo_set_tx_rate_limit',
                    call_params = {
                        'netuid': netuid,
                        'tx_rate_limit': tx_rate_limit
                    }
                )

            wrapped_call = self.wrap_sudo(call)

            extrinsic = substrate.create_signed_extrinsic( call = wrapped_call, keypair = self.sudo_keypair )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )

            if not wait_for_finalization:
                return True, None
            
            response.process_events()
            if response.is_success:
                return True, None
            else:
                return False, response.error_message
        
    def sudo_set_difficulty(self, netuid: int, difficulty: int, wait_for_inclusion: bool = True, wait_for_finalization: bool = True ) -> Tuple[bool, Optional[str]]:
        r""" Sets the difficulty of the mock chain using the sudo key.
        """
        with self.substrate as substrate:
            call = substrate.compose_call(
                    call_module='SubtensorModule',
                    call_function='sudo_set_difficulty',
                    call_params = {
                        'netuid': netuid,
                        'difficulty': difficulty
                    }
                )

            wrapped_call = self.wrap_sudo(call)

            extrinsic = substrate.create_signed_extrinsic( call = wrapped_call, keypair = self.sudo_keypair )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )

            if not wait_for_finalization:
                return True, None
            
            response.process_events()
            if response.is_success:
                return True, None
            else:
                return False, response.error_message
            
    def sudo_set_max_difficulty(self, netuid: int, max_difficulty: int, wait_for_inclusion: bool = True, wait_for_finalization: bool = True ) -> Tuple[bool, Optional[str]]:
        r""" Sets the max difficulty of the mock chain using the sudo key.
        """
        with self.substrate as substrate:
            call = substrate.compose_call(
                    call_module='SubtensorModule',
                    call_function='sudo_set_max_difficulty',
                    call_params = {
                        'netuid': netuid,
                        'max_difficulty': max_difficulty
                    }
                )

            wrapped_call = self.wrap_sudo(call)

            extrinsic = substrate.create_signed_extrinsic( call = wrapped_call, keypair = self.sudo_keypair )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )

            if not wait_for_finalization:
                return True, None
            
            response.process_events()
            if response.is_success:
                return True, None
            else:
                return False, response.error_message

    def sudo_set_min_difficulty(self, netuid: int, min_difficulty: int, wait_for_inclusion: bool = True, wait_for_finalization: bool = True ) -> Tuple[bool, Optional[str]]:
        r""" Sets the min difficulty of the mock chain using the sudo key.
        """
        with self.substrate as substrate:
            call = substrate.compose_call(
                    call_module='SubtensorModule',
                    call_function='sudo_set_min_difficulty',
                    call_params = {
                        'netuid': netuid,
                        'min_difficulty': min_difficulty
                    }
                )

            wrapped_call = self.wrap_sudo(call)

            extrinsic = substrate.create_signed_extrinsic( call = wrapped_call, keypair = self.sudo_keypair )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )

            if not wait_for_finalization:
                return True, None
            
            response.process_events()
            if response.is_success:
                return True, None
            else:
                return False, response.error_message

    def sudo_add_network(self, netuid: int, tempo: int = 0, modality: int = 0, wait_for_inclusion: bool = True, wait_for_finalization: bool = True ) -> Tuple[bool, Optional[str]]:
        r""" Adds a network to the mock chain using the sudo key.
        """
        with self.substrate as substrate:
            call = substrate.compose_call(
                    call_module='SubtensorModule',
                    call_function='sudo_add_network',
                    call_params = {
                        'netuid': netuid,
                        'tempo': tempo,
                        'modality': modality
                    }
                )

            wrapped_call = self.wrap_sudo(call)

            extrinsic = substrate.create_signed_extrinsic( call = wrapped_call, keypair = self.sudo_keypair )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )

            if not wait_for_finalization:
                return True, None
            
            response.process_events()
            if response.is_success:
                return True, None
            else:
                return False, response.error_message
            
    def sudo_register(self, netuid: int, hotkey: str, coldkey: str, stake: int = 0, balance: int = 0, wait_for_inclusion: bool = True, wait_for_finalization: bool = True ) -> Tuple[bool, Optional[str]]:
        r""" Registers a neuron to the subnet using sudo.
        """
        with self.substrate as substrate:
            call = substrate.compose_call(
                    call_module='SubtensorModule',
                    call_function='sudo_register',
                    call_params = {
                        'netuid': netuid,
                        'hotkey': hotkey,
                        'coldkey': coldkey,
                        'stake': stake,
                        'balance': balance
                    }
                )

            wrapped_call = self.wrap_sudo(call)

            extrinsic = substrate.create_signed_extrinsic( call = wrapped_call, keypair = self.sudo_keypair )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )

            if not wait_for_finalization:
                return True, None
            
            response.process_events()
            if response.is_success:
                return True, None
            else:
                return False, response.error_message