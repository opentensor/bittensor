# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
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

import time
import grpc
import uuid
import unittest
import unittest.mock as mock
from unittest.mock import MagicMock

import bittensor
from bittensor.utils.test_utils import get_random_unused_port

wallet = bittensor.wallet.mock()
axon = bittensor.axon( wallet = wallet, metagraph = None )

sender_wallet = bittensor.wallet.mock()

def gen_nonce():
    return f"{time.monotonic_ns()}"

def test_axon_start():
    mock_wallet = MagicMock(
        spec=bittensor.Wallet,
        coldkey=MagicMock(),
        coldkeypub=MagicMock(
            # mock ss58 address
            ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        ),
        hotkey=MagicMock(
            ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        ),
    )
    axon = bittensor.axon( wallet = mock_wallet, metagraph = None )
    axon.start()
    assert axon.server._state.stage == grpc._server._ServerStage.STARTED

def test_axon_stop():
    mock_wallet = MagicMock(
        spec=bittensor.Wallet,
        coldkey=MagicMock(),
        coldkeypub=MagicMock(
            # mock ss58 address
            ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        ),
        hotkey=MagicMock(
            ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        ),
    )
    axon = bittensor.axon( wallet = mock_wallet, metagraph = None )
    axon.start()
    time.sleep( 1 )
    axon.stop()
    time.sleep( 1 )
    assert axon.server._state.stage == grpc._server._ServerStage.STOPPED

def sign_v2(sender_wallet, receiver_wallet):
    nonce, receptor_uid = gen_nonce(), str(uuid.uuid1())
    sender_hotkey = sender_wallet.hotkey.ss58_address
    receiver_hotkey = receiver_wallet.hotkey.ss58_address
    message = f"{nonce}.{sender_hotkey}.{receiver_hotkey}.{receptor_uid}"
    signature = f"0x{sender_wallet.hotkey.sign(message).hex()}"
    return ".".join([nonce, sender_hotkey, signature, receptor_uid])

def sign(sender_wallet, receiver_wallet, receiver_version):
    return sign_v2(sender_wallet, receiver_wallet)

def test_sign_v2():
    sign_v2(sender_wallet, wallet)

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        val = s.connect_ex(('localhost', port))
        if val == 0:
            return True
        else:
            return False

def test_axon_is_destroyed():
    mock_wallet = MagicMock(
        spec=bittensor.Wallet,
        coldkey=MagicMock(),
        coldkeypub=MagicMock(
            # mock ss58 address
            ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        ),
        hotkey=MagicMock(
            ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        ),
    )

    port = get_random_unused_port()
    assert is_port_in_use( port ) == False
    axon = bittensor.axon ( wallet = mock_wallet, metagraph = None, port = port )
    assert is_port_in_use( port ) == True
    axon.start()
    assert is_port_in_use( port ) == True
    axon.stop()
    assert is_port_in_use( port ) == False
    axon.__del__()
    assert is_port_in_use( port ) == False

    port = get_random_unused_port()
    assert is_port_in_use( port ) == False
    axon2 = bittensor.axon ( wallet = mock_wallet, metagraph = None, port = port )
    assert is_port_in_use( port ) == True
    axon2.start()
    assert is_port_in_use( port ) == True
    axon2.__del__()
    assert is_port_in_use( port ) == False

    port_3 = get_random_unused_port()
    assert is_port_in_use( port_3 ) == False
    axonA = bittensor.axon ( wallet = mock_wallet, metagraph = None, port = port_3 )
    assert is_port_in_use( port_3 ) == True
    axonB = bittensor.axon ( wallet = mock_wallet, metagraph = None, port = port_3 )
    assert axonA.server != axonB.server
    assert is_port_in_use( port_3 ) == True
    axonA.start()
    assert is_port_in_use( port_3 ) == True
    axonB.start()
    assert is_port_in_use( port_3 ) == True
    axonA.__del__()
    assert is_port_in_use( port ) == False
    axonB.__del__()
    assert is_port_in_use( port ) == False

# test external axon args
class TestExternalAxon(unittest.TestCase):
    """
    Tests the external axon config flags
    `--axon.external_port` and `--axon.external_ip`
    Need to verify the external config is used when broadcasting to the network
    and the internal config is used when creating the grpc server

    Also test the default behaviour when no external axon config is provided
    (should use the internal axon config, like usual)
    """

    def test_external_ip_not_set_dont_use_internal_ip(self):
        # Verify that not setting the external ip arg will NOT default to the internal axon ip
        mock_add_insecure_port = mock.MagicMock(return_value=None)
        mock_server = mock.MagicMock(
            add_insecure_port=mock_add_insecure_port
        )

        mock_config = bittensor.axon.config()
        mock_wallet = MagicMock(
            spec=bittensor.Wallet,
            coldkey=MagicMock(),
            coldkeypub=MagicMock(
                # mock ss58 address
                ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
            ),
            hotkey=MagicMock(
                ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
            ),
        )
        axon = bittensor.axon ( wallet = mock_wallet, metagraph = None, ip = 'fake_ip', server = mock_server, config = mock_config )
        assert axon.external_ip != axon.ip # should be different
        assert (axon.external_ip is None) or (axon.external_ip == bittensor.utils.networking.get_external_ip()) # should be None OR default from bittensor.utils

    def test_external_port_not_set_use_internal_port(self):
        # Verify that not setting the external port arg will default to the internal axon port
        mock_config = bittensor.axon.config()

        mock_wallet = mock.MagicMock(
            hotkey = mock.MagicMock(
                ss58_address = 'fake_hotkey_address',
                spec = bittensor.Keypair
            ),
            spec = bittensor.Wallet
        )

        with mock.patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.return_value = mock_wallet
            axon = bittensor.axon ( wallet = mock_wallet, metagraph = None, port = 1234, config=mock_config )
            assert axon.external_port == axon.port

    def test_external_port_set_full_address_internal(self):
        internal_port = 1234
        external_port = 5678
        mock_wallet = mock.MagicMock(
            hotkey = mock.MagicMock(
                ss58_address = 'fake_hotkey_address',
                spec = bittensor.Keypair
            ),
            spec = bittensor.Wallet
        )
        mock_add_insecure_port = mock.MagicMock(return_value=None)
        mock_server = mock.MagicMock(
            add_insecure_port=mock_add_insecure_port
        )

        mock_config = bittensor.axon.config()

        _ = bittensor.axon( wallet = mock_wallet, metagraph = None, port = internal_port, external_port = external_port, server = mock_server, config = mock_config )

        mock_add_insecure_port.assert_called_once()
        args, _ = mock_add_insecure_port.call_args
        full_address0 = args[0]

        assert f'{internal_port}' in full_address0 and f':{external_port}' not in full_address0

        mock_add_insecure_port.reset_mock()

        # Test using config
        mock_config = bittensor.axon.config()

        mock_config.axon.port = internal_port
        mock_config.axon.external_port = external_port

        _ = bittensor.axon( wallet = mock_wallet, metagraph = None, config = mock_config, server = mock_server )

        mock_add_insecure_port.assert_called_once()
        args, _ = mock_add_insecure_port.call_args
        full_address0 = args[0]

        assert f'{internal_port}' in full_address0, f'{internal_port} was not found in {full_address0}'
        assert f':{external_port}' not in full_address0, f':{external_port} was found in {full_address0}'

    def test_external_ip_set_full_address_internal(self):
        internal_ip = 'fake_ip_internal'
        external_ip = 'fake_ip_external'

        mock_wallet = mock.MagicMock(
            hotkey = mock.MagicMock(
                ss58_address = 'fake_hotkey_address',
                spec = bittensor.Keypair
            ),
            spec = bittensor.Wallet
        )

        mock_add_insecure_port = mock.MagicMock(return_value=None)
        mock_server = mock.MagicMock(
            add_insecure_port=mock_add_insecure_port
        )

        mock_config = bittensor.axon.config()

        _ = bittensor.axon( wallet = mock_wallet, metagraph = None, ip=internal_ip, external_ip=external_ip, server=mock_server, config=mock_config )

        mock_add_insecure_port.assert_called_once()
        args, _ = mock_add_insecure_port.call_args
        full_address0 = args[0]

        assert f'{internal_ip}' in full_address0 and f'{external_ip}' not in full_address0

        mock_add_insecure_port.reset_mock()

        # Test using config
        mock_config = bittensor.axon.config()
        mock_config.axon.external_ip = external_ip
        mock_config.axon.ip = internal_ip

        _ = bittensor.axon( wallet = mock_wallet, metagraph = None, config=mock_config, server=mock_server )

        mock_add_insecure_port.assert_called_once()
        args, _ = mock_add_insecure_port.call_args
        full_address0 = args[0]

        assert f'{internal_ip}' in full_address0, f'{internal_ip} was not found in {full_address0}'
        assert f'{external_ip}' not in full_address0, f'{external_ip} was found in {full_address0}'

    def test_external_ip_port_set_full_address_internal(self):
        internal_ip = 'fake_ip_internal'
        external_ip = 'fake_ip_external'
        internal_port = 1234
        external_port = 5678

        mock_wallet = mock.MagicMock(
            hotkey = mock.MagicMock(
                ss58_address = 'fake_hotkey_address',
                spec = bittensor.Keypair
            ),
            spec = bittensor.Wallet
        )

        mock_add_insecure_port = mock.MagicMock(return_value=None)
        mock_server = mock.MagicMock(
            add_insecure_port=mock_add_insecure_port
        )

        mock_config = bittensor.axon.config()

        _ = bittensor.axon( wallet = mock_wallet, metagraph = None, ip=internal_ip, external_ip=external_ip, port=internal_port, external_port=external_port, server=mock_server, config=mock_config )

        mock_add_insecure_port.assert_called_once()
        args, _ = mock_add_insecure_port.call_args
        full_address0 = args[0]

        assert f'{internal_ip}:{internal_port}' == full_address0 and f'{external_ip}:{external_port}' != full_address0

        mock_add_insecure_port.reset_mock()

        # Test using config
        mock_config = bittensor.axon.config()

        mock_config.axon.ip = internal_ip
        mock_config.axon.external_ip = external_ip
        mock_config.axon.port = internal_port
        mock_config.axon.external_port = external_port

        _ = bittensor.axon( wallet = mock_wallet, metagraph = None, config=mock_config, server=mock_server )

        mock_add_insecure_port.assert_called_once()
        args, _ = mock_add_insecure_port.call_args
        full_address1 = args[0]

        assert f'{internal_ip}:{internal_port}' == full_address1, f'{internal_ip}:{internal_port} is not eq to {full_address1}'
        assert f'{external_ip}:{external_port}' != full_address1, f'{external_ip}:{external_port} is eq to {full_address1}'