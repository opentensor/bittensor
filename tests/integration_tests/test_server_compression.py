# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import bittensor
import torch
import time
import uuid
from concurrent import futures
import contextlib
import functools
import itertools
import logging
import os
import unittest
from grpc._cython.cygrpc import CompressionLevel
import grpc
from grpc import _grpcio_metadata
import datetime
import select
import socket
import threading

"""Tests bittensor server and client side compression 
Apdated from: https://github.com/grpc/grpc/blob/0045e27bf93e4376363a86a6ce8bc7d13565b1da/src/python/grpcio_tests/tests/unit/_compression_test.py#L22

TODO (const): add tests for other grpc connections: currently, only unary_unary is tested
"""

_DEFAULT_SOCK_OPTIONS = (socket.SO_REUSEADDR,
                         socket.SO_REUSEPORT) if os.name != 'nt' else (
                             socket.SO_REUSEADDR,)

def get_socket(bind_address='localhost',
               port=0,
               listen=True,
               sock_options=_DEFAULT_SOCK_OPTIONS):
    """Opens a socket.
    Useful for reserving a port for a system-under-test.
    Args:
      bind_address: The host to which to bind.
      port: The port to which to bind.
      listen: A boolean value indicating whether or not to listen on the socket.
      sock_options: A sequence of socket options to apply to the socket.
    Returns:
      A tuple containing:
        - the address to which the socket is bound
        - the port to which the socket is bound
        - the socket object itself
    """
    _sock_options = sock_options if sock_options else []
    if socket.has_ipv6:
        address_families = (socket.AF_INET6, socket.AF_INET)
    else:
        address_families = (socket.AF_INET)
    for address_family in address_families:
        try:
            sock = socket.socket(address_family, socket.SOCK_STREAM)
            for sock_option in _sock_options:
                sock.setsockopt(socket.SOL_SOCKET, sock_option, 1)
            sock.bind((bind_address, port))
            if listen:
                sock.listen(1)
            return bind_address, sock.getsockname()[1], sock
        except OSError as os_error:
            sock.close()

            # This variable was undefined--what were the previously unrecoverable error numbers?
            # if os_error.errno in _UNRECOVERABLE_ERRNOS:
            #    raise
            # else:
            continue
        # For PY2, socket.error is a child class of IOError; for PY3, it is
        # pointing to OSError. We need this catch to make it 2/3 agnostic.
        except socket.error:  # pylint: disable=duplicate-except
            sock.close()
            continue
    raise RuntimeError("Failed to bind to {} with sock_options {}".format(
        bind_address, sock_options))


""" Proxies a TCP connection between a single client-server pair.
This proxy is not suitable for production, but should work well for cases in
which a test needs to spy on the bytes put on the wire between a server and
a client.
"""

_TCP_PROXY_BUFFER_SIZE = 1024
_TCP_PROXY_TIMEOUT = datetime.timedelta(milliseconds=500)


def _init_proxy_socket(gateway_address, gateway_port):
    proxy_socket = socket.create_connection((gateway_address, gateway_port))
    return proxy_socket


class TcpProxy(object):
    """Proxies a TCP connection between one client and one server."""

    def __init__(self, bind_address, gateway_address, gateway_port):
        self._bind_address = bind_address
        self._gateway_address = gateway_address
        self._gateway_port = gateway_port

        self._byte_count_lock = threading.RLock()
        self._sent_byte_count = 0
        self._received_byte_count = 0

        self._stop_event = threading.Event()

        self._port = None
        self._listen_socket = None
        self._proxy_socket = None

        # The following three attributes are owned by the serving thread.
        self._northbound_data = b""
        self._southbound_data = b""
        self._client_sockets = []

        self._thread = threading.Thread(target=self._run_proxy)

    def start(self):
        _, self._port, self._listen_socket = get_socket(
            bind_address=self._bind_address)
        self._proxy_socket = _init_proxy_socket(self._gateway_address,
                                                self._gateway_port)
        self._thread.start()

    def get_port(self):
        return self._port

    def _handle_reads(self, sockets_to_read):
        for socket_to_read in sockets_to_read:
            if socket_to_read is self._listen_socket:
                client_socket, client_address = socket_to_read.accept()
                self._client_sockets.append(client_socket)
            elif socket_to_read is self._proxy_socket:
                data = socket_to_read.recv(_TCP_PROXY_BUFFER_SIZE)
                with self._byte_count_lock:
                    self._received_byte_count += len(data)
                self._northbound_data += data
            elif socket_to_read in self._client_sockets:
                data = socket_to_read.recv(_TCP_PROXY_BUFFER_SIZE)
                if data:
                    with self._byte_count_lock:
                        self._sent_byte_count += len(data)
                    self._southbound_data += data
                else:
                    self._client_sockets.remove(socket_to_read)
            else:
                raise RuntimeError('Unidentified socket appeared in read set.')

    def _handle_writes(self, sockets_to_write):
        for socket_to_write in sockets_to_write:
            if socket_to_write is self._proxy_socket:
                if self._southbound_data:
                    self._proxy_socket.sendall(self._southbound_data)
                    self._southbound_data = b""
            elif socket_to_write in self._client_sockets:
                if self._northbound_data:
                    socket_to_write.sendall(self._northbound_data)
                    self._northbound_data = b""

    def _run_proxy(self):
        while not self._stop_event.is_set():
            expected_reads = (self._listen_socket, self._proxy_socket) + tuple(
                self._client_sockets)
            expected_writes = expected_reads
            sockets_to_read, sockets_to_write, _ = select.select(
                expected_reads, expected_writes, (),
                _TCP_PROXY_TIMEOUT.total_seconds())
            self._handle_reads(sockets_to_read)
            self._handle_writes(sockets_to_write)
        for client_socket in self._client_sockets:
            client_socket.close()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        self._listen_socket.close()
        self._proxy_socket.close()

    def get_byte_count(self):
        with self._byte_count_lock:
            return self._sent_byte_count, self._received_byte_count

    def reset_byte_count(self):
        with self._byte_count_lock:
            self._byte_count = 0
            self._received_byte_count = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

"""
Bittensor Wallet/Data Setup

"""

def sign(wallet):
    nounce = str(int(time.time() * 1000))
    receptor_uid = str(uuid.uuid1())
    message  = "{}{}{}".format(nounce, str(wallet.hotkey.ss58_address), receptor_uid)
    spliter = 'bitxx'
    signature = spliter.join([ nounce, str(wallet.hotkey.ss58_address), "0x" + wallet.hotkey.sign(message).hex(), receptor_uid])
    return signature


inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)



wallet = bittensor.wallet.mock()


"""
Grpc server/client compression tests

"""

STREAM_LENGTH = 200


_UNARY_UNARY = '/test/UnaryUnary'

# Cut down on test time.
_STREAM_LENGTH = STREAM_LENGTH // 16

_HOST = 'localhost'

_REQUEST = inputs_serialized.buffer
_COMPRESSION_RATIO_THRESHOLD = 0.05
_COMPRESSION_METHODS = (
    None,
    # Disabled for test tractability.
    #grpc.Compression.NoCompression,
    #grpc.Compression.Deflate,
    grpc.Compression.Gzip,
)
_COMPRESSION_NAMES = {
    None: 'Uncompressed',
    grpc.Compression.NoCompression: 'NoCompression',
    grpc.Compression.Deflate: 'DeflateCompression',
    grpc.Compression.Gzip: 'GzipCompression',
}

_TEST_OPTIONS = {
    'client_streaming': (False,),
    'server_streaming': (False,),
    'channel_compression': _COMPRESSION_METHODS,
    'multicallable_compression': (None,),
    'server_compression': _COMPRESSION_METHODS,
    'server_call_compression': _COMPRESSION_METHODS,
}


def _make_handle_unary_unary(pre_response_callback):

    def _handle_unary(request, servicer_context):
        if pre_response_callback:
            pre_response_callback(request, servicer_context)
        return request

    return _handle_unary

def set_call_compression(compression_method, request_or_iterator,
                         servicer_context):
    del request_or_iterator
    servicer_context.set_compression(compression_method)


def disable_next_compression(request, servicer_context):
    del request
    servicer_context.disable_next_message_compression()


def disable_first_compression(request, servicer_context):
    if int(request.decode('ascii')) == 0:
        servicer_context.disable_next_message_compression()


class _MethodHandler(grpc.RpcMethodHandler):

    def __init__(self, request_streaming, response_streaming,
                 pre_response_callback):
        self.request_streaming = request_streaming
        self.response_streaming = response_streaming
        self.request_deserializer = None
        self.response_serializer = None
        self.unary_unary = None
        self.unary_stream = None
        self.stream_unary = None
        self.stream_stream = None

        if self.request_streaming and self.response_streaming:
            self.stream_stream = _make_handle_stream_stream(
                pre_response_callback)
        elif not self.request_streaming and not self.response_streaming:
            self.unary_unary = _make_handle_unary_unary(pre_response_callback)
        elif not self.request_streaming and self.response_streaming:
            self.unary_stream = _make_handle_unary_stream(pre_response_callback)
        else:
            self.stream_unary = _make_handle_stream_unary(pre_response_callback)


class _GenericHandler(grpc.GenericRpcHandler):

    def __init__(self, pre_response_callback):
        self._pre_response_callback = pre_response_callback

    def service(self, handler_call_details):
        if handler_call_details.method == _UNARY_UNARY:
            return _MethodHandler(False, False, self._pre_response_callback)
        else:
            return None


@contextlib.contextmanager
def _instrumented_client_server_pair(channel_kwargs, server_kwargs,
                                     server_handler):
    if server_kwargs == {}:
        
        axon_server = grpc.server(futures.ThreadPoolExecutor(), **server_kwargs)
    else: 
        options=[('grpc.default_compression_level', CompressionLevel.high)]
        axon_server = grpc.server(futures.ThreadPoolExecutor(), **server_kwargs, options=options)
    server = bittensor.axon (
        port = 7081,
        ip = '127.0.0.1',
        server= axon_server,
        wallet = wallet,
        netuid = -1,
    )
    
    server.server.add_generic_rpc_handlers((server_handler,))
    server_port = server.server.add_insecure_port('{}:0'.format(_HOST))
    server.start()
    with TcpProxy(_HOST, _HOST, server_port) as proxy:
        proxy_port = proxy.get_port()
        options=[('grpc.default_compression_level', CompressionLevel.high)]
        with grpc.insecure_channel('{}:{}'.format(_HOST, proxy_port), options=options,
                                   **channel_kwargs) as client_channel:
            try:
                yield client_channel, proxy, server
            finally:
                server.server.stop(None)


def _get_byte_counts(channel_kwargs, multicallable_kwargs, client_function,
                     server_kwargs, server_handler, message):
    with _instrumented_client_server_pair(channel_kwargs, server_kwargs,
                                          server_handler) as pipeline:
        client_channel, proxy, server = pipeline
        client_function(client_channel, multicallable_kwargs, message)
        return proxy.get_byte_count()


def _get_compression_ratios(client_function, first_channel_kwargs,
                            first_multicallable_kwargs, first_server_kwargs,
                            first_server_handler, second_channel_kwargs,
                            second_multicallable_kwargs, second_server_kwargs,
                            second_server_handler, message):
    try:
        # This test requires the byte length of each connection to be deterministic. As
        # it turns out, flow control puts bytes on the wire in a nondeterministic
        # manner. We disable it here in order to measure compression ratios
        # deterministically.
        os.environ['GRPC_EXPERIMENTAL_DISABLE_FLOW_CONTROL'] = 'true'
        first_bytes_sent, first_bytes_received = _get_byte_counts(
            first_channel_kwargs, first_multicallable_kwargs, client_function,
            first_server_kwargs, first_server_handler, message)
        second_bytes_sent, second_bytes_received = _get_byte_counts(
            second_channel_kwargs, second_multicallable_kwargs, client_function,
            second_server_kwargs, second_server_handler, message)
        return ((second_bytes_sent - first_bytes_sent) /
                float(first_bytes_sent),
                (second_bytes_received - first_bytes_received) /
                float(first_bytes_received))
    finally:
        del os.environ['GRPC_EXPERIMENTAL_DISABLE_FLOW_CONTROL']


def _unary_unary_client(channel, multicallable_kwargs, message):
    multi_callable = channel.unary_unary(_UNARY_UNARY)
    response = multi_callable(request = message,
                              metadata= (
                                        ('rpc-auth-header','Bittensor'),
                                        ('bittensor-signature',sign(wallet)),
                                        ('bittensor-version',str(bittensor.__version_as_int__)),
                                        ('request_type', str(bittensor.proto.RequestType.FORWARD)),
                                        )
                              , **multicallable_kwargs)
    if response != message:
        raise RuntimeError("Request '{}' != Response '{}'".format(
            message, response))

class CompressionTest(unittest.TestCase):

    def assertCompressed(self, compression_ratio):
        print('Actual compression ratio: {}'.format(compression_ratio))
        self.assertLess(
            compression_ratio,
            -1.0 * _COMPRESSION_RATIO_THRESHOLD,
            msg='Actual compression ratio: {}'.format(compression_ratio))
        

    def assertNotCompressed(self, compression_ratio):
        self.assertGreaterEqual(
            compression_ratio,
            -1.0 * _COMPRESSION_RATIO_THRESHOLD,
            msg='Actual compession ratio: {}'.format(compression_ratio))

    def assertConfigurationCompressed(self, client_streaming, server_streaming,
                                      channel_compression,
                                      multicallable_compression,
                                      server_compression,
                                      server_call_compression):
        client_side_compressed = channel_compression or multicallable_compression
        server_side_compressed = server_compression or server_call_compression
        print(client_side_compressed)
        channel_kwargs = {
            'compression': channel_compression,
        } if channel_compression else {}
        multicallable_kwargs = {
            'compression': multicallable_compression,
        } if multicallable_compression else {}

        client_function = _unary_unary_client

        server_kwargs = {
            'compression': server_compression,
        } if server_compression else {}
        server_handler = _GenericHandler(
            functools.partial(set_call_compression, grpc.Compression.Gzip)
        ) if server_call_compression else _GenericHandler(None)
        sent_ratio, received_ratio = _get_compression_ratios(
            client_function, {}, {}, {}, _GenericHandler(None), channel_kwargs,
            multicallable_kwargs, server_kwargs, server_handler, _REQUEST)
        print(server_compression,channel_compression,multicallable_compression,server_call_compression)
        print(sent_ratio, received_ratio)

        if client_side_compressed:
            self.assertCompressed(sent_ratio)
        else:
            self.assertNotCompressed(sent_ratio)

        if server_side_compressed:
            self.assertCompressed(received_ratio)
        else:
            self.assertNotCompressed(received_ratio)


def _get_compression_str(name, value):
    return '{}{}'.format(name, _COMPRESSION_NAMES[value])


def _get_compression_test_name(client_streaming, server_streaming,
                               channel_compression, multicallable_compression,
                               server_compression, server_call_compression):
    client_arity = 'Stream' if client_streaming else 'Unary'
    server_arity = 'Stream' if server_streaming else 'Unary'
    arity = '{}{}'.format(client_arity, server_arity)
    channel_compression_str = _get_compression_str('Channel',
                                                   channel_compression)
    multicallable_compression_str = _get_compression_str(
        'Multicallable', multicallable_compression)
    server_compression_str = _get_compression_str('Server', server_compression)
    server_call_compression_str = _get_compression_str('ServerCall',
                                                       server_call_compression)
    return 'test{}_{}_{}_{}_{}'.format(arity, channel_compression_str,
                                   multicallable_compression_str,
                                   server_compression_str,
                                   server_call_compression_str)


def _test_options():
    for test_parameters in itertools.product(*_TEST_OPTIONS.values()):
        yield dict(zip(_TEST_OPTIONS.keys(), test_parameters))


for options in _test_options():

    def test_compression(**kwargs):

        def _test_compression(self):
            self.assertConfigurationCompressed(**kwargs)

        return _test_compression

    setattr(CompressionTest, _get_compression_test_name(**options),
            test_compression(**options))
if __name__ == '__main__':
    logging.basicConfig()
    unittest.main(verbosity=2)
