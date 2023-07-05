""" Create and init Axon, whcih services Forward and Backward requests from other neurons.
"""
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

import os
import uuid
import copy
import json
import time
import uvicorn
import argparse
import threading
import bittensor
import contextlib

from threading import Lock
from inspect import signature
from types import SimpleNamespace
from substrateinterface import Keypair
from fastapi import FastAPI, APIRouter, Request
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from typing import Dict, Optional, Tuple, Union, List, Callable


""" FastAPI server that runs in a thread. 
"""
class FastAPIThreadedServer(uvicorn.Server):
    should_exit: bool = False
    is_running: bool = False

    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()

    def _wrapper_run(self):
        with self.run_in_thread():
            while not self.should_exit:
                time.sleep(1e-3)

    def start(self):
        if not self.is_running:
            self.should_exit = False
            thread = threading.Thread(target=self._wrapper_run, daemon=True)
            thread.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.should_exit = True

class axon:
    """ Axon object for serving synapse receptors. """

    def info(self) -> 'bittensor.AxonInfo':
        """Returns the axon info object associate with this axon.""" 
        return bittensor.AxonInfo(
            version = bittensor.__version_as_int__,
            ip = self.external_ip,
            ip_type = 4,
            port = self.external_port,
            hotkey = self.wallet.hotkey.ss58_address,
            coldkey = self.wallet.coldkeypub.ss58_address,
            protocol = 4,
            placeholder1 = 0, # placeholder1 = fast_api_port
            placeholder2 = 0,
        )
        
    def __init__(
        self,
        wallet: "bittensor.wallet" = None,
        config: Optional["bittensor.config"] = None,
        port: Optional[int] = None,
        ip: Optional[str] = None,
        external_ip: Optional[str] = None,
        external_port: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> "bittensor.axon":
        r"""Creates a new bittensor.Axon object from passed arguments.
        Args:
            config (:obj:`Optional[bittensor.Config]`, `optional`):
                bittensor.axon.config()
            wallet (:obj:`Optional[bittensor.wallet]`, `optional`):
                bittensor wallet with hotkey and coldkeypub.
            port (:type:`Optional[int]`, `optional`):
                Binding port.
            ip (:type:`Optional[str]`, `optional`):
                Binding ip.
            external_ip (:type:`Optional[str]`, `optional`):
                The external ip of the server to broadcast to the network.
            external_port (:type:`Optional[int]`, `optional`):
                The external port of the server to broadcast to the network.
            max_workers (:type:`Optional[int]`, `optional`):
                Used to create the threadpool if not passed, specifies the number of active threads servicing requests.
        """
        # Build and check config.
        if config is None: config = axon.config()
        config = copy.deepcopy(config)
        config.axon.ip = ip or config.axon.ip
        config.axon.port = port or config.axon.port
        config.axon.external_ip = external_ip or config.axon.external_ip
        config.axon.external_port = external_port or config.axon.external_port
        config.axon.max_workers = max_workers or config.axon.max_workers
        axon.check_config(config)
        self.config = config

        # Get wallet or use default.
        self.wallet = wallet or bittensor.wallet()

        # Build axon objects.
        self.uuid = str(uuid.uuid1())
        self.ip = self.config.axon.ip
        self.port = self.config.axon.port
        self.external_ip = self.config.axon.external_ip if self.config.axon.external_ip != None else bittensor.utils.networking.get_external_ip()
        self.external_port = self.config.axon.external_port if self.config.axon.external_port != None else self.config.axon.port
        self.full_address = str(self.config.axon.ip) + ":" + str(self.config.axon.port)
        self.started = False

        # Build middleware
        self.thread_pool = bittensor.PriorityThreadPoolExecutor( max_workers = self.config.axon.max_workers )
        self.nonces = {}

        # Request default functions.
        self.blacklist_fns = {}
        self.priority_fns = {}
        self.forward_fns = {}
        self.verify_fns = {}

        # Instantiate FastAPI
        self.app = FastAPI()
        log_level = 'trace' if bittensor.logging.__trace_on__ else 'critical'
        self.fast_config = uvicorn.Config( self.app, host = '0.0.0.0', port = self.config.axon.port, log_level = log_level)
        self.fast_server = FastAPIThreadedServer( config = self.fast_config )
        self.router = APIRouter()
        self.app.include_router( self.router )

        # Build ourselves as the middleware.
        # super().__init__( self.app )
        self.app.add_middleware( AxonMiddleware, axon = self )

        # Attach default forward.
        def ping( r: bittensor.BaseRequest ) -> bittensor.BaseRequest: return r 
        self.attach( forward_fn = ping, verify_fn = None, blacklist_fn = None, priority_fn = None )
    
    def attach(
            self, 
            forward_fn: Callable,
            blacklist_fn: Callable = None,
            priority_fn: Callable = None,
            verify_fn: Callable = None,
        ):
        """
        This method is used to register an API endpoint in the FastAPI application router.
        It uses the name of the first argument of the 'forward_fn' function as the endpoint name.
        
        Args:
            forward_fn (Callable): The function that will be called when the API endpoint is accessed. 
                                   It should have at least one argument.
            blacklist_fn (Callable, optional): A function to filter out unwanted requests. This function should take the same arguments 
                                               as 'forward_fn' and return a boolean value. If None, 'self.default_blacklist' will be used.
            priority_fn (Callable, optional): A function to sort the requests based on their priority. This function should take the same arguments 
                                              as 'forward_fn' and return a numerical value representing the request's priority. 
                                              If None, 'self.default_priority' will be used.

        After execution, the following associations are made:
            - API endpoint to 'forward_fn' in self.router (available on both GET and POST methods)
            - 'forward_fn' to 'blacklist_fn' in self.blacklist_fns
            - 'forward_fn' to 'priority_fn' in self.priority_fns
            - 'forward_fn' to itself in self.forward_fns

        Note:
            The default 'blacklist_fn' and 'priority_fn' should be defined in the instance (self) which calls this method.
            The 'forward_fn', 'blacklist_fn', and 'priority_fn' should be designed to receive the same parameters.
        """

        # Attains the http route by getting the class name of the first parameter to the passed function/
        sig = signature(forward_fn)
        assert len( list(sig.parameters) ) == 1, "The passed function must have exactly one argument"
        
        request_name = sig.parameters[list(sig.parameters)[0]].annotation.__name__
        self.router.add_api_route(f"/{request_name}", forward_fn, methods=["GET", "POST"])
        self.app.include_router(self.router)

        # Attach functions.
        self.blacklist_fns[request_name] = blacklist_fn or None # Uses no blacklist if not set.
        self.priority_fns[request_name] = priority_fn or None # No request priority if not set.
        self.verify_fns[request_name] = verify_fn or self.verify_fn # Default verify check signature. 
        self.forward_fns[request_name] = forward_fn # Attaches the forward function (simply placeholder)

        return self

    def verify_fn( 
        self, 
        base_request: bittensor.BaseRequest,
    ):
        """
        This method is used to verify the authenticity of a received message using a digital signature.
        It ensures that the message was not tampered with and was sent by the expected sender.

        Args:
            request: dict
                base_request built from request headers.

        Raises:
            Exception: If the receiver_hotkey doesn't match with self.receiver_hotkey.
            Exception: If the nonce is not larger than the previous nonce for the same endpoint key.
            Exception: If the signature verification fails.

        After successful verification, the nonce for the given endpoint key is updated.

        Note:
            The verification process assumes the use of an asymmetric encryption algorithm,
            where the sender signs the message with their private key and the receiver verifies the signature using the sender's public key.
        """
        print( base_request )
        # Build the keypair from the sender_hotkey
        keypair = Keypair(ss58_address = base_request.sender_hotkey)

        # Check that the receiver_hotkey's match.
        if base_request.receiver_hotkey != self.wallet.hotkey.ss58_address:
            raise Exception("receiver_hotkey does not match.")

        # Build the signature messages.
        message = f"{base_request.sender_nonce}.{base_request.sender_hotkey}.{base_request.receiver_hotkey}.{base_request.sender_uuid}"

        # Build the unique endpoint key.
        endpoint_key = f"{base_request.sender_hotkey}:{base_request.sender_uuid}"

        # Check the nonce from the endpoint key.
        if endpoint_key in self.nonces.keys():

            # Ensure the nonce increases.
            if base_request.sender_nonce <= self.nonces[endpoint_key]:
                raise Exception("Nonce is too small")
            
        if not keypair.verify(message, base_request.sender_signature):
            raise Exception("Signature mismatch")
        
        # Success
        self.nonces[endpoint_key] = base_request.sender_nonce

    @classmethod
    def config(cls) -> "bittensor.config":
        """Get config from the argument parser
        Return: bittensor.config object
        """
        parser = argparse.ArgumentParser()
        axon.add_args(parser)
        return bittensor.config(parser)

    @classmethod
    def help(cls):
        """Print help to stdout"""
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        print(cls.__new__.__doc__)
        parser.print_help()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None):
        """Accept specific arguments from parser"""
        prefix_str = "" if prefix is None else prefix + "."
        try:
            default_axon_port = os.getenv("BT_AXON_PORT") or 8091
            default_axon_ip = os.getenv("BT_AXON_IP") or "[::]"
            default_axon_external_port = os.getenv("BT_AXON_EXTERNAL_PORT") or None
            default_axon_external_ip = os.getenv("BT_AXON_EXTERNAL_IP") or None
            default_axon_max_workers = os.getenv("BT_AXON_MAX_WORERS") or 10
            parser.add_argument(
                "--" + prefix_str + "axon.port",
                type =int,
                help = """The local port this axon endpoint is bound to. i.e. 8091""",
                default = default_axon_port,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.ip",
                type = str,
                help = """The local ip this axon binds to. ie. [::]""",
                default = default_axon_ip,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.external_port",
                type = int,
                required = False,
                help = """The public port this axon broadcasts to the network. i.e. 8091""",
                default = default_axon_external_port,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.external_ip",
                type = str,
                required = False,
                help = """The external ip this axon broadcasts to the network to. ie. [::]""",
                default = default_axon_external_ip,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.max_workers",
                type = int,
                help = """The maximum number connection handler threads working simultaneously on this endpoint. 
                        The grpc server distributes new worker threads to service requests up to this number.""",
                default = default_axon_max_workers,
            )
        
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod
    def check_config(cls, config: "bittensor.Config"):
        """Check config for axon port and wallet"""
        assert (
            config.axon.port > 1024 and config.axon.port < 65535
        ), "port must be in range [1024, 65535]"
        assert config.axon.external_port is None or (
            config.axon.external_port > 1024 and config.axon.external_port < 65535
        ), "external port must be in range [1024, 65535]"

    def __str__(self) -> str:
        return "axon({}, {}, {}, {}, {})".format(
            self.ip,
            self.port,
            self.wallet.hotkey.ss58_address,
            "started" if self.started else "stopped",
            list(self.forward_fns.keys())
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __del__(self):
        r"""Called when this axon is deleted, ensures background threads shut down properly."""
        self.stop()

    def start(self) -> "bittensor.axon":
        r"""Starts the standalone axon GRPC server thread."""
        self.fast_server.start()
        self.started = True
        return self

    def stop(self) -> "bittensor.axon":
        r"""Stop the axon grpc server."""
        self.fast_server.stop()
        self.started = False
        return self


class AxonMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, axon):
        super().__init__(app)
        self.axon = axon

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Request:
        """
        This method handles incoming requests by performing several checks and operations, including
        parsing metadata, verifying signature, checking blacklist, acquiring a lock based on priority, 
        forwarding the request to the next endpoint and finally, returning the response.

        Args:
            request (Request): The incoming FastAPI request.
            call_next (RequestResponseEndpoint): The next middleware operation in the stack.

        Returns:
            default_response (bittensor.BaseRequest): Default response object. It's updated based on the 
                                                    result of various checks and returned in case of an error.
            response: The response from the next middleware operation in case of successful checks.

        Note:
            This method assumes that the incoming request contains necessary headers or metadata
            required for performing various checks such as 'sender_timeout', 'sender_version', 
            'sender_nonce', 'sender_uuid', 'sender_hotkey', 'sender_signature', 'receiver_hotkey'.
            
            In case any of these checks fail, it logs the error and returns the default response 
            with appropriate return code and message.
        """
        # Pull original timestamp.
        start_time = time.time()
        base_response = bittensor.BaseRequest.from_headers( request )
        base_response.log_axon_inbound()
        # Verify metadata
        try:
            verify_fn = self.axon.verify_fns[ base_response.request_name ]
            if verify_fn: verify_fn( base_response )
        except Exception as e:
            # Failed to parse metadata.
            base_response.return_code = bittensor.ReturnCode.FAILEDVERIFICATION.value
            base_response.return_message = f"Error checking signature {str(e)}"
            base_response.log_axon_outbound()
            return base_response

        # Check blacklist    
        try:    
            # Check the blacklist function.
            blacklist_fn = self.axon.blacklist_fns[ base_response.request_name ]
            if blacklist_fn:
                if blacklist_fn( base_response ):
                    raise Exception("Blacklisted")
        except Exception as e:
            # Item was blacklisted.
            base_response.return_code = bittensor.ReturnCode.BLACKLISTED.value
            base_response.return_message = "BLACKLISTED"
            base_response.log_axon_outbound()
            return base_response

        try:
            priority_fn = self.axon.priority_fns[ base_response.request_name ]
            if priority_fn:
                # Uses the priority function to run calls in order.
                # Once the event is set via the priority pool the process continues and calls the asyncio request.
                event = threading.Event()
                def set_event() -> bool: event.set()
                priority = priority_fn( base_response )
                future = self.axon.thread_pool.submit( set_event, priority = priority )
                future.result( timeout = float( base_response.sender_timeout ) )
                event.wait()
                response = await call_next( request )
            else:
                # Runs the forward function over the request without priority.
                response = await call_next( request )

        except TimeoutError as e:
            # Call timed out.
            base_response.return_code = bittensor.ReturnCode.TIMEOUT.value
            base_response.return_message = "TIMEOUT"
            base_response.log_axon_outbound()
            return base_response
        
        except Exception as e:
            # Unknown error on forward call. 
            base_response.return_code = bittensor.ReturnCode.UNKNOWN.value
            base_response.return_message = f"Unknown exception{str(e)}"
            base_response.log_axon_outbound()
            return base_response
        
        finally:
            # Success, no errors.
            # Unwrap message body.
            response_body = [ section async for section in response.body_iterator ]
            response_dict = json.loads( response_body[0] )

            # Fill response time.
            response_dict['process_time'] = (time.time() - start_time)

            # Back to bytes
            data = json.dumps(response_dict)
            byte_data = bytes(data, "utf-8")

            # Wrap in a new response object, specifying the Content-Length.
            response = Response( content=byte_data, headers={"Content-Length": str(len(byte_data))} )

            # Log outgoing response.
            base_response.log_axon_outbound()
            return response
    