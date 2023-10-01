""" Create and init Axon, whcih services Forward and Backward requests from other neurons.
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

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
import base64
import asyncio
import inspect
import uvicorn
import argparse
import traceback
import threading
import bittensor
import contextlib

from inspect import signature, Signature, Parameter
from fastapi.responses import JSONResponse
from substrateinterface import Keypair
from fastapi import FastAPI, APIRouter, Request, Response, Depends
from starlette.types import Scope, Message
from starlette.responses import Response
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from typing import Dict, Optional, Tuple, Union, List, Callable, Any


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
    """
    The `axon` class is an object that forms the core part of bittensor's serving synapses.
    The class relies heavily on an underlying FastAPI router, which is utilized to create endpoints for different message types.
    Methods in this class are equipped to deal with incoming requests from other scenarios in the network and serve as the server face for a neuron.

    It accepts multiple arguments, like wallet, configuration parameters, ip address, server binding port, external ip, external port and max workers.
    Key methods involve managing and operating the FastAPI application router, including the attachment and operation of endpoints.
    The `axon` class offers flexibility to specify custom rules to forward, blacklist, prioritize and verify incoming requests against custom functions.

    The class also encapsulates methods to add command-line arguments for user-friendly interaction with the program, and supports the handling of these arguments,
    to define the behavior of the axon object.

    Internal mechanisms manage a thread pool to support concurrent requests handling, using defined priority levels.

    ### Example usage:

    ```python
    import bittensor

    class MySyanpse( bittensor.Synapse ):
        input: int = 1
        output: int = None

    # Define a custom request forwarding function
    def forward( synapse: MySyanpse ) -> MySyanpse:
        # Apply custom logic to synapse and return it
        synapse.output = 2
        return synapse

    # Define a custom request verification function
    def verify_my_synapse( synapse: MySyanpse ):
        # Apply custom verification logic to synapse
        # Optionally raise Exception

    # Define a custom request blacklist fucntion
    def blacklist_my_synapse( synapse: MySyanpse ) -> bool:
        # Apply custom blacklist
        # return False ( if non blacklisted ) or True ( if blacklisted )

    # Define a custom request priority fucntion
    def prioritize_my_synape( synapse: MySyanpse ) -> float:
        # Apply custom priority
        return 1.0

    # Initialize Axon object with a custom configuration
    my_axon = bittensor.axon(config=my_config, wallet=my_wallet, port=9090, ip="192.0.2.0", external_ip="203.0.113.0", external_port=7070)

    # Attach the endpoint with the specified verification and forwarding functions
    my_axon.attach(
        forward_fn = forward_my_synapse,
        verify_fn = verify_my_synapse,
        blacklist_fn = blacklist_my_synapse,
        priority_fn = prioritize_my_synape
    ).attach(
        forward_fn = forward_my_synapse_2,
        verify_fn = verify_my_synapse_2,
        blacklist_fn = blacklist_my_synapse_2,
        priority_fn = prioritize_my_synape_2
    ).serve(
        netuid = ...
        subtensor = ...
    ).start()
    ```
    """

    def info(self) -> "bittensor.AxonInfo":
        """Returns the axon info object associated with this axon."""
        return bittensor.AxonInfo(
            version=bittensor.__version_as_int__,
            ip=self.external_ip,
            ip_type=4,
            port=self.external_port,
            hotkey=self.wallet.hotkey.ss58_address,
            coldkey=self.wallet.coldkeypub.ss58_address,
            protocol=4,
            placeholder1=0,
            placeholder2=0,
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
            config (:obj:`Optional[bittensor.config]`, `optional`):
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
        if config is None:
            config = axon.config()
        config = copy.deepcopy(config)
        config.axon.ip = ip or config.axon.get("ip", bittensor.defaults.axon.ip)
        config.axon.port = port or config.axon.get("port", bittensor.defaults.axon.port)
        config.axon.external_ip = external_ip or config.axon.get(
            "external_ip", bittensor.defaults.axon.external_ip
        )
        config.axon.external_port = external_port or config.axon.get(
            "external_port", bittensor.defaults.axon.external_port
        )
        config.axon.max_workers = max_workers or config.axon.get(
            "max_workers", bittensor.defaults.axon.max_workers
        )
        axon.check_config(config)
        self.config = config

        # Get wallet or use default.
        self.wallet = wallet or bittensor.wallet()

        # Build axon objects.
        self.uuid = str(uuid.uuid1())
        self.ip = self.config.axon.ip
        self.port = self.config.axon.port
        self.external_ip = (
            self.config.axon.external_ip
            if self.config.axon.external_ip != None
            else bittensor.utils.networking.get_external_ip()
        )
        self.external_port = (
            self.config.axon.external_port
            if self.config.axon.external_port != None
            else self.config.axon.port
        )
        self.full_address = str(self.config.axon.ip) + ":" + str(self.config.axon.port)
        self.started = False

        # Build middleware
        self.thread_pool = bittensor.PriorityThreadPoolExecutor(
            max_workers=self.config.axon.max_workers
        )
        self.nonces = {}

        # Request default functions.
        self.forward_class_types = {}
        self.blacklist_fns = {}
        self.priority_fns = {}
        self.forward_fns = {}
        self.verify_fns = {}
        self.required_hash_fields = {}

        # Instantiate FastAPI
        self.app = FastAPI()
        log_level = "trace" if bittensor.logging.__trace_on__ else "critical"
        self.fast_config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.config.axon.port, log_level=log_level
        )
        self.fast_server = FastAPIThreadedServer(config=self.fast_config)
        self.router = APIRouter()
        self.app.include_router(self.router)

        # Build ourselves as the middleware.
        self.app.add_middleware(AxonMiddleware, axon=self)

        # Attach default forward.
        def ping(r: bittensor.Synapse) -> bittensor.Synapse:
            return r

        self.attach(
            forward_fn=ping, verify_fn=None, blacklist_fn=None, priority_fn=None
        )

    def attach(
        self,
        forward_fn: Callable,
        blacklist_fn: Callable = None,
        priority_fn: Callable = None,
        verify_fn: Callable = None,
    ) -> "bittensor.axon":
        """
        Registers an API endpoint to the FastAPI application router.
        It uses the name of the first argument of the 'forward_fn' function as the endpoint name.

        Args:
            forward_fn (Callable): Function to be called when the API endpoint is accessed.
                                   It should have at least one argument.
            blacklist_fn (Callable, optional): Function to filter out undesired requests. It should take the same arguments
                                               as 'forward_fn' and return a boolean value. Defaults to None, meaning no blacklist filter will be used.
            priority_fn (Callable, optional): Function to rank requests based on their priority. It should take the same arguments
                                              as 'forward_fn' and return a numerical value representing the request's priority.
                                              Defaults to None, meaning no priority sorting will be applied.
            verify_fn (Callable, optional): Function to verify requests. It should take the same arguments as 'forward_fn' and return
                                            a boolean value. If None, 'self.default_verify' function will be used.

        Note: 'forward_fn', 'blacklist_fn', 'priority_fn', and 'verify_fn' should be designed to receive the same parameters.

        Raises:
            AssertionError: If 'forward_fn' does not have the signature: forward( synapse: YourSynapse ) -> synapse:
            AssertionError: If 'blacklist_fn' does not have the signature: blacklist( synapse: YourSynapse ) -> bool
            AssertionError: If 'priority_fn' does not have the signature: priority( synapse: YourSynapse ) -> float
            AssertionError: If 'verify_fn' does not have the signature: verify( synapse: YourSynapse ) -> None

        Returns:
            self: Returns the instance of the AxonServer class for potential method chaining.
        """

        # Assert 'forward_fn' has exactly one argument
        forward_sig = signature(forward_fn)
        assert (
            len(list(forward_sig.parameters)) == 1
        ), "The passed function must have exactly one argument"

        # Obtain the class of the first argument of 'forward_fn'
        request_class = forward_sig.parameters[
            list(forward_sig.parameters)[0]
        ].annotation

        # Assert that the first argument of 'forward_fn' is a subclass of 'bittensor.Synapse'
        assert issubclass(
            request_class, bittensor.Synapse
        ), "The argument of forward_fn must inherit from bittensor.Synapse"

        # Obtain the class name of the first argument of 'forward_fn'
        request_name = forward_sig.parameters[
            list(forward_sig.parameters)[0]
        ].annotation.__name__

        # Add the endpoint to the router, making it available on both GET and POST methods
        self.router.add_api_route(
            f"/{request_name}",
            forward_fn,
            methods=["GET", "POST"],
            dependencies=[Depends(self.verify_body_integrity)],
        )
        self.app.include_router(self.router)

        # Expected signatures for 'blacklist_fn', 'priority_fn' and 'verify_fn'
        blacklist_sig = Signature(
            [
                Parameter(
                    "synapse",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=forward_sig.parameters[
                        list(forward_sig.parameters)[0]
                    ].annotation,
                )
            ],
            return_annotation=Tuple[bool, str],
        )
        priority_sig = Signature(
            [
                Parameter(
                    "synapse",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=forward_sig.parameters[
                        list(forward_sig.parameters)[0]
                    ].annotation,
                )
            ],
            return_annotation=float,
        )
        verify_sig = Signature(
            [
                Parameter(
                    "synapse",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=forward_sig.parameters[
                        list(forward_sig.parameters)[0]
                    ].annotation,
                )
            ],
            return_annotation=None,
        )

        # Check the signature of blacklist_fn, priority_fn and verify_fn if they are provided
        if blacklist_fn:
            assert (
                signature(blacklist_fn) == blacklist_sig
            ), "The blacklist_fn function must have the signature: blacklist( synapse: {} ) -> Tuple[bool, str]".format(
                request_name
            )
        if priority_fn:
            assert (
                signature(priority_fn) == priority_sig
            ), "The priority_fn function must have the signature: priority( synapse: {} ) -> float".format(
                request_name
            )
        if verify_fn:
            assert (
                signature(verify_fn) == verify_sig
            ), "The verify_fn function must have the signature: verify( synapse: {} ) -> None".format(
                request_name
            )

        # Store functions in appropriate attribute dictionaries
        self.forward_class_types[request_name] = forward_sig.parameters[
            list(forward_sig.parameters)[0]
        ].annotation
        self.blacklist_fns[request_name] = blacklist_fn
        self.priority_fns[request_name] = priority_fn
        self.verify_fns[request_name] = (
            verify_fn or self.default_verify
        )  # Use 'default_verify' if 'verify_fn' is None
        self.forward_fns[request_name] = forward_fn

        # Parse required hash fields from the forward function protocol defaults
        required_hash_fields = request_class.__dict__["__fields__"][
            "required_hash_fields"
        ].default
        self.required_hash_fields[request_name] = required_hash_fields

        return self

    @classmethod
    def config(cls) -> "bittensor.config":
        """
        Parses command-line arguments to form a bittensor configuration object.

        Returns:
            bittensor.config: Configuration object with settings from command-line arguments.
        """
        parser = argparse.ArgumentParser()
        axon.add_args(parser)  # Add specific axon-related arguments
        return bittensor.config(parser, args=[])

    @classmethod
    def help(cls):
        """
        Prints the help text (list of command-line arguments and their descriptions) to stdout.
        """
        parser = argparse.ArgumentParser()
        axon.add_args(parser)  # Add specific axon-related arguments
        print(cls.__new__.__doc__)  # Print docstring of the class
        parser.print_help()  # Print parser's help text

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None):
        """
        Adds AxonServer-specific command-line arguments to the argument parser.

        Args:
            parser (argparse.ArgumentParser): Argument parser to which the arguments will be added.
            prefix (str, optional): Prefix to add to the argument names. Defaults to None.

        Note:
            Environment variables are used to define default values for the arguments.
        """
        prefix_str = "" if prefix is None else prefix + "."
        try:
            # Get default values from environment variables or use default values
            default_axon_port = os.getenv("BT_AXON_PORT") or 8091
            default_axon_ip = os.getenv("BT_AXON_IP") or "[::]"
            default_axon_external_port = os.getenv("BT_AXON_EXTERNAL_PORT") or None
            default_axon_external_ip = os.getenv("BT_AXON_EXTERNAL_IP") or None
            default_axon_max_workers = os.getenv("BT_AXON_MAX_WORERS") or 10

            # Add command-line arguments to the parser
            parser.add_argument(
                "--" + prefix_str + "axon.port",
                type=int,
                help="The local port this axon endpoint is bound to. i.e. 8091",
                default=default_axon_port,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.ip",
                type=str,
                help="""The local ip this axon binds to. ie. [::]""",
                default=default_axon_ip,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.external_port",
                type=int,
                required=False,
                help="""The public port this axon broadcasts to the network. i.e. 8091""",
                default=default_axon_external_port,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.external_ip",
                type=str,
                required=False,
                help="""The external ip this axon broadcasts to the network to. ie. [::]""",
                default=default_axon_external_ip,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.max_workers",
                type=int,
                help="""The maximum number connection handler threads working simultaneously on this endpoint. 
                        The grpc server distributes new worker threads to service requests up to this number.""",
                default=default_axon_max_workers,
            )

        except argparse.ArgumentError:
            # Exception handling for re-parsing arguments
            pass

    async def verify_body_integrity(self, request: Request):
        """
        Asynchronously verifies the integrity of the body of a request by comparing the hash of required fields
        with the corresponding hashes provided in the request headers. This method is critical for ensuring
        that the incoming request payload has not been altered or tampered with during transmission, establishing
        a level of trust and security between the sender and receiver in the network.

        Args:
            request (Request): The incoming FastAPI request object containing both headers and the request body.

        Returns:
            dict: Returns the parsed body of the request as a dictionary if all the hash comparisons match,
                indicating that the body is intact and has not been tampered with.

        Raises:
            JSONResponse: Raises a JSONResponse with a 400 status code if any of the hash comparisons fail,
                        indicating a potential integrity issue with the incoming request payload.
                        The response includes the detailed error message specifying which field has a hash mismatch.

        Example:
            Assuming this method is set as a dependency in a route:

            @app.post("/some_endpoint")
            async def some_endpoint(body_dict: dict = Depends(verify_body_integrity)):
                # body_dict is the parsed body of the request and is available for use in the route function.
                # The function only executes if the body integrity verification is successful.
                ...
        """
        # Await and load the request body so we can inspect it
        body = await request.body()
        request_body = body.decode() if isinstance(body, bytes) else body

        # Gather the required field names from the axon's required_hash_fields dict
        request_name = request.url.path.split("/")[1]
        required_hash_fields = self.required_hash_fields[request_name]

        # Load the body dict and check if all required field hashes match
        body_dict = json.loads(request_body)
        field_hashes = []
        for required_field in required_hash_fields:
            # Hash the field in the body to compare against the header hashes
            body_value = body_dict.get(required_field, None)
            if body_value == None:
                raise ValueError(f"Missing required field {required_field}")
            field_hash = bittensor.utils.hash(str(body_value))
            field_hashes.append(field_hash)

        parsed_body_hash = bittensor.utils.hash("".join(field_hashes))
        body_hash = request.headers.get("computed_body_hash", "")
        if parsed_body_hash != body_hash:
            raise ValueError(
                f"Hash mismatch between header body hash {body_hash} and parsed body hash {parsed_body_hash}"
            )

        # If body is good, return the parsed body so that it can be passed onto the route function
        return body_dict

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        """
        This method checks the configuration for the axon's port and wallet.

        Args:
            config (bittensor.config): The config object holding axon settings.

        Raises:
            AssertionError: If the axon or external ports are not in range [1024, 65535]
        """
        assert (
            config.axon.port > 1024 and config.axon.port < 65535
        ), "Axon port must be in range [1024, 65535]"

        assert config.axon.external_port is None or (
            config.axon.external_port > 1024 and config.axon.external_port < 65535
        ), "External port must be in range [1024, 65535]"

    def __str__(self) -> str:
        """
        Provides a human-readable representation of the Axon instance.
        """
        return "Axon({}, {}, {}, {}, {})".format(
            self.ip,
            self.port,
            self.wallet.hotkey.ss58_address,
            "started" if self.started else "stopped",
            list(self.forward_fns.keys()),
        )

    def __repr__(self) -> str:
        """
        Provides a machine-readable (unambiguous) representation of the Axon instance.
        It is made identical to __str__ in this case.
        """
        return self.__str__()

    def __del__(self):
        """
        This magic method is called when the Axon object is about to be destroyed.
        It ensures that the Axon server shuts down properly.
        """
        self.stop()

    def start(self) -> "bittensor.axon":
        """
        Starts the Axon server's GRPC server thread and marks the Axon as started.

        Returns:
            bittensor.axon: The started Axon instance.
        """
        self.fast_server.start()
        self.started = True
        return self

    def stop(self) -> "bittensor.axon":
        """
        Stops the Axon server's GRPC server thread and marks the Axon as stopped.

        Returns:
            bittensor.axon: The stopped Axon instance.
        """
        self.fast_server.stop()
        self.started = False
        return self

    def serve(
        self, netuid: int, subtensor: bittensor.subtensor = None
    ) -> "bittensor.axon":
        """
        Serves the axon on the passed subtensor connection using the axon wallet.

        Args:
            netuid: int
                The subnet uid to register on.
            subtensor: Optional[ bittensor.Subtensor ]
                The subtensor connection to use for serving.
        Returns:
            bittensor.axon: The served Axon instance.
        """
        if subtensor == None:
            subtensor = bittensor.subtensor()
        subtensor.serve_axon(netuid=netuid, axon=self)
        return self

    async def default_verify(self, synapse: bittensor.Synapse):
        """
        This method is used to verify the authenticity of a received message using a digital signature.
        It ensures that the message was not tampered with and was sent by the expected sender.

        Args:
            synapse: bittensor.Synapse
                bittensor request synapse.

        Raises:
            Exception: If the receiver_hotkey doesn't match with self.receiver_hotkey.
            Exception: If the nonce is not larger than the previous nonce for the same endpoint key.
            Exception: If the signature verification fails.

        After successful verification, the nonce for the given endpoint key is updated.

        Note:
            The verification process assumes the use of an asymmetric encryption algorithm,
            where the sender signs the message with their private key and the receiver verifies the signature using the sender's public key.
        """
        # Build the keypair from the dendrite_hotkey
        keypair = Keypair(ss58_address=synapse.dendrite.hotkey)

        # Build the signature messages.
        message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}.{self.wallet.hotkey.ss58_address}.{synapse.dendrite.uuid}.{synapse.computed_body_hash}"

        # Build the unique endpoint key.
        endpoint_key = f"{synapse.dendrite.hotkey}:{synapse.dendrite.uuid}"

        # Check the nonce from the endpoint key.
        if endpoint_key in self.nonces.keys():
            # Ensure the nonce increases.
            if synapse.dendrite.nonce <= self.nonces[endpoint_key]:
                raise Exception("Nonce is too small")

        if not keypair.verify(message, synapse.dendrite.signature):
            raise Exception(
                f"Signature mismatch with {message} and {synapse.dendrite.signature}"
            )

        # Success
        self.nonces[endpoint_key] = synapse.dendrite.nonce


class AxonMiddleware(BaseHTTPMiddleware):
    """
    Class AxonMiddleware handles the entire process of the request to the Axon.
    It fills the necessary information into the synapse and manages the logging of messages and errors.
    It handles the verification, blacklist checking and running priority functions.
    This class also runs the requested function and updates the headers of the response.
    """

    def __init__(self, app: "AxonMiddleware", axon: "bittensor.axon"):
        """
        Initialize the AxonMiddleware class.

        Args:
        app (object): An instance of the application where the middleware processor is used.
        axon (object): The axon instance used to process the requests.
        """
        super().__init__(app)
        self.axon = axon

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Request:
        """
        Processes incoming requests.

        Args:
            request(starlette: Request): The incoming request.
            call_next(starlette: RequestResponseEndpoint): The function to call after processing the request.

        Returns:
            response (starlette: Response): The processed request.
        """
        # Records the start time of the request processing.
        start_time = time.time()

        try:
            # Set up the synapse from its headers.
            synapse: bittensor.Synapse = await self.preprocess(request)

            # Logs the start of the request processing
            bittensor.logging.debug(
                f"axon     | <-- | {request.headers.get('content-length', -1)} B | {synapse.name} | {synapse.dendrite.hotkey} | {synapse.dendrite.ip}:{synapse.dendrite.port} | 200 | Success "
            )

            # Call the blacklist function
            await self.blacklist(synapse)

            # Call verify and return the verified request
            await self.verify(synapse)

            # Call the priority function
            await self.priority(synapse)

            # Call the run function
            response = await self.run(synapse, call_next, request)

            # Call the postprocess function
            response = await self.postprocess(synapse, response, start_time)

        # Start of catching all exceptions, updating the status message, and processing time.
        except Exception as e:
            # Log the exception for debugging purposes.
            bittensor.logging.trace(f"Forward exception: {traceback.format_exc()}")

            # Set the status message of the synapse to the string representation of the exception.
            synapse.axon.status_message = f"{str(e)}"

            # Calculate the processing time by subtracting the start time from the current time.
            synapse.axon.process_time = str(time.time() - start_time)

            # Create a JSON response with a status code of 500 (internal server error),
            # synapse headers, and an empty content.
            response = JSONResponse(
                status_code=500, headers=synapse.to_headers(), content={}
            )

        # Logs the end of request processing and returns the response
        finally:
            # Log the details of the processed synapse, including total size, name, hotkey, IP, port,
            # status code, and status message, using the debug level of the logger.
            bittensor.logging.debug(
                f"axon     | --> | {response.headers.get('content-length', -1)} B | {synapse.name} | {synapse.dendrite.hotkey} | {synapse.dendrite.ip}:{synapse.dendrite.port}  | {synapse.axon.status_code} | {synapse.axon.status_message}"
            )

            # Return the response to the requester.
            return response

    async def preprocess(self, request: Request) -> bittensor.Synapse:
        """
        Perform preprocess operations for the request and generate the synapse state object.

        Args:
            synapse (Synapse): The synapse instance representing the request.
        """
        # Extracts the request name from the URL path.
        request_name = request.url.path.split("/")[1]

        # Creates a synapse instance from the headers using the appropriate forward class type
        # based on the request name obtained from the URL path.
        synapse = self.axon.forward_class_types[request_name].from_headers(
            request.headers
        )
        synapse.name = request_name

        # Fills the local axon information into the synapse.
        synapse.axon.__dict__.update(
            {
                "version": str(bittensor.__version_as_int__),
                "uuid": str(self.axon.uuid),
                "nonce": f"{time.monotonic_ns()}",
                "status_message": "Success",
                "status_code": "100",
            }
        )

        # Fills the dendrite information into the synapse.
        synapse.dendrite.__dict__.update(
            {"port": str(request.client.port), "ip": str(request.client.host)}
        )

        # Signs the synapse from the axon side using the wallet hotkey.
        message = f"{synapse.axon.nonce}.{synapse.dendrite.hotkey}.{synapse.axon.hotkey}.{synapse.axon.uuid}"
        synapse.axon.signature = f"0x{self.axon.wallet.hotkey.sign(message).hex()}"

        # Return the setup synapse.
        return synapse

    async def verify(self, synapse: bittensor.Synapse):
        """
        Verify the request.

        Args:
            synapse ( bittensor.Synapse ): The synapse instance representing the request.

        Raises:
            Exception: If verification fails.
        """
        # Start of the verification process. Verification is the process where we ensure that
        # the incoming request is from a trusted source or fulfills certain requirements.
        # We get a specific verification function from 'verify_fns' dictionary that corresponds
        # to our request's name. Each request name (synapse name) has its unique verification function.
        verify_fn = self.axon.verify_fns[synapse.name]

        # If a verification function exists for the request's name
        if verify_fn:
            try:
                # We attempt to run the verification function using the synapse instance
                # created from the request. If this function runs without throwing an exception,
                # it means that the verification was successful.
                await verify_fn(synapse) if inspect.iscoroutinefunction(
                    verify_fn
                ) else verify_fn(synapse)
            except Exception as e:
                # If there was an exception during the verification process, we log that
                # there was a verification exception.
                bittensor.logging.trace(f"Verify exception {str(e)}")

                # We set the status code of the synapse to "401" which denotes an unauthorized access.
                synapse.axon.status_code = "401"

                # We raise an exception to stop the process and return the error to the requester.
                # The error message includes the original exception message.
                raise Exception(f"Not Verified with error: {str(e)}")

    async def blacklist(self, synapse: bittensor.Synapse):
        """
        Check if the request is blacklisted.

        Args:
            synapse (Synapse): The synapse instance representing the request.

        Raises:
            Exception: If the request is blacklisted.
        """
        # A blacklist is a list of keys or identifiers
        # that are prohibited from accessing certain resources.
        # We retrieve the blacklist checking function from the 'blacklist_fns' dictionary
        # that corresponds to the request's name (synapse name).
        blacklist_fn = self.axon.blacklist_fns[synapse.name]

        # If a blacklist checking function exists for the request's name
        if blacklist_fn:
            # We execute the blacklist checking function using the synapse instance as input.
            # If the function returns True, it means that the key or identifier is blacklisted.
            blacklisted, reason = (
                await blacklist_fn(synapse)
                if inspect.iscoroutinefunction(blacklist_fn)
                else blacklist_fn(synapse)
            )
            if blacklisted:
                # We log that the key or identifier is blacklisted.
                bittensor.logging.trace(f"Blacklisted: {blacklisted}, {reason}")

                # We set the status code of the synapse to "403" which indicates a forbidden access.
                synapse.axon.status_code = "403"

                # We raise an exception to halt the process and return the error message to the requester.
                raise Exception(f"Forbidden. Key is blacklisted: {reason}.")

    async def priority(self, synapse: bittensor.Synapse):
        """
        Execute the priority function for the request.

        A priority function is a function that determines the priority or urgency of processing the request compared to other requests.
        Args:
            synapse (bittensor.Synapse): The synapse instance representing the request.

        Raises:
            Exception: If the priority function times out.
        """
        # Retrieve the priority function from the 'priority_fns' dictionary that corresponds
        # to the request's name (synapse name).
        priority_fn = self.axon.priority_fns[synapse.name]

        async def submit_task(
            executor: bittensor.threadpool, priority: float
        ) -> Tuple[float, Any]:
            """
            Submits the given priority function to the specified executor for asynchronous execution.
            The function will run in the provided executor and return the priority value along with the result.

            Args:
                executor: The executor in which the priority function will be run.
                priority: The priority function to be executed.

            Returns:
                tuple: A tuple containing the priority value and the result of the priority function execution.
            """
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(executor, lambda: priority)
            result = await future
            return priority, result

        # If a priority function exists for the request's name
        if priority_fn:
            try:
                # Execute the priority function and get the priority value.
                priority = (
                    await priority_fn(synapse)
                    if inspect.iscoroutinefunction(priority_fn)
                    else priority_fn(synapse)
                )

                # Submit the task to the thread pool for execution with the given priority.
                # The submit_task function will handle the execution and return the result.
                _, result = await submit_task(self.axon.thread_pool, priority)

            except TimeoutError as e:
                # If the execution of the priority function exceeds the timeout,
                # it raises an exception to handle the timeout error.
                bittensor.logging.trace(f"TimeoutError: {str(e)}")

                # Set the status code of the synapse to "408" which indicates a timeout error.
                synapse.axon.status_code = "408"

                # Raise an exception to stop the process and return an appropriate error message to the requester.
                raise Exception(f"Response timeout after: {synapse.timeout}s")

    async def run(
        self,
        synapse: bittensor.Synapse,
        call_next: RequestResponseEndpoint,
        request: Request,
    ) -> Response:
        """
        Execute the requested function.

        Args:
            synapse: ( bittensor.Synapse ): call state.
            call_next: ( starlet RequestResponseEndpoint ): The function to call after processing the request.
            request: ( starlet Request ): The incoming request.

        Returns:
            response (starlet Response): The processed request.
        """
        try:
            # The requested function is executed by calling the 'call_next' function,
            # passing the original request as an argument. This function processes the request
            # and returns the response.
            response = await call_next(request)

        except Exception as e:
            # If an exception occurs during the execution of the requested function,
            # it is caught and handled here.

            # Log the exception for debugging purposes.
            bittensor.logging.trace(f"Run exception: {str(e)}")

            # Set the status code of the synapse to "500" which indicates an internal server error.
            synapse.axon.status_code = "500"

            # Raise an exception to stop the process and return an appropriate error message to the requester.
            raise Exception(f"Internal server error with error: {str(e)}")

        # Return the starlet response
        return response

    async def postprocess(
        self, synapse: bittensor.Synapse, response: Response, start_time: float
    ) -> Response:
        """
        Perform post-processing operations on the request.

        Args:
            synapse (bittensor.Synapse): The synapse instance representing the request.
            response (starlet Response): The response from the requested function.
            start_time (float): The start time of request processing.

        Returns:
            response (starlet Response): The processed request with updated headers.
        """
        # Set the status code of the synapse to "200" which indicates a successful response.
        synapse.axon.status_code = "200"

        # Set the status message of the synapse to "Success".
        synapse.axon.status_message = "Success"

        # Calculate the processing time by subtracting the start time from the current time.
        synapse.axon.process_time = str(time.time() - start_time)

        # Update the response headers with the headers from the synapse.
        response.headers.update(synapse.to_headers())

        return response
