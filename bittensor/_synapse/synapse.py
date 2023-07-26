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

import grpc
import time
import torch
import asyncio
import bittensor

from typing import Union, Optional, Callable, List, Dict, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SynapseCall(ABC):
    """Base class for all synapse calls."""

    is_forward: bool  # If it is an forward of backward
    name: str  # The name of the call.

    def __init__(
        self,
        synapse: "bittensor.Synapse",
        request_proto: object,
        context: grpc.ServicerContext,
    ):
        metadata = dict(context.invocation_metadata())
        (
            _,
            sender_hotkey,
            _,
            _,
        ) = synapse.axon.auth_interceptor.parse_signature(metadata)

        self.completed = False
        self.start_time = time.time()
        self.timeout = request_proto.timeout
        self.src_version = request_proto.version
        self.src_hotkey = sender_hotkey
        self.dest_hotkey = synapse.axon.wallet.hotkey.ss58_address
        self.dest_version = bittensor.__version_as_int__
        self.return_code: bittensor.proto.ReturnCode = (
            bittensor.proto.ReturnCode.Success
        )
        self.return_message: str = "Success"
        self.priority: float = 0

    def __repr__(self) -> str:
        return f"SynapseCall( from: {self.src_hotkey}, forward: {self.is_forward}, start: {self.start_time}, timeout: {self.timeout}, priority: {self.priority}, completed: {self.completed})"

    def __str__(self) -> str:
        return self.__repr__()

    @abstractmethod
    def get_inputs_shape(self) -> torch.Size:
        ...

    @abstractmethod
    def get_outputs_shape(self) -> torch.Size:
        ...

    @abstractmethod
    def get_response_proto(self) -> object:
        ...

    def _get_response_proto(self) -> object:
        proto = self.get_response_proto()
        proto.return_code = self.return_code
        proto.return_message = self.return_message
        return proto

    @abstractmethod
    def apply(self):
        ...

    def _apply(self):
        # TODO(const): measure apply time.
        self.apply()

    def end(self):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        self.completed = True

    def log_outbound(self):
        bittensor.logging.rpc_log(
            axon=True,
            forward=self.is_forward,
            is_response=False,
            code=self.return_code,
            call_time=0,
            pubkey=self.src_hotkey,
            uid=None,
            inputs=self.get_outputs_shape(),
            outputs=self.get_outputs_shape(),
            message=self.return_message,
            synapse=self.name,
        )

    def log_inbound(self):
        bittensor.logging.rpc_log(
            axon=True,
            forward=self.is_forward,
            is_response=True,
            code=self.return_code,
            call_time=self.elapsed if self.completed else 0,
            pubkey=self.src_hotkey,
            uid=None,
            inputs=self.get_inputs_shape(),
            outputs=self.get_inputs_shape(),
            message=self.return_message,
            synapse=self.name,
        )


class Synapse(ABC):
    name: str

    def __init__(self, axon: bittensor.axon):
        self.axon = axon

    @abstractmethod
    def blacklist(self, call: SynapseCall) -> Union[Tuple[bool, str], bool]:
        ...

    def _blacklist(self, call: SynapseCall) -> Union[bool, str]:
        blacklist = self.blacklist(call)
        if isinstance(blacklist, tuple):
            return blacklist
        elif isinstance(blacklist, bool):
            return blacklist, "no reason specified"
        else:
            raise ValueError(
                "Blacklist response had type {} expected one of bool or Tuple[bool, str]".format(
                    blacklist
                )
            )

    @abstractmethod
    def priority(self, call: SynapseCall) -> float:
        ...

    def apply(self, call: SynapseCall) -> object:
        bittensor.logging.trace("Synapse: {} received call: {}".format(self.name, call))
        try:
            call.log_inbound()

            # Check blacklist.
            blacklist, reason = self._blacklist(call)
            if blacklist:
                call.return_code = bittensor.proto.ReturnCode.Blacklisted
                call.return_message = reason
                bittensor.logging.info(
                    "Synapse: {} blacklisted call: {} reason: {}".format(
                        self.name, call, reason
                    )
                )

            # Make call.
            else:
                # Queue the forward call with priority.
                call.priority = self.priority(call)
                future = self.axon.priority_threadpool.submit(
                    call._apply,
                    priority=call.priority,
                )
                bittensor.logging.trace(
                    "Synapse: {} loaded future: {}".format(self.name, future)
                )
                future.result(timeout=call.timeout)
                bittensor.logging.trace(
                    "Synapse: {} completed call: {}".format(self.name, call)
                )

        # Catch timeouts
        except asyncio.TimeoutError:
            bittensor.logging.trace(
                "Synapse: {} timeout: {}".format(self.name, call.timeout)
            )
            call.return_code = bittensor.proto.ReturnCode.Timeout
            call.return_message = "GRPC request timeout after: {}s".format(call.timeout)

        # Catch unknown exceptions.
        except Exception as e:
            bittensor.logging.trace(
                "Synapse: {} unknown error: {}".format(self.name, str(e))
            )
            call.return_code = bittensor.proto.ReturnCode.UnknownException
            call.return_message = str(e)

        # Finally return the call.
        finally:
            bittensor.logging.trace(
                "Synapse: {} finalize call {}".format(self.name, call)
            )
            call.end()
            call.log_outbound()
            return call._get_response_proto()
