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

import argparse
import copy
import os
import time
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Union
from warnings import warn

import torch

import grpc

import bittensor


class Synapse(ABC):

    synapse_name: str = "base"
    default_blacklist_stake: float = -1.0

    def __init__(self, config: "bittensor.Config" = None):
        """Initializes a new Synapse.
        Args:
            config (:obj:`bittensor.Config`, `optional`, defaults to bittensor.config()):
                bittensor config object.
            metagraph (:obj:`bittensor.metagraph.Metagraph`, `optional`, defaults to bittensor.metagraph.Metagraph()):
                bittensor metagraph object.
        """

        if self.synapse_name == "base":
            raise ValueError
        if self.default_blacklist_stake < 0:
            raise ValueError
        if config is None:
            config = Synapse.config()
        Synapse.check_config(config)
        print(config)

        self.config = copy.deepcopy(config)
        self.is_attached = False
        self.axon = None

    def __str__(self):
        return self.synapse_name

    ## Methods to be defined in the terminal child.
    @abstractmethod
    def _priority(self, forward_call: bittensor.BittensorCall) -> Union[float, None]:
        raise NotImplementedError("Must implement _priority() in subclass.")

    @abstractmethod
    def _blacklist(self, forward_call: bittensor.BittensorCall) -> bool:
        raise NotImplementedError("Must implement subclass_blacklist() in subclass.")

    @abstractmethod
    def forward(self, inputs: torch.Tensor,
                forward_call: bittensor.BittensorCall) -> bittensor.BittensorCall: # TODO: Return tensors instead, attach to call obj in middle child Forward
        raise NotImplementedError("Must implement forward() in subclass.")

    ## Methods to be defined in the request-specific synapse.
    # TODO: Refactor config so that it can exist in this base class alone
    @abstractmethod
    def _attach(self, axon: "bittensor.axon"):
        ...

    @abstractmethod
    def pre_process_request_proto_to_forward_call(
        self, request_proto: "bittensor.ForwardRequest"
    ) -> "bittensor.BittensorCall":
        """pre_process_request_proto_to_forward_call
        ------------------------------------------
        Args:
            request_proto (bittensor.ForwardRequest):
                request_proto to process in to a forward call.
        Returns:
            bittensor.BittensorCall (:obj:`bittensor.BittensorCall`, `required`):
                forward call processed from the request proto.
        """
        ...

    @abstractmethod
    def pre_process_request_proto_to_backward_call(
        self, request_proto: "bittensor.BackwardRequest"
    ) -> "bittensor.BittensorCall":
        """pre_process_request_proto_to_backward_call
        ------------------------------------------
        Args:
            request_proto (bittensor.BackwardRequest):
                request_proto to process in to a backward call.
        Returns:
            bittensor.BittensorCall (:obj:`bittensor.BittensorCall`, `required`):
                backward call processed from the request proto.
        """
        ...

    @abstractmethod
    def post_process_forward_call_to_response_proto(
        self, forward_call: "bittensor.BittensorCall"
    ) -> "bittensor.ForwardResponse":
        """post_process_forward_call_to_response_proto
        --------------------------------------------
        Args:
            forward_call (bittensor.BittensorCall):
                forward_call to process in to a response proto.
        Returns:
            response (bittensor.ForwardResponse):
                response proto processed from the forward call.
        """
        ...

    ## Base class methods, not to be modified
    def add_defaults(self, defaults: SimpleNamespace):
        """Add default values to defaults object"""
        env_default_stake_name = f"BT_{self.synapse_name}_blacklist_stake".upper()
        default_allow_non_registered_name = (
            f"BT_{self.synapse_name}_blacklist_allow_non_registered".upper()
        )

        defaults.synapse[self.synapse_name] = bittensor.Config()

        defaults.synapse[self.synapse_name].blacklist.stake = (
            os.getenv(env_default_stake_name)
            if os.getenv(env_default_stake_name) is not None
            else self.default_blacklist_stake
        )
        defaults.synapse[self.synapse_name].blacklist.allow_non_registered = (
            os.getenv(default_allow_non_registered_name)
            if os.getenv(default_allow_non_registered_name) is not None
            else True
        )

    def add_args(self, parser: argparse.ArgumentParser, prefix: str = None):
        """Accept specific arguments from parser"""
        prefix_str = "" if prefix is None else prefix + "."

        arg_stake_name = "".join(["--", prefix_str, f"synapse.{self.synapse_name}.blacklist.stake"])
        arg_allow_non_registered_name = "".join(
            ["--", prefix_str, f"synapse.{self.synapse_name}.blacklist.allow_non_registered"]
        )

        try:
            parser.add_argument(
                arg_stake_name,
                type=float,
                help="The amount of stake (tao) required to make a call.",
                default=self.default_blacklist_stake,
            )
            parser.add_argument(
                arg_allow_non_registered_name,
                action="store_true",
                help="""If true, allow non-registered peers""",
                default=True,
            )
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    def priority(self, forward_call: bittensor.BittensorCall) -> float:
        """_priority: Returns the priority of the forward call.
        Args:
            forward_call (:obj:`bittensor.BittensorCall`, `required`):
                forward_call to check.
        Returns:
            float: priority of the forward call.
        """
        # Call subclass priority, if not implemented use the
        # metagraph priority based on stake.
        assert self.is_attached
        try:
            priority = self._priority(forward_call)
            if priority is not None:
                return priority

        except NotImplementedError:
            warn("_priority is not implemented in the subclass!")
            if self.axon.metagraph is not None:
                uid = self.axon.metagraph.hotkeys.index(forward_call.hotkey)
                return self.axon.metagraph.S[uid].item()
            else:
                return 0.0

    def blacklist(self, forward_call: bittensor.BittensorCall) -> bool:
        """__blacklist: Checks if the forward call is blacklisted.
        Args:
            forward_call (:obj:`bittensor.BittensorCall`, `required`):
                forward_call to check.
        Returns:
            bool: True if blacklisted, False otherwise.
        """
        assert self.is_attached
        # Call subclass blacklist and optionaly return if metagraph is None.
        try:
            sub_blacklist = self._blacklist(forward_call)
        except NotImplementedError:
            warn("_blacklist is not defined in the terminal child class, defaulting to blacklist=True.")
            sub_blacklist = True
        if self.axon.metagraph is None:
            return sub_blacklist

        # Check for registration
        def registration_check():
            is_registered = forward_call.hotkey in self.axon.metagraph.hotkeys
            if not is_registered:
                if self.synapse_config.blacklist.allow_non_registered:
                    return False
                raise Exception("Registration blacklist")

        # Blacklist based on stake.
        def stake_check() -> bool:
            uid = self.axon.metagraph.hotkeys.index(forward_call.hotkey)
            if self.axon.metagraph.S[uid].item() < self.config.synapse.blacklist.stake:
                raise Exception("Stake blacklist")
            return False

        # Optionally blacklist based on checks.
        try:
            registration_check()
            stake_check()
            return sub_blacklist
        except Exception as e:
            return True

    def attach(self, axon):
        """Attach Synapse to the axon."""
        assert not self.is_attached

        self._attach(axon)
        self.axon = axon
        self.is_attached = True

    def config(self) -> "bittensor.Config":
        """Returns the config for this synapse."""
        parser = argparse.ArgumentParser()
        self.add_args(parser)
        return bittensor.config(parser)

    def help(self):
        """Print help to stdout"""
        parser = argparse.ArgumentParser()
        self.add_args(parser)
        print(self.__new__.__doc__)
        parser.print_help()

    @staticmethod
    def check_config(config: "bittensor.Config"):
        pass

    def Forward(
        self, request: "bittensor.ForwardRequest", context: grpc.ServicerContext
    ) -> "bittensor.ForwardResponse":
        """ForwardTextLastHiddenState
        ----------------------------
        Args:
            request (bittensor.ForwardRequest):
                request.version (int): version of the caller.
                request.hotkey (string): hotkey of the neuron.
                request.timeout (float): timeout for the request.
            context (grpc.ServicerContext):
                grpc tcp context.
        Returns:
            response (bittensor.ForwardResponse):
                response.serialized_hidden_states (string): serialized hidden states.
        """
        if not self.is_attached:
            raise Exception("Synapse cannot be called unless it is attached. Call attach() first.")

        try:
            # Build forward call.
            forward_call = self.pre_process_request_proto_to_forward_call(request_proto=request)
            forward_call.hotkey = request.hotkey
            forward_call.start_time = time.time()
            forward_call.timeout = request.timeout
            forward_call.version = request.version

            # Check blacklist.
            if self.blacklist(forward_call):
                raise Exception("Blacklisted")
            # Get priority.
            priority = self._priority(forward_call)
            # Queue the forward call.
            future = self.axon.priority_threadpool.submit(
                self.forward,
                forward_call=forward_call,
                priority=priority,
            )
        except Exception as e:
            forward_call.request_code = bittensor.proto.ReturnCode.UnknownException
            forward_call.request_message = str(e)

        finally:
            # Log request.
            bittensor.logging.rpc_log(
                axon=True,
                forward=True,
                is_response=False,
                code=forward_call.request_code,
                call_time=time.time() - forward_call.start_time,
                pubkey=forward_call.hotkey,
                uid=None,
                inputs=forward_call.get_inputs_shape()
                if forward_call.request_code == bittensor.proto.ReturnCode.Success
                else None,
                outputs=None,
                message=forward_call.request_message,
                synapse=self.__str__(),
            )
            if forward_call.request_code != bittensor.proto.ReturnCode.Success:

                response = self.post_process_forward_call_to_response_proto(forward_call)
                response.hotkey = self.axon.wallet.hotkey.ss58_address
                response.version = bittensor.__version_as_int__
                return response

        # Do forward.
        try:
            # Get the result.
            tensor = future.result(timeout=forward_call.timeout)
            forward_call.text_outputs = tensor

        except Exception as e:
            forward_call.response_code = bittensor.proto.ReturnCode.UnknownException
            forward_call.response_message = str(e)
        finally:
            # Log response
            bittensor.logging.rpc_log(
                axon=True,
                forward=True,
                is_response=True,
                code=forward_call.response_code,
                call_time=time.time() - forward_call.start_time,
                pubkey=forward_call.hotkey,
                uid=None,
                inputs=list(forward_call.get_inputs_shape())
                if forward_call.response_code == bittensor.proto.ReturnCode.Success
                else None,
                outputs=list(forward_call.get_outputs_shape())
                if forward_call.response_code == bittensor.proto.ReturnCode.Success
                else None,
                message=forward_call.response_message,
                synapse=self.__str__(),
            )

            response = self.post_process_forward_call_to_response_proto(forward_call)
            response.hotkey = self.axon.wallet.hotkey.ss58_address
            response.version = bittensor.__version_as_int__
            return response

    def Backward(
        self, request: "bittensor.BackwardRequest", context: grpc.ServicerContext
    ) -> "bittensor.ForwardResponse":
        """ForwardTextLastHiddenState
        ----------------------------
        Args:
            request (bittensor.BackwardRequest):
                request.version (int): version of the caller.
                request.hotkey (string): hotkey of the neuron.
            context (grpc.ServicerContext):
                grpc tcp context.
        Returns:
            response (bittensor.BackwardResponse):
                response from the backward call.

        """
        if not self.is_attached:
            raise Exception("Synapse cannot be called unless it is attached. Call attach() first.")
        try:
            # Build backward call.
            backward_call = self.pre_process_request_proto_to_backward_call(request_proto=request)
            backward_call.hotkey = request.hotkey
            backward_call.start_time = time.time()
            backward_call.version = request.version

            # Check blacklist.
            if self.__blacklist(backward_call):
                raise Exception("Blacklisted")
            # Get priority.
            priority = self._priority(backward_call)
            # Queue the backward call.
            future = self.axon.priority_threadpool.submit(
                self.backward,
                backward_call=backward_call,
                priority=priority,
            )
        except Exception as e:
            backward_call.request_code = bittensor.proto.ReturnCode.UnknownException
            backward_call.request_message = str(e)
        finally:
            # Log request.
            bittensor.logging.rpc_log(
                axon=True,
                forward=False,
                is_response=False,
                code=backward_call.request_code,
                call_time=time.time() - backward_call.start_time,
                pubkey=backward_call.hotkey,
                uid=None,
                inputs=backward_call.get_inputs_shape()
                if backward_call.request_code == bittensor.proto.ReturnCode.Success
                else None,
                outputs=None,
                message=backward_call.request_message,
                synapse=self.__str__(),
            )
            if backward_call.request_code != bittensor.proto.ReturnCode.Success:
                response_proto = self.post_process_backward_call_to_response_proto(backward_call)
                response_proto.hotkey = self.axon.wallet.hotkey.ss58_address
                response_proto.version = bittensor.__version_as_int__
                response_proto.return_code = backward_call.request_code
                response_proto.message = backward_call.request_message
                return response_proto

        # Do backward.
        try:
            # Get the result.
            future.result(timeout=bittensor.__blocktime__)

        except Exception as e:
            backward_call.response_code = bittensor.proto.ReturnCode.UnknownException
            backward_call.response_message = str(e)
        finally:
            # Log response
            bittensor.logging.rpc_log(
                axon=True,
                forward=False,
                is_response=True,
                code=backward_call.response_code,
                call_time=time.time() - backward_call.start_time,
                pubkey=backward_call.hotkey,
                uid=None,
                inputs=list(backward_call.get_inputs_shape())
                if backward_call.response_code == bittensor.proto.ReturnCode.Success
                else None,
                outputs=list(backward_call.get_outputs_shape())
                if backward_call.response_code == bittensor.proto.ReturnCode.Success
                else None,
                message=backward_call.response_message,
                synapse=self.__str__(),
            )
            response_proto = self.post_process_backward_call_to_response_proto(backward_call)
            response_proto.hotkey = self.axon.wallet.hotkey.ss58_address
            response_proto.version = bittensor.__version_as_int__
            response_proto.return_code = backward_call.request_code
            response_proto.message = backward_call.request_message
            return response_proto
