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
import copy
import torch
import argparse
import bittensor

from dataclasses import dataclass
from substrateinterface import Keypair
from typing import Dict, Optional, Tuple

class axon:
    """ Axon object for serving synapse receptors. """

    def info(self) -> 'axon_info':
        """Returns the axon info object associate with this axon."""
        return axon_info(
            version = bittensor.__version_as_int__,
            ip = self.external_ip,
            ip_type = 4,
            port = self.external_port,
            hotkey = self.wallet.hotkey.ss58_address,
            coldkey = self.wallet.coldkeypub.ss58_address,
            protocol = 4,
            placeholder1 = 0,
            placeholder2 = 0,
        )

    def __init__(
        self,
        wallet: "bittensor.Wallet",
        config: Optional["bittensor.config"] = None,
        port: Optional[int] = None,
        ip: Optional[str] = None,
        external_ip: Optional[str] = None,
        external_port: Optional[int] = None,
    ) -> "bittensor.Axon":
        r"""Creates a new bittensor.Axon object from passed arguments.
        Args:
            config (:obj:`Optional[bittensor.Config]`, `optional`):
                bittensor.axon.config()
            wallet (:obj:`Optional[bittensor.Wallet]`, `optional`):
                bittensor wallet with hotkey and coldkeypub.
            port (:type:`Optional[int]`, `optional`):
                Binding port.
            ip (:type:`Optional[str]`, `optional`):
                Binding ip.
            external_ip (:type:`Optional[str]`, `optional`):
                The external ip of the server to broadcast to the network.
            external_port (:type:`Optional[int]`, `optional`):
                The external port of the server to broadcast to the network.
        """
        self.wallet = wallet

        # Build and check config.
        if config is None:
            config = axon.config()
        config = copy.deepcopy(config)
        config.axon.port = port if port is not None else config.axon.port
        config.axon.ip = ip if ip is not None else config.axon.ip
        config.axon.external_ip = external_ip if external_ip is not None else config.axon.external_ip
        config.axon.external_port = (
            external_port if external_port is not None else config.axon.external_port
        )
        axon.check_config(config)
        self.config = config

        # Build axon objects.
        self.ip = self.config.axon.ip
        self.port = self.config.axon.port
        self.external_ip = self.config.axon.external_ip if self.config.axon.external_ip != None else bittensor.utils.networking.get_external_ip()
        self.external_port = self.config.axon.external_port if self.config.axon.external_port != None else self.config.axon.port
        self.full_address = str(self.config.axon.ip) + ":" + str(self.config.axon.port)

        # Build interceptor.
        self.receiver_hotkey = self.wallet.hotkey.ss58_address
        self.auth_interceptor = AuthInterceptor(receiver_hotkey=self.receiver_hotkey)

    @classmethod
    def config(cls) -> "bittensor.Config":
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
        if prefix is not None:
            if not hasattr(bittensor.defaults, prefix):
                setattr(bittensor.defaults, prefix, bittensor.Config())
            getattr(bittensor.defaults, prefix).axon = bittensor.defaults.axon

        bittensor.prioritythreadpool.add_args(parser, prefix=prefix_str + "axon")
        try:
            parser.add_argument(
                "--" + prefix_str + "axon.port",
                type=int,
                help="""The local port this axon endpoint is bound to. i.e. 8091""",
                default=bittensor.defaults.axon.port,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.ip",
                type=str,
                help="""The local ip this axon binds to. ie. [::]""",
                default=bittensor.defaults.axon.ip,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.external_port",
                type=int,
                required=False,
                help="""The public port this axon broadcasts to the network. i.e. 8091""",
                default=bittensor.defaults.axon.external_port,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.external_ip",
                type=str,
                required=False,
                help="""The external ip this axon broadcasts to the network to. ie. [::]""",
                default=bittensor.defaults.axon.external_ip,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.max_workers",
                type=int,
                help="""The maximum number connection handler threads working simultaneously on this endpoint.
                        The grpc server distributes new worker threads to service requests up to this number.""",
                default=bittensor.defaults.axon.max_workers,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.maximum_concurrent_rpcs",
                type=int,
                help="""Maximum number of allowed active connections""",
                default=bittensor.defaults.axon.maximum_concurrent_rpcs,
            )
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod
    def add_defaults(cls, defaults):
        """Adds parser defaults to object from enviroment variables."""
        defaults.axon = bittensor.Config()
        defaults.axon.port = (
            os.getenv("BT_AXON_PORT") if os.getenv("BT_AXON_PORT") is not None else 8091
        )
        defaults.axon.ip = os.getenv("BT_AXON_IP") if os.getenv("BT_AXON_IP") is not None else "[::]"
        defaults.axon.external_port = (
            os.getenv("BT_AXON_EXTERNAL_PORT")
            if os.getenv("BT_AXON_EXTERNAL_PORT") is not None
            else None
        )
        defaults.axon.external_ip = (
            os.getenv("BT_AXON_EXTERNAL_IP") if os.getenv("BT_AXON_EXTERNAL_IP") is not None else None
        )
        defaults.axon.max_workers = (
            os.getenv("BT_AXON_MAX_WORERS") if os.getenv("BT_AXON_MAX_WORERS") is not None else 10
        )
        defaults.axon.maximum_concurrent_rpcs = (
            os.getenv("BT_AXON_MAXIMUM_CONCURRENT_RPCS")
            if os.getenv("BT_AXON_MAXIMUM_CONCURRENT_RPCS") is not None
            else 400
        )

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
        return "Axon({}, {}, {}, {})".format(
            self.ip,
            self.port,
            self.wallet.hotkey.ss58_address,
            "started" if self.started else "stopped",
        )

    def __repr__(self) -> str:
        return self.__str__()

class AuthInterceptor:
    """Creates a new server interceptor that authenticates incoming messages from passed arguments."""

    def __init__(
        self,
        receiver_hotkey: str,
    ):
        r"""Creates a new server interceptor that authenticates incoming messages from passed arguments.
        Args:
            receiver_hotkey(str):
                the SS58 address of the hotkey which should be targeted by RPCs
        """
        super().__init__()
        self.nonces = {}
        self.receiver_hotkey = receiver_hotkey


    def parse_signature_v2(self, signature: str) -> Optional[Tuple[int, str, str, str]]:
        r"""Attempts to parse a signature using the v2 format"""
        parts = signature.split(".")
        if len(parts) != 4:
            return None
        try:
            nonce = int(parts[0])
        except ValueError:
            return None
        sender_hotkey = parts[1]
        signature = parts[2]
        receptor_uuid = parts[3]
        return (nonce, sender_hotkey, signature, receptor_uuid)

    def parse_signature(self, metadata: Dict[str, str]) -> Tuple[int, str, str, str]:
        r"""Attempts to parse a signature from the metadata"""
        signature = metadata.get("bittensor-signature")
        version = metadata.get('bittensor-version')
        if signature is None:
            raise Exception("Request signature missing")
        if int(version) < 370:
            raise Exception("Incorrect Version")
        parts = self.parse_signature_v2(signature)
        if parts is not None:
            return parts
        raise Exception("Unknown signature format")

    def check_signature(
        self,
        nonce: int,
        sender_hotkey: str,
        signature: str,
        receptor_uuid: str,
    ):
        r"""verification of signature in metadata. Uses the pubkey and nonce"""
        keypair = Keypair(ss58_address=sender_hotkey)
        # Build the expected message which was used to build the signature.
        message = f"{nonce}.{sender_hotkey}.{self.receiver_hotkey}.{receptor_uuid}"

        # Build the key which uniquely identifies the endpoint that has signed
        # the message.
        endpoint_key = f"{sender_hotkey}:{receptor_uuid}"

        if endpoint_key in self.nonces.keys():
            previous_nonce = self.nonces[endpoint_key]
            # Nonces must be strictly monotonic over time.
            if nonce <= previous_nonce:
                raise Exception("Nonce is too small")

        if not keypair.verify(message, signature):
            raise Exception("Signature mismatch")
        self.nonces[endpoint_key] = nonce

    def intercept_service(self, continuation, handler_call_details):
        r"""Authentication between bittensor nodes. Intercepts messages and checks them"""
        metadata = dict(handler_call_details.invocation_metadata)
        (
            nonce,
            sender_hotkey,
            signature,
            receptor_uuid,
        ) = self.parse_signature(metadata)

        # signature checking
        self.check_signature(
            nonce, sender_hotkey, signature, receptor_uuid
        )

        return continuation(handler_call_details)