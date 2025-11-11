"""
This module defines the `AxonInfo` class, a data structure used to represent information about an axon endpoint
in the bittensor network.
"""

from dataclasses import asdict, dataclass
from typing import Any, Union

import netaddr
from async_substrate_interface.utils import json
from bittensor.core.chain_data.info_base import InfoBase
from bittensor.utils import networking
from bittensor.utils.btlogging import logging
from bittensor.utils.registration import torch, use_torch


@dataclass
class AxonInfo(InfoBase):
    """
    The `AxonInfo` class represents information about an axon endpoint in the bittensor network. This includes
    properties such as IP address, ports, and relevant keys.

    Attributes:
        version (int): The version of the axon endpoint.
        ip (str): The IP address of the axon endpoint.
        port (int): The port number the axon endpoint uses.
        ip_type (int): The type of IP protocol (e.g., IPv4 or IPv6).
        hotkey (str): The hotkey associated with the axon endpoint.
        coldkey (str): The coldkey associated with the axon endpoint.
        protocol (int): The protocol version (default is 4).
        placeholder1 (int): Reserved field (default is 0).
        placeholder2 (int): Reserved field (default is 0).
    """

    version: int
    ip: str
    port: int
    ip_type: int
    hotkey: str
    coldkey: str
    protocol: int = 4
    placeholder1: int = 0
    placeholder2: int = 0

    @property
    def is_serving(self) -> bool:
        """True if the endpoint is serving."""
        return self.ip != "0.0.0.0"

    def ip_str(self) -> str:
        """Return the whole IP as string"""
        return networking.ip__str__(self.ip_type, self.ip, self.port)

    def __eq__(self, other: "AxonInfo"):
        if other is None or not isinstance(other, AxonInfo):
            return False

        return (
            self.version == other.version
            and self.ip == other.ip
            and self.port == other.port
            and self.ip_type == other.ip_type
            and self.coldkey == other.coldkey
            and self.hotkey == other.hotkey
        )

    def __str__(self):
        return f"AxonInfo( {self.ip_str()}, {self.hotkey}, {self.coldkey}, {self.version} )"

    def __repr__(self):
        return self.__str__()

    def to_string(self) -> str:
        """Converts the `AxonInfo` object to a string representation using JSON."""
        try:
            return json.dumps(asdict(self))
        except (TypeError, ValueError) as e:
            logging.error(f"Error converting AxonInfo to string: {e}")
            return AxonInfo(0, "", 0, 0, "", "").to_string()

    @classmethod
    def _from_dict(cls, data):
        """Returns a AxonInfo object from decoded chain data."""
        return AxonInfo(
            version=data["version"],
            ip=str(netaddr.IPAddress(data["ip"])),
            port=data["port"],
            ip_type=data["ip_type"],
            placeholder1=data["placeholder1"],
            placeholder2=data["placeholder2"],
            protocol=data["protocol"],
            hotkey=data["hotkey"],
            coldkey=data["coldkey"],
        )

    @classmethod
    def from_string(cls, json_string: str) -> "AxonInfo":
        """
        Creates an `AxonInfo` object from its string representation using JSON.

        Parameters:
            json_string: The JSON string representation of the AxonInfo object.

        Returns:
            An instance of AxonInfo created from the JSON string. If decoding fails, returns a default AxonInfo` object
                with default values.

        Raises:
            json.JSONDecodeError: If there is an error in decoding the JSON string.
            TypeError: If there is a type error when creating the AxonInfo object.
            ValueError: If there is a value error when creating the AxonInfo object.
        """
        try:
            data = json.loads(json_string)
            return cls(**data)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
        except TypeError as e:
            logging.error(f"Type error: {e}")
        except ValueError as e:
            logging.error(f"Value error: {e}")
        return AxonInfo(0, "", 0, 0, "", "")

    @classmethod
    def from_neuron_info(cls, neuron_info: dict) -> "AxonInfo":
        """
        Converts a dictionary to an `AxonInfo` object.

        Parameters:
            neuron_info: A dictionary containing the neuron information.

        Returns:
            An instance of AxonInfo created from the dictionary.
        """
        return cls(
            version=neuron_info["axon_info"]["version"],
            ip=networking.int_to_ip(int(neuron_info["axon_info"]["ip"])),
            port=neuron_info["axon_info"]["port"],
            ip_type=neuron_info["axon_info"]["ip_type"],
            hotkey=neuron_info["hotkey"],
            coldkey=neuron_info["coldkey"],
        )

    def to_parameter_dict(
        self,
    ) -> Union[dict[str, Union[int, str]], "torch.nn.ParameterDict"]:
        """Returns a torch tensor or dict of the subnet info, depending on the USE_TORCH flag set."""
        if use_torch():
            return torch.nn.ParameterDict(self.__dict__)
        else:
            return self.__dict__

    @classmethod
    def from_parameter_dict(
        cls, parameter_dict: Union[dict[str, Any], "torch.nn.ParameterDict"]
    ) -> "AxonInfo":
        """Returns an axon_info object from a torch parameter_dict or a parameter dict."""
        if use_torch():
            return cls(**dict(parameter_dict))
        else:
            return cls(**parameter_dict)
