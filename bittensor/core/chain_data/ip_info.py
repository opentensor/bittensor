from dataclasses import dataclass
from typing import Any, Union

from bittensor.utils import networking as net
from bittensor.utils.registration import torch, use_torch


@dataclass
class IPInfo:
    """
    Dataclass representing IP information.

    Attributes:
        ip: The IP address as a string.
        ip_type: The type of the IP address (e.g., IPv4, IPv6).
        protocol: The protocol associated with the IP (e.g., TCP, UDP).
    """

    ip: str
    ip_type: int
    protocol: int

    def encode(self) -> dict[str, Any]:
        """Returns a dictionary of the IPInfo object that can be encoded."""
        return {
            "ip": net.ip_to_int(
                self.ip
            ),  # IP type and protocol are encoded together as a u8
            "ip_type_and_protocol": ((self.ip_type << 4) + self.protocol) & 0xFF,
        }

    @classmethod
    def _from_dict(cls, decoded: dict) -> "IPInfo":
        """Returns a IPInfo object from decoded chain data."""
        return IPInfo(
            ip_type=decoded["ip_type_and_protocol"] >> 4,
            ip=net.int_to_ip(decoded["ip"]),
            protocol=decoded["ip_type_and_protocol"] & 0xF,
        )

    def to_parameter_dict(
        self,
    ) -> Union[dict[str, Union[str, int]], "torch.nn.ParameterDict"]:
        """Returns a torch tensor or dict of the subnet IP info."""
        if use_torch():
            return torch.nn.ParameterDict(self.__dict__)
        else:
            return self.__dict__

    @classmethod
    def from_parameter_dict(
        cls, parameter_dict: Union[dict[str, Any], "torch.nn.ParameterDict"]
    ) -> "IPInfo":
        """Creates a IPInfo instance from a parameter dictionary."""
        if use_torch():
            return cls(**dict(parameter_dict))
        else:
            return cls(**parameter_dict)
