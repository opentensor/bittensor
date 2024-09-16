from dataclasses import dataclass
from typing import Optional, Any, Union

from bittensor.core.chain_data.utils import from_scale_encoding, ChainDataType
from bittensor.utils import networking as net
from bittensor.utils.registration import torch, use_torch


@dataclass
class IPInfo:
    """
    Dataclass representing IP information.

    Attributes:
        ip (str): The IP address as a string.
        ip_type (int): The type of the IP address (e.g., IPv4, IPv6).
        protocol (int): The protocol associated with the IP (e.g., TCP, UDP).
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
    def from_vec_u8(cls, vec_u8: list[int]) -> Optional["IPInfo"]:
        """Returns a IPInfo object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None

        decoded = from_scale_encoding(vec_u8, ChainDataType.IPInfo)
        if decoded is None:
            return None

        return IPInfo.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: list[int]) -> list["IPInfo"]:
        """Returns a list of IPInfo objects from a ``vec_u8``."""
        decoded = from_scale_encoding(vec_u8, ChainDataType.IPInfo, is_vec=True)

        if decoded is None:
            return []

        return [IPInfo.fix_decoded_values(d) for d in decoded]

    @classmethod
    def fix_decoded_values(cls, decoded: dict) -> "IPInfo":
        """Returns a SubnetInfo object from a decoded IPInfo dictionary."""
        return IPInfo(
            ip=net.int_to_ip(decoded["ip"]),
            ip_type=decoded["ip_type_and_protocol"] >> 4,
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
