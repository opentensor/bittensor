from dataclasses import dataclass

import netaddr

from bittensor.core.chain_data.info_base import InfoBase


@dataclass
class PrometheusInfo(InfoBase):
    """
    Dataclass representing information related to Prometheus.

    Attributes:
        block (int): The block number associated with the Prometheus data.
        version (int): The version of the Prometheus data.
        ip (str): The IP address associated with Prometheus.
        port (int): The port number for Prometheus.
        ip_type (int): The type of IP address (e.g., IPv4, IPv6).
    """

    block: int
    version: int
    ip: str
    port: int
    ip_type: int

    @classmethod
    def _from_dict(cls, data):
        """Returns a PrometheusInfo object from decoded chain data."""
        return cls(
            block=data["block"],
            ip_type=data["ip_type"],
            ip=str(netaddr.IPAddress(data["ip"])),
            port=data["port"],
            version=data["version"],
        )
