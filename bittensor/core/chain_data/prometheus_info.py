from dataclasses import dataclass
from typing import Dict

from bittensor.utils import networking


@dataclass
class PrometheusInfo:
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
    def fix_decoded_values(cls, prometheus_info_decoded: Dict) -> "PrometheusInfo":
        """Returns a PrometheusInfo object from a prometheus_info_decoded dictionary."""
        prometheus_info_decoded["ip"] = networking.int_to_ip(
            int(prometheus_info_decoded["ip"])
        )
        return cls(**prometheus_info_decoded)
