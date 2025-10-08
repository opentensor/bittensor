from dataclasses import dataclass
from typing import Optional
from bittensor.utils import Certificate, networking as net
from bittensor.core.types import AxonServeCallParams
from bittensor.core.settings import version_as_int


@dataclass
class ServingParams:
    @classmethod
    def serve_axon_and_tls(
        cls,
        hotkey_ss58: str,
        coldkey_ss58: str,
        netuid: int,
        ip: str,
        port: int,
        protocol: int,
        placeholder1: int,
        placeholder2: int,
        certificate: Optional[Certificate] = None,
    ) -> AxonServeCallParams:
        """Returns the parameters for the `root_register`."""
        return AxonServeCallParams(
            **{
                "hotkey": hotkey_ss58,
                "coldkey": coldkey_ss58,
                "netuid": netuid,
                "ip": net.ip_to_int(ip),
                "port": port,
                "protocol": protocol,
                "certificate": certificate,
                "ip_type": net.ip_version(ip),
                "version": version_as_int,
                "placeholder1": placeholder1,
                "placeholder2": placeholder2,
            }
        )

    @classmethod
    def set_commitment(cls, netuid: int, info_fields: list) -> dict:
        """Returns the parameters for the `set_commitment`."""
        return {"netuid": netuid, "info": {"fields": [info_fields]}}
