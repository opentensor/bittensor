from typing import TypedDict, Optional

from bittensor.utils import Certificate


class AxonServeCallParams(TypedDict):
    """Axon serve chain call parameters."""

    version: int
    ip: int
    port: int
    ip_type: int
    netuid: int
    certificate: Optional[Certificate]


class PrometheusServeCallParams(TypedDict):
    """Prometheus serve chain call parameters."""

    version: int
    ip: int
    port: int
    ip_type: int
    netuid: int


class ParamWithTypes(TypedDict):
    name: str  # Name of the parameter.
    type: str  # ScaleType string of the parameter.
