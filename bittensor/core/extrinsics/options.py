import dataclasses
from typing import Union


@dataclasses.dataclass(frozen=True)
class ExtrinsicEra:
    """
    Defines blocks for which the transaction should be valid.

    Attributes:
        period (int): Length (in blocks) for which the transaction should be valid.
    """

    period: int


ExtrinsicEraTypes = Union[
    dict[str, int],
    ExtrinsicEra,
]


DEFAULT_SET_WEIGHTS_EXTRINSIC_ERA = ExtrinsicEra(
    period=5,
)
