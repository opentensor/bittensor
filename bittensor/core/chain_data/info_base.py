from dataclasses import dataclass
from typing import Any, Self

from bittensor.core.errors import SubstrateRequestException


@dataclass
class InfoBase:
    """Base dataclass for info objects."""

    @classmethod
    def from_dict(cls, decoded: dict) -> Self:
        try:
            return cls._from_dict(decoded)
        except KeyError as e:
            raise SubstrateRequestException(
                f"The {cls} structure is missing {e} from the chain.",
            )

    @classmethod
    def list_from_dicts(cls, any_list: list[Any]) -> list[Self]:
        return [cls.from_dict(any_) for any_ in any_list]

    @classmethod
    def _from_dict(cls, decoded: dict) -> Self:
        return cls(**decoded)
