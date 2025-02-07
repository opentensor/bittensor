from dataclasses import dataclass, fields
from typing import Any, TypeVar

from bittensor.core.errors import SubstrateRequestException

T = TypeVar("T", bound="InfoBase")


@dataclass
class InfoBase:
    """Base dataclass for info objects."""

    @classmethod
    def from_dict(cls, decoded: dict) -> T:
        try:
            class_fields = {f.name for f in fields(cls)}
            extra_keys = decoded.keys() - class_fields
            instance = cls._from_dict(
                {k: v for k, v in decoded.items() if k in class_fields}
            )
            [setattr(instance, k, decoded[k]) for k in extra_keys]
            return instance
        except KeyError as e:
            raise SubstrateRequestException(
                f"The {cls} structure is missing {e} from the chain.",
            )

    @classmethod
    def list_from_dicts(cls, any_list: list[Any]) -> list[T]:
        return [cls.from_dict(any_) for any_ in any_list]

    @classmethod
    def _from_dict(cls, decoded: dict) -> T:
        return cls(**decoded)
