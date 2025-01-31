from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T", bound="InfoBase")


@dataclass
class InfoBase:
    """Base dataclass for info objects."""

    @classmethod
    def from_dict(cls, decoded: dict) -> T:
        return cls(**decoded)

    @classmethod
    def list_from_dicts(cls, any_list: list[Any]) -> list[T]:
        return [cls.from_dict(any_) for any_ in any_list]
