from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar

import munch

T = TypeVar("T", bound="InfoBase")


@dataclass
class InfoBase:
    """Base dataclass for info objects."""

    @abstractmethod
    def _fix_decoded(self, decoded: Any) -> T:
        raise NotImplementedError(
            "This is an abstract method and must be implemented in a subclass."
        )

    @classmethod
    def from_any(cls, any_: Any) -> T:
        any_ = munch.munchify(any_)
        return cls._fix_decoded(any_)

    @classmethod
    def list_from_any(cls, any_list: list[Any]) -> list[T]:
        return [cls.from_any(any_) for any_ in any_list]
