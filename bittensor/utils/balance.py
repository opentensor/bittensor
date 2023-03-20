# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
from typing import Union

import bittensor

class Balance:
    """
    Represents the bittensor balance of the wallet, stored as rao (int)
    The Balance object is immutable, and can be used as a number or as a string
    Can only guarantee that the balance is accurate to 9 decimal places (tao)

    Note: In operations between Balance and int/float, the other value is assumed to be in rao
    """

    unit: str = bittensor.__tao_symbol__ # This is the tao unit
    rao_unit: str = bittensor.__rao_symbol__ # This is the rao unit
    rao: int
    tao: float

    def __init__(self, balance: Union[int, float]):
        if isinstance(balance, int):
            self.rao = balance
        elif isinstance(balance, float):
            # Assume tao value for the float
            self.rao = int(balance * pow(10, 9))
        else:
            raise TypeError("balance must be an int (rao) or a float (tao)")

    @property
    def tao(self):
        return self.rao / pow(10, 9)

    def __int__(self):
        return self.rao

    def __float__(self):
        return self.tao

    def __str__(self):
        return f"{self.unit}{float(self.tao):,.9f}"

    def __rich__(self):
        return "[green]{}[/green][green]{}[/green][green].[/green][dim green]{}[/dim green]".format(
            self.unit,
            format(float(self.tao), "f").split(".")[0],
            format(float(self.tao), "f").split(".")[1],
        )

    def __str_rao__(self):
        return f"{self.rao_unit}{int(self.rao)}"

    def __rich_rao__(self):
        return f"[green]{self.rao_unit}{int(self.rao)}[/green]"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: Union[int, float, "Balance"]):
        if other is None:
            return False
            
        if hasattr(other, "rao"):
            return self.rao == other.rao
        else:
            try:
                # Attempt to cast to int from rao
                other_rao = int(other)
                return self.rao == other_rao
            except (TypeError, ValueError):
                raise NotImplementedError("Unsupported type")

    def __ne__(self, other: Union[int, float, "Balance"]):
        return not self == other

    def __gt__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "rao"):
            return self.rao > other.rao
        else:
            try:
                # Attempt to cast to int from rao
                other_rao = int(other)
                return self.rao > other_rao
            except ValueError:
                raise NotImplementedError("Unsupported type")

    def __lt__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "rao"):
            return self.rao < other.rao
        else:
            try:
                # Attempt to cast to int from rao
                other_rao = int(other)
                return self.rao < other_rao
            except ValueError:
                raise NotImplementedError("Unsupported type")

    def __le__(self, other: Union[int, float, "Balance"]):
        try:
            return self < other or self == other
        except (TypeError):
            raise NotImplementedError("Unsupported type")

    def __ge__(self, other: Union[int, float, "Balance"]):
        try:
            return self > other or self == other
        except (TypeError):
            raise NotImplementedError("Unsupported type")

    def __add__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "rao"):
            return Balance.from_rao(int(self.rao + other.rao))
        else:
            try:
                # Attempt to cast to int from rao
                return Balance.from_rao(int(self.rao + other))
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __radd__(self, other: Union[int, float, "Balance"]):
        try:
            return self + other
        except (TypeError):
            raise NotImplementedError("Unsupported type")

    def __sub__(self, other: Union[int, float, "Balance"]):
        try:
            return self + -other
        except (TypeError):
            raise NotImplementedError("Unsupported type")

    def __rsub__(self, other: Union[int, float, "Balance"]):
        try:
            return -self + other
        except (TypeError):
            raise NotImplementedError("Unsupported type")

    def __mul__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "rao"):
            return Balance.from_rao(int(self.rao * other.rao))
        else:
            try:
                # Attempt to cast to int from rao
                return Balance.from_rao(int(self.rao * other))
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __rmul__(self, other: Union[int, float, "Balance"]):
        return self * other

    def __truediv__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "rao"):
            return Balance.from_rao(int(self.rao / other.rao))
        else:
            try:
                # Attempt to cast to int from rao
                return Balance.from_rao(int(self.rao / other))
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __rtruediv__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "rao"):
            return Balance.from_rao(int(other.rao / self.rao))
        else:
            try:
                # Attempt to cast to int from rao
                return Balance.from_rao(int(other / self.rao))
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __floordiv__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "rao"):
            return Balance.from_rao(int(self.tao // other.tao))
        else:
            try:
                # Attempt to cast to int from rao
                return Balance.from_rao(int(self.rao // other))
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __rfloordiv__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "rao"):
            return Balance.from_rao(int(other.rao // self.rao))
        else:
            try:
                # Attempt to cast to int from rao
                return Balance.from_rao(int(other // self.rao))
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __int__(self) -> int:
        return self.rao

    def __float__(self) -> float:
        return self.tao

    def __nonzero__(self) -> bool:
        return bool(self.rao)

    def __neg__(self):
        return Balance.from_rao(-self.rao)

    def __pos__(self):
        return Balance.from_rao(self.rao)

    def __abs__(self):
        return Balance.from_rao(abs(self.rao))

    @staticmethod
    def from_float(amount: float):
        """Given tao (float), return Balance object with rao(int) and tao(float), where rao = int(tao*pow(10,9))"""
        rao = int(amount * pow(10, 9))
        return Balance(rao)

    @staticmethod
    def from_tao(amount: float):
        """Given tao (float), return Balance object with rao(int) and tao(float), where rao = int(tao*pow(10,9))"""
        rao = int(amount * pow(10, 9))
        return Balance(rao)

    @staticmethod
    def from_rao(amount: int):
        """Given rao (int), return Balance object with rao(int) and tao(float), where rao = int(tao*pow(10,9))"""
        return Balance(amount)
