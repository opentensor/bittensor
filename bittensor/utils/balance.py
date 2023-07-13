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
    Represents the bittensor balance of the wallet, stored as rao (int).
    This class provides a way to interact with balances in two different units: rao and tao.
    It provides methods to convert between these units, as well as to perform arithmetic and comparison operations.
    
    Attributes:
        unit: A string representing the symbol for the tao unit.
        rao_unit: A string representing the symbol for the rao unit.
        rao: An integer that stores the balance in rao units.
        tao: A float property that gives the balance in tao units.
    """

    unit: str = bittensor.__tao_symbol__ # This is the tao unit
    rao_unit: str = bittensor.__rao_symbol__ # This is the rao unit
    rao: int
    tao: float

    def __init__(self, balance: Union[int, float]):
        """
        Initialize a Balance object. If balance is an int, it's assumed to be in rao.
        If balance is a float, it's assumed to be in tao.
        
        Args:
            balance: The initial balance, in either rao (if an int) or tao (if a float).
        """
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
        """
        Convert the Balance object to an int. The resulting value is in rao.
        """
        return self.rao

    def __float__(self):
        """
        Convert the Balance object to a float. The resulting value is in tao.
        """
        return self.tao

    def __str__(self):
        """
        Returns the Balance object as a string in the format "symbolvalue", where the value is in tao.
        """
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
        """
        Given tao (float), return Balance object with rao(int) and tao(float), where rao = int(tao*pow(10,9))
        Args:
            amount: The amount in tao.

        Returns:
            A Balance object representing the given amount.
        """
        rao = int(amount * pow(10, 9))
        return Balance(rao)

    @staticmethod
    def from_tao(amount: float):
        """
        Given tao (float), return Balance object with rao(int) and tao(float), where rao = int(tao*pow(10,9))

        Args:
            amount: The amount in tao.

        Returns:
            A Balance object representing the given amount.
        """
        rao = int(amount * pow(10, 9))
        return Balance(rao)

    @staticmethod
    def from_rao(amount: int):
        """
        Given rao (int), return Balance object with rao(int) and tao(float), where rao = int(tao*pow(10,9))
        
        Args:
            amount: The amount in rao.

        Returns:
            A Balance object representing the given amount.
        """
        return Balance(amount)


#########
# Tests #
########

import unittest

import pytest
from hypothesis import given
from hypothesis import strategies as st

class CLOSE_IN_VALUE():
    value: Union[float, int, Balance]
    tolerance: Union[float, int, Balance]

    def __init__(self, value: Union[float, int, Balance], tolerance: Union[float, int, Balance] = 0.0) -> None:
        self.value = value
        self.tolerance = tolerance

    def __eq__(self, __o: Union[float, int, Balance]) -> bool:
        # True if __o \in [value - tolerance, value + tolerance]
        # or if value \in [__o - tolerance, __o + tolerance]
        return ((self.value - self.tolerance) <= __o and __o <= (self.value + self.tolerance)) or \
                ((__o - self.tolerance) <= self.value and self.value <= (__o + self.tolerance))


"""
Test the Balance class
"""
valid_tao_numbers_strategy = st.one_of(st.integers(max_value=21_000_000, min_value=-21_000_000), st.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False, max_value=21_000_000.00, min_value=-21_000_000.00))

def remove_zero_filter(x):
    """Remove zero and rounded to zero from the list of valid numbers"""
    return int(x * pow(10, 9)) != 0

class TestBalance(unittest.TestCase):
    @given(balance=valid_tao_numbers_strategy)
    def test_balance_init(self, balance: Union[int, float]):
        """
        Test the initialization of the Balance object.
        """
        balance_ = Balance(balance)
        if isinstance(balance, int):
            assert balance_.rao == balance
        elif isinstance(balance, float):
            assert balance_.tao == CLOSE_IN_VALUE(balance, 0.00001)

    @given(balance=valid_tao_numbers_strategy, balance2=valid_tao_numbers_strategy)
    def test_balance_add(self, balance: Union[int, float], balance2: Union[int, float]):
        """
        Test the addition of two Balance objects.
        """
        balance_ = Balance(balance)
        balance2_ = Balance(balance2)
        rao_: int
        rao2_: int
        if isinstance(balance, int):
            rao_ = balance
        elif isinstance(balance, float):
            rao_ = int(balance * pow(10, 9))
        if isinstance(balance2, int):
            rao2_ = balance2
        elif isinstance(balance2, float):
            rao2_ = int(balance2 * pow(10, 9))

        sum_ = balance_ + balance2_
        assert isinstance(sum_, Balance)
        assert CLOSE_IN_VALUE(sum_.rao, 5) == rao_ + rao2_

    @given(balance=valid_tao_numbers_strategy, balance2=valid_tao_numbers_strategy)
    def test_balance_add_other_not_balance(self, balance: Union[int, float], balance2: Union[int, float]):
        """
        Test the addition of a Balance object and a non-Balance object.
        """
        balance_ = Balance(balance)
        balance2_ = balance2
        rao_: int
        rao2_: int
        if isinstance(balance, int):
            rao_ = balance
        elif isinstance(balance, float):
            rao_ = int(balance * pow(10, 9))
        # convert balance2 to rao. Assume balance2 was rao
        rao2_ = int(balance2)

        sum_ = balance_ + balance2_
        assert isinstance(sum_, Balance)
        assert CLOSE_IN_VALUE(sum_.rao, 5) == rao_ + rao2_

    @given(balance=valid_tao_numbers_strategy)
    def test_balance_eq_other_not_balance(self, balance: Union[int, float]):
        """
        Test the equality of a Balance object and a non-Balance object.
        """
        balance_ = Balance(balance)
        rao2_: int
        # convert balance2 to rao. This assumes balance2 is a rao value
        rao2_ = int(balance_.rao)

        self.assertEqual(CLOSE_IN_VALUE(rao2_, 5), balance_, msg=f"Balance {balance_} is not equal to {rao2_}")

    @given(balance=valid_tao_numbers_strategy, balance2=valid_tao_numbers_strategy)
    def test_balance_radd_other_not_balance(self, balance: Union[int, float], balance2: Union[int, float]):
        """
        Test the right addition (radd) of a Balance object and a non-Balance object.
        """
        balance_ = Balance(balance)
        balance2_ = balance2
        rao_: int
        rao2_: int
        if isinstance(balance, int):
            rao_ = balance
        elif isinstance(balance, float):
            rao_ = int(balance * pow(10, 9))
        # assume balance2 is a rao value
        rao2_ = int(balance2)

        sum_ =  balance2_ + balance_ # This is an radd
        assert isinstance(sum_, Balance)
        assert CLOSE_IN_VALUE(sum_.rao, 5) == rao2_ + rao_

    @given(balance=valid_tao_numbers_strategy, balance2=valid_tao_numbers_strategy)
    def test_balance_sub(self, balance: Union[int, float], balance2: Union[int, float]):
        """
        Test the subtraction of two Balance objects.
        """
        balance_ = Balance(balance)
        balance2_ = Balance(balance2)
        rao_: int
        rao2_: int
        if isinstance(balance, int):
            rao_ = balance
        elif isinstance(balance, float):
            rao_ = int(balance * pow(10, 9))
        if isinstance(balance2, int):
            rao2_ = balance2
        elif isinstance(balance2, float):
            rao2_ = int(balance2 * pow(10, 9))

        diff_ = balance_ - balance2_
        assert isinstance(diff_, Balance)
        assert CLOSE_IN_VALUE(diff_.rao, 5) == rao_ - rao2_

    @given(balance=valid_tao_numbers_strategy, balance2=valid_tao_numbers_strategy)
    def test_balance_sub_other_not_balance(self, balance: Union[int, float], balance2: Union[int, float]):
        """
        Test the subtraction of a Balance object and a non-Balance object.
        """
        balance_ = Balance(balance)
        balance2_ = balance2
        rao_: int
        rao2_: int
        if isinstance(balance, int):
            rao_ = balance
        elif isinstance(balance, float):
            rao_ = int(balance * pow(10, 9))
        # assume balance2 is a rao value
        rao2_ = int(balance2)

        diff_ =  balance_ - balance2_
        assert isinstance(diff_, Balance)
        assert CLOSE_IN_VALUE(diff_.rao, 5) == rao_ - rao2_

    @given(balance=valid_tao_numbers_strategy, balance2=valid_tao_numbers_strategy)
    def test_balance_rsub_other_not_balance(self, balance: Union[int, float], balance2: Union[int, float]):
        """
        Test the right subtraction (rsub) of a Balance object and a non-Balance object.
        """
        balance_ = Balance(balance)
        balance2_ = balance2
        rao_: int
        rao2_: int
        if isinstance(balance, int):
            rao_ = balance
        elif isinstance(balance, float):
            rao_ = int(balance * pow(10, 9))
        # assume balance2 is a rao value
        rao2_ = int(balance2)

        diff_ =  balance2_ - balance_ # This is an rsub
        assert isinstance(diff_, Balance)
        assert CLOSE_IN_VALUE(diff_.rao, 5) == rao2_ - rao_

    @given(balance=valid_tao_numbers_strategy, balance2=valid_tao_numbers_strategy)
    def test_balance_mul(self, balance: Union[int, float], balance2: Union[int, float]):
        """
        Test the multiplication of two Balance objects.
        """
        balance_ = Balance(balance)
        balance2_ = Balance(balance2)
        rao_: int
        if isinstance(balance, int):
            rao_ = balance
        elif isinstance(balance, float):
            rao_ = int(balance * pow(10, 9))
        if isinstance(balance2, int):
            rao2_ = balance2
        elif isinstance(balance2, float):
            rao2_ = int(balance2 * pow(10, 9))

        prod_ = balance_ * balance2_
        assert isinstance(prod_, Balance)
        self.assertAlmostEqual(prod_.rao, rao_ * rao2_, 9, msg="{} * {} == {} != {} * {} == {}".format(balance_, balance2_, prod_.rao, rao_, balance2, rao_ * balance2))

    @given(balance=valid_tao_numbers_strategy, balance2=valid_tao_numbers_strategy)
    def test_balance_mul_other_not_balance(self, balance: Union[int, float], balance2: Union[int, float]):
        """
        Test the multiplication of a Balance object and a non-Balance object.
        """
        balance_ = Balance(balance)
        balance2_ = balance2
        rao_: int
        if isinstance(balance, int):
            rao_ = balance
        elif isinstance(balance, float):
            rao_ = int(balance * pow(10, 9))

        prod_ = balance_ * balance2_
        assert isinstance(prod_, Balance)
        self.assertAlmostEqual(prod_.rao, int(rao_ * balance2), delta=20)

    @given(balance=valid_tao_numbers_strategy, balance2=valid_tao_numbers_strategy)
    def test_balance_rmul_other_not_balance(self, balance: Union[int, float], balance2: Union[int, float]):
        """
        Test the right multiplication (rmul) of a Balance object and a non-Balance object.
        """
        balance_ = Balance(balance)
        balance2_ = balance2
        rao_: int
        if isinstance(balance, int):
            rao_ = balance
        elif isinstance(balance, float):
            rao_ = int(balance * pow(10, 9))

        prod_ =  balance2_ * balance_ # This is an rmul
        assert isinstance(prod_, Balance)
        self.assertAlmostEqual(prod_.rao, int(balance2 * rao_), delta=20, msg=f"{balance2_} * {balance_} = {prod_} != {balance2} * {rao_} == {balance2 * rao_}")

    @given(balance=valid_tao_numbers_strategy, balance2=valid_tao_numbers_strategy.filter(remove_zero_filter)) # Avoid zero division
    def test_balance_truediv(self, balance: Union[int, float], balance2: Union[int, float]):
        """
        Test the true division (/) of two Balance objects.
        """
        balance_ = Balance(balance)
        balance2_ = Balance(balance2)
        rao_: int
        rao2_: int
        if isinstance(balance, int):
            rao_ = balance
        elif isinstance(balance, float):
            rao_ = int(balance * pow(10, 9))
        if isinstance(balance2, int):
            rao2_ = balance2
        elif isinstance(balance2, float):
            rao2_ = int(balance2 * pow(10, 9))

        quot_ = balance_ / balance2_
        assert isinstance(quot_, Balance)
        self.assertAlmostEqual(quot_.rao, int(rao_ / rao2_), delta=2, msg=f"{balance_} / {balance2_} = {quot_} != {rao_} / {rao2_} == {int(rao_ / rao2_)}")

    @given(balance=valid_tao_numbers_strategy,  balance2=valid_tao_numbers_strategy.filter(remove_zero_filter))
    def test_balance_truediv_other_not_balance(self, balance: Union[int, float], balance2: Union[int, float]):
        """
        Test the true division (/) of a Balance object and a non-Balance object.
        """
        balance_ = Balance(balance)
        balance2_ = balance2
        rao_: int
        rao2_: int
        if isinstance(balance, int):
            rao_ = balance
        elif isinstance(balance, float):
            rao_ = int(balance * pow(10, 9))
        # assume balance2 is a rao value
        rao2_ = balance2

        quot_ = balance_ / balance2_
        self.assertAlmostEqual(quot_.rao, int(rao_ / rao2_), delta=10, msg="{} / {} = {} != {}".format(balance_, balance2_, quot_.rao, int(rao_ / rao2_)))

    @given(balance=valid_tao_numbers_strategy.filter(remove_zero_filter), balance2=valid_tao_numbers_strategy) # This is a filter to avoid division by zero
    def test_balance_rtruediv_other_not_balance(self, balance: Union[int, float], balance2: Union[int, float]):
        """
        Test the right true division (rtruediv) of a Balance object and a non-Balance object.
        """
        balance_ = Balance(balance)
        balance2_ = balance2
        rao_: int
        rao2_: int
        if isinstance(balance, int):
            rao_ = balance
        elif isinstance(balance, float):
            rao_ = int(balance * pow(10, 9))
        # assume balance2 is a rao value
        rao2_ = balance2

        quot_ =  balance2_ / balance_ # This is an rtruediv
        assert isinstance(quot_, Balance)
        self.assertAlmostEqual(quot_.rao, int(rao2_ / rao_), delta=5, msg="{} / {} = {}".format(balance2_, balance_, quot_))

    @given(balance=valid_tao_numbers_strategy, balance2=valid_tao_numbers_strategy.filter(remove_zero_filter)) # Avoid zero division
    def test_balance_floordiv(self, balance: Union[int, float], balance2: Union[int, float]):
        """
        Test the floor division (//) of two Balance objects.
        """
        balance_ = Balance(balance)
        balance2_ = Balance(balance2)
        rao_: int
        rao2_: int
        if isinstance(balance, int):
            rao_ = balance
        elif isinstance(balance, float):
            rao_ = int(balance * pow(10, 9))
        if isinstance(balance2, int):
            rao2_ = balance2
        elif isinstance(balance2, float):
            rao2_ = int(balance2 * pow(10, 9))

        quot_ = balance_ // balance2_
        assert isinstance(quot_, Balance)
        assert CLOSE_IN_VALUE(quot_.rao, 5) == rao_ // rao2_

    @given(balance=valid_tao_numbers_strategy,  balance2=valid_tao_numbers_strategy.filter(remove_zero_filter))
    def test_balance_floordiv_other_not_balance(self, balance: Union[int, float], balance2: Union[int, float]):
        """
        Test the floor division (//) of a Balance object and a non-Balance object.
        """
        balance_ = Balance(balance)
        balance2_ = balance2
        rao_: int
        rao2_: int
        if isinstance(balance, int):
            rao_ = balance
        elif isinstance(balance, float):
            rao_ = int(balance * pow(10, 9))
        # assume balance2 is a rao value
        rao2_ = balance2

        quot_ = balance_ // balance2_
        assert isinstance(quot_, Balance)
        self.assertAlmostEqual(quot_.rao, rao_ // rao2_, delta=5, msg="{} // {} = {} != {}".format(balance_, balance2_, quot_.rao, rao_ // rao2_))

    @given(balance=valid_tao_numbers_strategy.filter(remove_zero_filter), balance2=valid_tao_numbers_strategy) # This is a filter to avoid division by zero
    def test_balance_rfloordiv_other_not_balance(self, balance: Union[int, float], balance2: Union[int, float]):
        """
        Test the right floor division (rfloordiv) of a Balance object and a non-Balance object.
        """
        balance_ = Balance(balance)
        balance2_ = balance2
        rao_: int
        rao2_: int
        if isinstance(balance, int):
            rao_ = balance
        elif isinstance(balance, float):
            rao_ = int(balance * pow(10, 9))
        # assume balance2 is a rao value
        rao2_ = balance2

        quot_ =  balance2_ // balance_ # This is an rfloordiv
        assert isinstance(quot_, Balance)
        self.assertAlmostEqual(quot_.rao, rao2_ // rao_, delta=5)

    @given(balance=valid_tao_numbers_strategy)
    def test_balance_not_eq_none(self, balance: Union[int, float]):
        """
        Test the inequality (!=) of a Balance object and None.
        """
        balance_ = Balance(balance)
        assert not balance_ == None

    @given(balance=valid_tao_numbers_strategy)
    def test_balance_neq_none(self, balance: Union[int, float]):
        """
        Test the inequality (!=) of a Balance object and None.
        """
        balance_ = Balance(balance)
        assert balance_ != None

    def test_balance_init_from_invalid_value(self):
        """
        Test the initialization of a Balance object with an invalid value.
        """
        with pytest.raises(TypeError):
            Balance('invalid not a number')

    @given(balance=valid_tao_numbers_strategy)
    def test_balance_add_invalid_type(self, balance: Union[int, float]):
        """
        Test the addition of a Balance object with an invalid type.
        """
        balance_ = Balance(balance)
        with pytest.raises(NotImplementedError):
            _ = balance_ + ""

    @given(balance=valid_tao_numbers_strategy)
    def test_balance_sub_invalid_type(self, balance: Union[int, float]):
        """
        Test the subtraction of a Balance object with an invalid type.
        """
        balance_ = Balance(balance)
        with pytest.raises(NotImplementedError):
            _ = balance_ - ""

    @given(balance=valid_tao_numbers_strategy)
    def test_balance_div_invalid_type(self, balance: Union[int, float]):
        """
        Test the division of a Balance object with an invalid type.
        """
        balance_ = Balance(balance)
        with pytest.raises(NotImplementedError):
            _ = balance_ / ""

    @given(balance=valid_tao_numbers_strategy)
    def test_balance_mul_invalid_type(self, balance: Union[int, float]):
        """
        Test the multiplication of a Balance object with an invalid type.
        """
        balance_ = Balance(balance)
        with pytest.raises(NotImplementedError):
            _ = balance_ * ""

    @given(balance=valid_tao_numbers_strategy)
    def test_balance_eq_invalid_type(self, balance: Union[int, float]):
        """
        Test the equality of a Balance object with an invalid type.
        """
        balance_ = Balance(balance)
        with pytest.raises(NotImplementedError):
            balance_ == ""
