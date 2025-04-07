import warnings
from typing import Union, TypedDict, Optional

from scalecodec import ScaleType

from bittensor.core import settings


def _check_currencies(self, other):
    """Checks that Balance objects have the same netuids to perform arithmetic operations.

    A warning is raised if the netuids differ.

    Example:
        >>> balance1 = Balance.from_rao(1000).set_unit(12)
        >>> balance2 = Balance.from_rao(500).set_unit(12)
        >>> balance1 + balance2  # No warning

        >>> balance3 = Balance.from_rao(200).set_unit(15)
        >>> balance1 + balance3  # Raises DeprecationWarning

    In this example:
        - `from_rao` creates a Balance instance from the amount in rao (smallest unit).
        - `set_unit(12)` sets the unit to correspond to subnet 12 (i.e., Alpha from netuid 12).
    """
    if self.netuid != other.netuid:
        warnings.simplefilter("default", DeprecationWarning)
        warnings.warn(
            "Balance objects must have the same netuid (Alpha currency) to perform arithmetic operations.\n"
            f"First balance is `{self}`.  Second balance is `{other}`.\n\n"
            "To create a Balance instance with the correct netuid, use:\n"
            "Balance.from_rao(1000).set_unit(12)  # 1000 rao in subnet 12",
            category=DeprecationWarning,
            stacklevel=2,
        )


class Balance:
    """
    Represents the bittensor balance of the wallet, stored as rao (int).
    This class provides a way to interact with balances in two different units: rao and tao.
    It provides methods to convert between these units, as well as to perform arithmetic and comparison operations.

    Attributes:
        unit (str): A string representing the symbol for the tao unit.
        rao_unit (str): A string representing the symbol for the rao unit.
        rao (int): An integer that stores the balance in rao units.
        tao (float): A float property that gives the balance in tao units.
    """

    unit: str = settings.TAO_SYMBOL  # This is the tao unit
    rao_unit: str = settings.RAO_SYMBOL  # This is the rao unit
    rao: int
    tao: float
    netuid: int = 0

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
        """Convert the Balance object to an int. The resulting value is in rao."""
        return self.rao

    def __float__(self):
        """Convert the Balance object to a float. The resulting value is in tao."""
        return self.tao

    def __str__(self):
        """
        Returns the Balance object as a string in the format "symbolvalue", where the value is in tao.
        """
        if self.unit == units[0]:
            return f"{self.unit}{float(self.tao):,.9f}"
        else:
            return f"\u200e{float(self.tao):,.9f}{self.unit}\u200e"

    def __rich__(self):
        int_tao, fract_tao = format(float(self.tao), "f").split(".")
        return f"[green]{self.unit}{int_tao}.{fract_tao}[/green]"

    def __str_rao__(self):
        return f"{self.rao_unit}{int(self.rao)}"

    def __rich_rao__(self):
        return f"[green]{self.rao_unit}{int(self.rao)}[/green]"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: Union[int, float, "Balance"]):
        if other is None:
            return False

        if isinstance(other, Balance):
            _check_currencies(self, other)
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
        if isinstance(other, Balance):
            _check_currencies(self, other)
            return self.rao > other.rao
        else:
            try:
                # Attempt to cast to int from rao
                other_rao = int(other)
                return self.rao > other_rao
            except ValueError:
                raise NotImplementedError("Unsupported type")

    def __lt__(self, other: Union[int, float, "Balance"]):
        if isinstance(other, Balance):
            _check_currencies(self, other)
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
            if isinstance(other, Balance):
                _check_currencies(self, other)
            return self < other or self == other
        except TypeError:
            raise NotImplementedError("Unsupported type")

    def __ge__(self, other: Union[int, float, "Balance"]):
        try:
            if isinstance(other, Balance):
                _check_currencies(self, other)
            return self > other or self == other
        except TypeError:
            raise NotImplementedError("Unsupported type")

    def __add__(self, other: Union[int, float, "Balance"]):
        if isinstance(other, Balance):
            _check_currencies(self, other)
            return Balance.from_rao(int(self.rao + other.rao)).set_unit(self.netuid)
        else:
            try:
                # Attempt to cast to int from rao
                return Balance.from_rao(int(self.rao + other)).set_unit(self.netuid)
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __radd__(self, other: Union[int, float, "Balance"]):
        try:
            if isinstance(other, Balance):
                _check_currencies(self, other)
            return self + other
        except TypeError:
            raise NotImplementedError("Unsupported type")

    def __sub__(self, other: Union[int, float, "Balance"]):
        try:
            if isinstance(other, Balance):
                _check_currencies(self, other)
            return self + -other
        except TypeError:
            raise NotImplementedError("Unsupported type")

    def __rsub__(self, other: Union[int, float, "Balance"]):
        try:
            if isinstance(other, Balance):
                _check_currencies(self, other)
            return -self + other
        except TypeError:
            raise NotImplementedError("Unsupported type")

    def __mul__(self, other: Union[int, float, "Balance"]):
        if isinstance(other, Balance):
            _check_currencies(self, other)
            return Balance.from_rao(int(self.rao * other.rao)).set_unit(self.netuid)
        else:
            try:
                # Attempt to cast to int from rao
                return Balance.from_rao(int(self.rao * other)).set_unit(self.netuid)
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __rmul__(self, other: Union[int, float, "Balance"]):
        if isinstance(other, Balance):
            _check_currencies(self, other)
        return self * other

    def __truediv__(self, other: Union[int, float, "Balance"]):
        if isinstance(other, Balance):
            _check_currencies(self, other)
            return Balance.from_rao(int(self.rao / other.rao)).set_unit(self.netuid)
        else:
            try:
                # Attempt to cast to int from rao
                return Balance.from_rao(int(self.rao / other)).set_unit(self.netuid)
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __rtruediv__(self, other: Union[int, float, "Balance"]):
        if isinstance(other, Balance):
            _check_currencies(self, other)
            return Balance.from_rao(int(other.rao / self.rao)).set_unit(self.netuid)
        else:
            try:
                # Attempt to cast to int from rao
                return Balance.from_rao(int(other / self.rao)).set_unit(self.netuid)
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __floordiv__(self, other: Union[int, float, "Balance"]):
        if isinstance(other, Balance):
            _check_currencies(self, other)
            return Balance.from_rao(int(self.tao // other.tao)).set_unit(self.netuid)
        else:
            try:
                # Attempt to cast to int from rao
                return Balance.from_rao(int(self.rao // other)).set_unit(self.netuid)
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __rfloordiv__(self, other: Union[int, float, "Balance"]):
        if isinstance(other, Balance):
            _check_currencies(self, other)
            return Balance.from_rao(int(other.rao // self.rao)).set_unit(self.netuid)
        else:
            try:
                # Attempt to cast to int from rao
                return Balance.from_rao(int(other // self.rao)).set_unit(self.netuid)
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __nonzero__(self) -> bool:
        return bool(self.rao)

    def __neg__(self):
        return Balance.from_rao(-self.rao).set_unit(self.netuid)

    def __pos__(self):
        return Balance.from_rao(self.rao).set_unit(self.netuid)

    def __abs__(self):
        return Balance.from_rao(abs(self.rao)).set_unit(self.netuid)

    @staticmethod
    def from_float(amount: float, netuid: int = 0) -> "Balance":
        """
        Given tao, return :func:`Balance` object with rao(``int``) and tao(``float``), where rao = int(tao*pow(10,9))
        Args:
            amount (float): The amount in tao.
            netuid (int): The subnet uid for set currency unit. Defaults to `0`.

        Returns:
            A Balance object representing the given amount.
        """
        rao_ = int(amount * pow(10, 9))
        return Balance(rao_).set_unit(netuid)

    @staticmethod
    def from_tao(amount: float, netuid: int = 0) -> "Balance":
        """
        Given tao, return Balance object with rao(``int``) and tao(``float``), where rao = int(tao*pow(10,9))

        Args:
            amount (float): The amount in tao.
            netuid (int): The subnet uid for set currency unit. Defaults to `0`.

        Returns:
            A Balance object representing the given amount.
        """
        rao_ = int(amount * pow(10, 9))
        return Balance(rao_).set_unit(netuid)

    @staticmethod
    def from_rao(amount: int, netuid: int = 0) -> "Balance":
        """
        Given rao, return Balance object with rao(``int``) and tao(``float``), where rao = int(tao*pow(10,9))

        Args:
            amount (int): The amount in rao.
            netuid (int): The subnet uid for set currency unit. Defaults to `0`.

        Returns:
            A Balance object representing the given amount.
        """
        return Balance(amount).set_unit(netuid)

    @staticmethod
    def get_unit(netuid: int) -> str:
        base = len(units)
        if netuid < base:
            return units[netuid]
        else:
            result = ""
            while netuid > 0:
                result = units[netuid % base] + result
                netuid //= base
            return result

    def set_unit(self, netuid: int):
        self.netuid = netuid
        self.unit = Balance.get_unit(netuid)
        self.rao_unit = Balance.get_unit(netuid)
        return self


class FixedPoint(TypedDict):
    """
    Represents a fixed point ``U64F64`` number.
    Where ``bits`` is a U128 representation of the fixed point number.

    This matches the type of the Alpha shares.
    """

    bits: int


def fixed_to_float(
    fixed: Union[FixedPoint, ScaleType], frac_bits: int = 64, total_bits: int = 128
) -> float:
    # By default, this is a U64F64
    # which is 64 bits of integer and 64 bits of fractional

    data: int = fixed["bits"]

    # Logical and to get the fractional part; remaining is the integer part
    fractional_part = data & (2**frac_bits - 1)
    # Shift to get the integer part from the remaining bits
    integer_part = data >> (total_bits - frac_bits)

    frac_float = fractional_part / (2**frac_bits)

    return integer_part + frac_float


units = [
    # Greek Alphabet (0-24)
    "\u03c4",  # τ (tau, 0)
    "\u03b1",  # α (alpha, 1)
    "\u03b2",  # β (beta, 2)
    "\u03b3",  # γ (gamma, 3)
    "\u03b4",  # δ (delta, 4)
    "\u03b5",  # ε (epsilon, 5)
    "\u03b6",  # ζ (zeta, 6)
    "\u03b7",  # η (eta, 7)
    "\u03b8",  # θ (theta, 8)
    "\u03b9",  # ι (iota, 9)
    "\u03ba",  # κ (kappa, 10)
    "\u03bb",  # λ (lambda, 11)
    "\u03bc",  # μ (mu, 12)
    "\u03bd",  # ν (nu, 13)
    "\u03be",  # ξ (xi, 14)
    "\u03bf",  # ο (omicron, 15)
    "\u03c0",  # π (pi, 16)
    "\u03c1",  # ρ (rho, 17)
    "\u03c3",  # σ (sigma, 18)
    "t",  # t (tau, 19)
    "\u03c5",  # υ (upsilon, 20)
    "\u03c6",  # φ (phi, 21)
    "\u03c7",  # χ (chi, 22)
    "\u03c8",  # ψ (psi, 23)
    "\u03c9",  # ω (omega, 24)
    # Hebrew Alphabet (25-51)
    "\u05d0",  # א (aleph, 25)
    "\u05d1",  # ב (bet, 26)
    "\u05d2",  # ג (gimel, 27)
    "\u05d3",  # ד (dalet, 28)
    "\u05d4",  # ה (he, 29)
    "\u05d5",  # ו (vav, 30)
    "\u05d6",  # ז (zayin, 31)
    "\u05d7",  # ח (het, 32)
    "\u05d8",  # ט (tet, 33)
    "\u05d9",  # י (yod, 34)
    "\u05da",  # ך (final kaf, 35)
    "\u05db",  # כ (kaf, 36)
    "\u05dc",  # ל (lamed, 37)
    "\u05dd",  # ם (final mem, 38)
    "\u05de",  # מ (mem, 39)
    "\u05df",  # ן (final nun, 40)
    "\u05e0",  # נ (nun, 41)
    "\u05e1",  # ס (samekh, 42)
    "\u05e2",  # ע (ayin, 43)
    "\u05e3",  # ף (final pe, 44)
    "\u05e4",  # פ (pe, 45)
    "\u05e5",  # ץ (final tsadi, 46)
    "\u05e6",  # צ (tsadi, 47)
    "\u05e7",  # ק (qof, 48)
    "\u05e8",  # ר (resh, 49)
    "\u05e9",  # ש (shin, 50)
    "\u05ea",  # ת (tav, 51)
    # Arabic Alphabet (52-81)
    "\u0627",  # ا (alif, 52)
    "\u0628",  # ب (ba, 53)
    "\u062a",  # ت (ta, 54)
    "\u062b",  # ث (tha, 55)
    "\u062c",  # ج (jeem, 56)
    "\u062d",  # ح (ha, 57)
    "\u062e",  # خ (kha, 58)
    "\u062f",  # د (dal, 59)
    "\u0630",  # ذ (dhal, 60)
    "\u0631",  # ر (ra, 61)
    "\u0632",  # ز (zay, 62)
    "\u0633",  # س (seen, 63)
    "\u0634",  # ش (sheen, 64)
    "\u0635",  # ص (sad, 65)
    "\u0636",  # ض (dad, 66)
    "\u0637",  # ط (ta, 67)
    "\u0638",  # ظ (dha, 68)
    "\u0639",  # ع (ain, 69)
    "\u063a",  # غ (ghain, 70)
    "\u0641",  # ف (fa, 71)
    "\u0642",  # ق (qaf, 72)
    "\u0643",  # ك (kaf, 73)
    "\u0644",  # ل (lam, 74)
    "\u0645",  # م (meem, 75)
    "\u0646",  # ن (noon, 76)
    "\u0647",  # ه (ha, 77)
    "\u0648",  # و (waw, 78)
    "\u064a",  # ي (ya, 79)
    "\u0649",  # ى (alef maksura, 80)
    "\u064a",  # ي (ya, 81)
    # Runic Alphabet (82-90)
    "\u16a0",  # ᚠ (fehu, 82)
    "\u16a2",  # ᚢ (uruz, 83)
    "\u16a6",  # ᚦ (thurisaz, 84)
    "\u16a8",  # ᚨ (ansuz, 85)
    "\u16b1",  # ᚱ (raidho, 86)
    "\u16b3",  # ᚲ (kaunan, 87)
    "\u16c7",  # ᛇ (eihwaz, 88)
    "\u16c9",  # ᛉ (algiz, 89)
    "\u16d2",  # ᛒ (berkanan, 90)
    # Ogham Alphabet (91-97)
    "\u1680",  #   (Space, 91)
    "\u1681",  # ᚁ (Beith, 92)
    "\u1682",  # ᚂ (Luis, 93)
    "\u1683",  # ᚃ (Fearn, 94)
    "\u1684",  # ᚄ (Sail, 95)
    "\u1685",  # ᚅ (Nion, 96)
    "\u169b",  # ᚛ (Forfeda, 97)
    # Georgian Alphabet (98-103)
    "\u10d0",  # ა (ani, 98)
    "\u10d1",  # ბ (bani, 99)
    "\u10d2",  # გ (gani, 100)
    "\u10d3",  # დ (doni, 101)
    "\u10d4",  # ე (eni, 102)
    "\u10d5",  # ვ (vini, 103)
    # Armenian Alphabet (104-110)
    "\u0531",  # Ա (Ayp, 104)
    "\u0532",  # Բ (Ben, 105)
    "\u0533",  # Գ (Gim, 106)
    "\u0534",  # Դ (Da, 107)
    "\u0535",  # Ե (Ech, 108)
    "\u0536",  # Զ (Za, 109)
    "\u055e",  # ՞ (Question mark, 110)
    # Cyrillic Alphabet (111-116)
    "\u0400",  # Ѐ (Ie with grave, 111)
    "\u0401",  # Ё (Io, 112)
    "\u0402",  # Ђ (Dje, 113)
    "\u0403",  # Ѓ (Gje, 114)
    "\u0404",  # Є (Ukrainian Ie, 115)
    "\u0405",  # Ѕ (Dze, 116)
    # Coptic Alphabet (117-122)
    "\u2c80",  # Ⲁ (Alfa, 117)
    "\u2c81",  # ⲁ (Small Alfa, 118)
    "\u2c82",  # Ⲃ (Vida, 119)
    "\u2c83",  # ⲃ (Small Vida, 120)
    "\u2c84",  # Ⲅ (Gamma, 121)
    "\u2c85",  # ⲅ (Small Gamma, 122)
    # Brahmi Script (123-127)
    "\U00011000",  # 𑀀 (A, 123)
    "\U00011001",  # 𑀁 (Aa, 124)
    "\U00011002",  # 𑀂 (I, 125)
    "\U00011003",  # 𑀃 (Ii, 126)
    "\U00011005",  # 𑀅 (U, 127)
    # Tifinagh Alphabet (128-133)
    "\u2d30",  # ⴰ (Ya, 128)
    "\u2d31",  # ⴱ (Yab, 129)
    "\u2d32",  # ⴲ (Yabh, 130)
    "\u2d33",  # ⴳ (Yag, 131)
    "\u2d34",  # ⴴ (Yagh, 132)
    "\u2d35",  # ⴵ (Yaj, 133)
    # Glagolitic Alphabet (134-166)
    "\u2c00",  # Ⰰ (Az, 134)
    "\u2c01",  # Ⰱ (Buky, 135)
    "\u2c02",  # Ⰲ (Vede, 136)
    "\u2c03",  # Ⰳ (Glagoli, 137)
    "\u2c04",  # Ⰴ (Dobro, 138)
    "\u2c05",  # Ⰵ (Yest, 139)
    "\u2c06",  # Ⰶ (Zhivete, 140)
    "\u2c07",  # Ⰷ (Zemlja, 141)
    "\u2c08",  # Ⰸ (Izhe, 142)
    "\u2c09",  # Ⰹ (Initial Izhe, 143)
    "\u2c0a",  # Ⰺ (I, 144)
    "\u2c0b",  # Ⰻ (Djerv, 145)
    "\u2c0c",  # Ⰼ (Kako, 146)
    "\u2c0d",  # Ⰽ (Ljudije, 147)
    "\u2c0e",  # Ⰾ (Myse, 148)
    "\u2c0f",  # Ⰿ (Nash, 149)
    "\u2c10",  # Ⱀ (On, 150)
    "\u2c11",  # Ⱁ (Pokoj, 151)
    "\u2c12",  # Ⱂ (Rtsy, 152)
    "\u2c13",  # Ⱃ (Slovo, 153)
    "\u2c14",  # Ⱄ (Tvrido, 154)
    "\u2c15",  # Ⱅ (Uku, 155)
    "\u2c16",  # Ⱆ (Fert, 156)
    "\u2c17",  # Ⱇ (Xrivi, 157)
    "\u2c18",  # Ⱈ (Ot, 158)
    "\u2c19",  # Ⱉ (Cy, 159)
    "\u2c1a",  # Ⱊ (Shcha, 160)
    "\u2c1b",  # Ⱋ (Er, 161)
    "\u2c1c",  # Ⱌ (Yeru, 162)
    "\u2c1d",  # Ⱍ (Small Yer, 163)
    "\u2c1e",  # Ⱎ (Yo, 164)
    "\u2c1f",  # Ⱏ (Yu, 165)
    "\u2c20",  # Ⱐ (Ja, 166)
    # Thai Alphabet (167-210)
    "\u0e01",  # ก (Ko Kai, 167)
    "\u0e02",  # ข (Kho Khai, 168)
    "\u0e03",  # ฃ (Kho Khuat, 169)
    "\u0e04",  # ค (Kho Khon, 170)
    "\u0e05",  # ฅ (Kho Rakhang, 171)
    "\u0e06",  # ฆ (Kho Khwai, 172)
    "\u0e07",  # ง (Ngo Ngu, 173)
    "\u0e08",  # จ (Cho Chan, 174)
    "\u0e09",  # ฉ (Cho Ching, 175)
    "\u0e0a",  # ช (Cho Chang, 176)
    "\u0e0b",  # ซ (So So, 177)
    "\u0e0c",  # ฌ (Cho Choe, 178)
    "\u0e0d",  # ญ (Yo Ying, 179)
    "\u0e0e",  # ฎ (Do Chada, 180)
    "\u0e0f",  # ฏ (To Patak, 181)
    "\u0e10",  # ฐ (Tho Than, 182)
    "\u0e11",  # ฑ (Tho Nangmontho, 183)
    "\u0e12",  # ฒ (Tho Phuthao, 184)
    "\u0e13",  # ณ (No Nen, 185)
    "\u0e14",  # ด (Do Dek, 186)
    "\u0e15",  # ต (To Tao, 187)
    "\u0e16",  # ถ (Tho Thung, 188)
    "\u0e17",  # ท (Tho Thahan, 189)
    "\u0e18",  # ธ (Tho Thong, 190)
    "\u0e19",  # น (No Nu, 191)
    "\u0e1a",  # บ (Bo Baimai, 192)
    "\u0e1b",  # ป (Po Pla, 193)
    "\u0e1c",  # ผ (Pho Phung, 194)
    "\u0e1d",  # ฝ (Fo Fa, 195)
    "\u0e1e",  # พ (Pho Phan, 196)
    "\u0e1f",  # ฟ (Fo Fan, 197)
    "\u0e20",  # ภ (Pho Samphao, 198)
    "\u0e21",  # ม (Mo Ma, 199)
    "\u0e22",  # ย (Yo Yak, 200)
    "\u0e23",  # ร (Ro Rua, 201)
    "\u0e25",  # ล (Lo Ling, 202)
    "\u0e27",  # ว (Wo Waen, 203)
    "\u0e28",  # ศ (So Sala, 204)
    "\u0e29",  # ษ (So Rusi, 205)
    "\u0e2a",  # ส (So Sua, 206)
    "\u0e2b",  # ห (Ho Hip, 207)
    "\u0e2c",  # ฬ (Lo Chula, 208)
    "\u0e2d",  # อ (O Ang, 209)
    "\u0e2e",  # ฮ (Ho Nokhuk, 210)
    # Hangul Consonants (211-224)
    "\u1100",  # ㄱ (Giyeok, 211)
    "\u1101",  # ㄴ (Nieun, 212)
    "\u1102",  # ㄷ (Digeut, 213)
    "\u1103",  # ㄹ (Rieul, 214)
    "\u1104",  # ㅁ (Mieum, 215)
    "\u1105",  # ㅂ (Bieup, 216)
    "\u1106",  # ㅅ (Siot, 217)
    "\u1107",  # ㅇ (Ieung, 218)
    "\u1108",  # ㅈ (Jieut, 219)
    "\u1109",  # ㅊ (Chieut, 220)
    "\u110a",  # ㅋ (Kieuk, 221)
    "\u110b",  # ㅌ (Tieut, 222)
    "\u110c",  # ㅍ (Pieup, 223)
    "\u110d",  # ㅎ (Hieut, 224)
    # Hangul Vowels (225-245)
    "\u1161",  # ㅏ (A, 225)
    "\u1162",  # ㅐ (Ae, 226)
    "\u1163",  # ㅑ (Ya, 227)
    "\u1164",  # ㅒ (Yae, 228)
    "\u1165",  # ㅓ (Eo, 229)
    "\u1166",  # ㅔ (E, 230)
    "\u1167",  # ㅕ (Yeo, 231)
    "\u1168",  # ㅖ (Ye, 232)
    "\u1169",  # ㅗ (O, 233)
    "\u116a",  # ㅘ (Wa, 234)
    "\u116b",  # ㅙ (Wae, 235)
    "\u116c",  # ㅚ (Oe, 236)
    "\u116d",  # ㅛ (Yo, 237)
    "\u116e",  # ㅜ (U, 238)
    "\u116f",  # ㅝ (Weo, 239)
    "\u1170",  # ㅞ (We, 240)
    "\u1171",  # ㅟ (Wi, 241)
    "\u1172",  # ㅠ (Yu, 242)
    "\u1173",  # ㅡ (Eu, 243)
    "\u1174",  # ㅢ (Ui, 244)
    "\u1175",  # ㅣ (I, 245)
    # Ethiopic Alphabet (246-274)
    "\u12a0",  # አ (Glottal A, 246)
    "\u12a1",  # ኡ (Glottal U, 247)
    "\u12a2",  # ኢ (Glottal I, 248)
    "\u12a3",  # ኣ (Glottal Aa, 249)
    "\u12a4",  # ኤ (Glottal E, 250)
    "\u12a5",  # እ (Glottal Ie, 251)
    "\u12a6",  # ኦ (Glottal O, 252)
    "\u12a7",  # ኧ (Glottal Wa, 253)
    "\u12c8",  # ወ (Wa, 254)
    "\u12c9",  # ዉ (Wu, 255)
    "\u12ca",  # ዊ (Wi, 256)
    "\u12cb",  # ዋ (Waa, 257)
    "\u12cc",  # ዌ (We, 258)
    "\u12cd",  # ው (Wye, 259)
    "\u12ce",  # ዎ (Wo, 260)
    "\u12b0",  # ኰ (Ko, 261)
    "\u12b1",  # ኱ (Ku, 262)
    "\u12b2",  # ኲ (Ki, 263)
    "\u12b3",  # ኳ (Kua, 264)
    "\u12b4",  # ኴ (Ke, 265)
    "\u12b5",  # ኵ (Kwe, 266)
    "\u12b6",  # ኶ (Ko, 267)
    "\u12a0",  # ጐ (Go, 268)
    "\u12a1",  # ጑ (Gu, 269)
    "\u12a2",  # ጒ (Gi, 270)
    "\u12a3",  # መ (Gua, 271)
    "\u12a4",  # ጔ (Ge, 272)
    "\u12a5",  # ጕ (Gwe, 273)
    "\u12a6",  # ጖ (Go, 274)
    # Devanagari Alphabet (275-318)
    "\u0905",  # अ (A, 275)
    "\u0906",  # आ (Aa, 276)
    "\u0907",  # इ (I, 277)
    "\u0908",  # ई (Ii, 278)
    "\u0909",  # उ (U, 279)
    "\u090a",  # ऊ (Uu, 280)
    "\u090b",  # ऋ (R, 281)
    "\u090f",  # ए (E, 282)
    "\u0910",  # ऐ (Ai, 283)
    "\u0913",  # ओ (O, 284)
    "\u0914",  # औ (Au, 285)
    "\u0915",  # क (Ka, 286)
    "\u0916",  # ख (Kha, 287)
    "\u0917",  # ग (Ga, 288)
    "\u0918",  # घ (Gha, 289)
    "\u0919",  # ङ (Nga, 290)
    "\u091a",  # च (Cha, 291)
    "\u091b",  # छ (Chha, 292)
    "\u091c",  # ज (Ja, 293)
    "\u091d",  # झ (Jha, 294)
    "\u091e",  # ञ (Nya, 295)
    "\u091f",  # ट (Ta, 296)
    "\u0920",  # ठ (Tha, 297)
    "\u0921",  # ड (Da, 298)
    "\u0922",  # ढ (Dha, 299)
    "\u0923",  # ण (Na, 300)
    "\u0924",  # त (Ta, 301)
    "\u0925",  # थ (Tha, 302)
    "\u0926",  # द (Da, 303)
    "\u0927",  # ध (Dha, 304)
    "\u0928",  # न (Na, 305)
    "\u092a",  # प (Pa, 306)
    "\u092b",  # फ (Pha, 307)
    "\u092c",  # ब (Ba, 308)
    "\u092d",  # भ (Bha, 309)
    "\u092e",  # म (Ma, 310)
    "\u092f",  # य (Ya, 311)
    "\u0930",  # र (Ra, 312)
    "\u0932",  # ल (La, 313)
    "\u0935",  # व (Va, 314)
    "\u0936",  # श (Sha, 315)
    "\u0937",  # ष (Ssa, 316)
    "\u0938",  # स (Sa, 317)
    "\u0939",  # ह (Ha, 318)
    # Katakana Alphabet (319-364)
    "\u30a2",  # ア (A, 319)
    "\u30a4",  # イ (I, 320)
    "\u30a6",  # ウ (U, 321)
    "\u30a8",  # エ (E, 322)
    "\u30aa",  # オ (O, 323)
    "\u30ab",  # カ (Ka, 324)
    "\u30ad",  # キ (Ki, 325)
    "\u30af",  # ク (Ku, 326)
    "\u30b1",  # ケ (Ke, 327)
    "\u30b3",  # コ (Ko, 328)
    "\u30b5",  # サ (Sa, 329)
    "\u30b7",  # シ (Shi, 330)
    "\u30b9",  # ス (Su, 331)
    "\u30bb",  # セ (Se, 332)
    "\u30bd",  # ソ (So, 333)
    "\u30bf",  # タ (Ta, 334)
    "\u30c1",  # チ (Chi, 335)
    "\u30c4",  # ツ (Tsu, 336)
    "\u30c6",  # テ (Te, 337)
    "\u30c8",  # ト (To, 338)
    "\u30ca",  # ナ (Na, 339)
    "\u30cb",  # ニ (Ni, 340)
    "\u30cc",  # ヌ (Nu, 341)
    "\u30cd",  # ネ (Ne, 342)
    "\u30ce",  # ノ (No, 343)
    "\u30cf",  # ハ (Ha, 344)
    "\u30d2",  # ヒ (Hi, 345)
    "\u30d5",  # フ (Fu, 346)
    "\u30d8",  # ヘ (He, 347)
    "\u30db",  # ホ (Ho, 348)
    "\u30de",  # マ (Ma, 349)
    "\u30df",  # ミ (Mi, 350)
    "\u30e0",  # ム (Mu, 351)
    "\u30e1",  # メ (Me, 352)
    "\u30e2",  # モ (Mo, 353)
    "\u30e4",  # ヤ (Ya, 354)
    "\u30e6",  # ユ (Yu, 355)
    "\u30e8",  # ヨ (Yo, 356)
    "\u30e9",  # ラ (Ra, 357)
    "\u30ea",  # リ (Ri, 358)
    "\u30eb",  # ル (Ru, 359)
    "\u30ec",  # レ (Re, 360)
    "\u30ed",  # ロ (Ro, 361)
    "\u30ef",  # ワ (Wa, 362)
    "\u30f2",  # ヲ (Wo, 363)
    "\u30f3",  # ン (N, 364)
    # Tifinagh Alphabet (365-400)
    "\u2d30",  # ⴰ (Ya, 365)
    "\u2d31",  # ⴱ (Yab, 366)
    "\u2d32",  # ⴲ (Yabh, 367)
    "\u2d33",  # ⴳ (Yag, 368)
    "\u2d34",  # ⴴ (Yagh, 369)
    "\u2d35",  # ⴵ (Yaj, 370)
    "\u2d36",  # ⴶ (Yach, 371)
    "\u2d37",  # ⴷ (Yad, 372)
    "\u2d38",  # ⴸ (Yadh, 373)
    "\u2d39",  # ⴹ (Yadh, emphatic, 374)
    "\u2d3a",  # ⴺ (Yaz, 375)
    "\u2d3b",  # ⴻ (Yazh, 376)
    "\u2d3c",  # ⴼ (Yaf, 377)
    "\u2d3d",  # ⴽ (Yak, 378)
    "\u2d3e",  # ⴾ (Yak, variant, 379)
    "\u2d3f",  # ⴿ (Yaq, 380)
    "\u2d40",  # ⵀ (Yah, 381)
    "\u2d41",  # ⵁ (Yahh, 382)
    "\u2d42",  # ⵂ (Yahl, 383)
    "\u2d43",  # ⵃ (Yahm, 384)
    "\u2d44",  # ⵄ (Yayn, 385)
    "\u2d45",  # ⵅ (Yakh, 386)
    "\u2d46",  # ⵆ (Yakl, 387)
    "\u2d47",  # ⵇ (Yahq, 388)
    "\u2d48",  # ⵈ (Yash, 389)
    "\u2d49",  # ⵉ (Yi, 390)
    "\u2d4a",  # ⵊ (Yij, 391)
    "\u2d4b",  # ⵋ (Yizh, 392)
    "\u2d4c",  # ⵌ (Yink, 393)
    "\u2d4d",  # ⵍ (Yal, 394)
    "\u2d4e",  # ⵎ (Yam, 395)
    "\u2d4f",  # ⵏ (Yan, 396)
    "\u2d50",  # ⵐ (Yang, 397)
    "\u2d51",  # ⵑ (Yany, 398)
    "\u2d52",  # ⵒ (Yap, 399)
    "\u2d53",  # ⵓ (Yu, 400)
    # Sinhala Alphabet (401-444)
    "\u0d85",  # අ (A, 401)
    "\u0d86",  # ආ (Aa, 402)
    "\u0d87",  # ඉ (I, 403)
    "\u0d88",  # ඊ (Ii, 404)
    "\u0d89",  # උ (U, 405)
    "\u0d8a",  # ඌ (Uu, 406)
    "\u0d8b",  # ඍ (R, 407)
    "\u0d8c",  # ඎ (Rr, 408)
    "\u0d8f",  # ඏ (L, 409)
    "\u0d90",  # ඐ (Ll, 410)
    "\u0d91",  # එ (E, 411)
    "\u0d92",  # ඒ (Ee, 412)
    "\u0d93",  # ඓ (Ai, 413)
    "\u0d94",  # ඔ (O, 414)
    "\u0d95",  # ඕ (Oo, 415)
    "\u0d96",  # ඖ (Au, 416)
    "\u0d9a",  # ක (Ka, 417)
    "\u0d9b",  # ඛ (Kha, 418)
    "\u0d9c",  # ග (Ga, 419)
    "\u0d9d",  # ඝ (Gha, 420)
    "\u0d9e",  # ඞ (Nga, 421)
    "\u0d9f",  # ච (Cha, 422)
    "\u0da0",  # ඡ (Chha, 423)
    "\u0da1",  # ජ (Ja, 424)
    "\u0da2",  # ඣ (Jha, 425)
    "\u0da3",  # ඤ (Nya, 426)
    "\u0da4",  # ට (Ta, 427)
    "\u0da5",  # ඥ (Tha, 428)
    "\u0da6",  # ඦ (Da, 429)
    "\u0da7",  # ට (Dha, 430)
    "\u0da8",  # ඨ (Na, 431)
    "\u0daa",  # ඪ (Pa, 432)
    "\u0dab",  # ණ (Pha, 433)
    "\u0dac",  # ඬ (Ba, 434)
    "\u0dad",  # ත (Bha, 435)
    "\u0dae",  # ථ (Ma, 436)
    "\u0daf",  # ද (Ya, 437)
    "\u0db0",  # ධ (Ra, 438)
    "\u0db1",  # ඲ (La, 439)
    "\u0db2",  # ඳ (Va, 440)
    "\u0db3",  # ප (Sha, 441)
    "\u0db4",  # ඵ (Ssa, 442)
    "\u0db5",  # බ (Sa, 443)
    "\u0db6",  # භ (Ha, 444)
]


def tao(amount: float, netuid: int = 0) -> Balance:
    """
    Helper function to create a Balance object from a float (Tao)
    """
    return Balance.from_tao(amount).set_unit(netuid)


def rao(amount: int, netuid: int = 0) -> Balance:
    """
    Helper function to create a Balance object from an int (Rao)
    """
    return Balance.from_rao(amount).set_unit(netuid)


def check_and_convert_to_balance(
    amount: Union[float, int, Optional[Balance]],
) -> Balance:
    """
    Helper function to check and convert the amount type to a Balance object.
    This is used to support backwards compatibility while also providing a deprecation notice.
    """
    if isinstance(amount, (float, int)):
        warnings.simplefilter("default", DeprecationWarning)
        warnings.warn(
            "Detected a non-balance amount. Converting to Balance from Tao for backwards compatibility."
            "Please update your code to use tao(amount) or Balance.from_tao(amount) for the main release 9.0.0.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        amount = tao(amount)
    return amount
