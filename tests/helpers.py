from typing import Union
from bittensor import Balance

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
