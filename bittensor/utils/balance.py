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

class Balance:
    unit = "Tao"
    rao : int
    tao: float

    # Balance should always be an int and represent the satoshi of our token:rao
    def __init__(self, balance):
        self.rao = balance
        self.tao = self.rao / pow(10, 9)

    def __int__(self):
        return self.rao

    def __float__(self):
        return self.tao

    def __str__(self):
        return "{unit:s} {balance:.9f}".format(unit=self.unit, balance=self.tao)

    def __eq__(self, other):
        return self.rao == other.rao

    def __ne__(self, other):
        return self.rao != other.rao

    def __gt__(self, other):
        return self.rao > other.rao

    def __lt__(self, other):
        return self.rao < other.rao

    def __le__(self, other):
        return self.rao <= other.rao

    def __ge__(self, other):
        return self.rao >= other.rao

    @staticmethod
    def from_float(amount : float):
        rao = int(amount * pow(10, 9))
        return Balance(rao)