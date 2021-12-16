""" Represent bittensor balance of the wallet with rao and tao
"""
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
    """ Represent bittensor balance of the wallet with rao and tao
    """
    unit = "\u03C4"
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
        return "\u03C4{}\u002C{}".format( str(float(self.tao)).split('.')[0], str(float(self.tao)).split('.')[1])

    def __rich__(self):
        return "[green]\u03C4[/green][green]{}[/green][green].[/green][dim green]{}[/dim green]".format( str(float(self.tao)).split('.')[0], str(float(self.tao)).split('.')[1])

    def __str_rao__(self):
        return "\u03C1{}".format( int(self.rao) )

    def __rich_rao__(self):
        return "[green]\u03C1{}[/green]".format( int(self.rao) )

    def __repr__(self):
        return self.__str__()

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

    def __add__(self, other):
        return Balance(self.rao + other.rao)

    def __sub__(self, other):
        return Balance(self.rao - other.rao)

    def __neg__(self):
        return Balance(-self.rao)

    def __pos__(self):
        return Balance(self.rao)

    def __abs__(self):
        return Balance(abs(self.rao))

    @staticmethod
    def from_float(amount : float):
        """ Given tao (float), return Balance object with rao(int) and tao(float), where rao = int(tao*pow(10,9))
        """
        rao = int(amount * pow(10, 9))
        return Balance(rao)

    @staticmethod
    def from_tao(amount : float):
        """ Given tao (float), return Balance object with rao(int) and tao(float), where rao = int(tao*pow(10,9))
        """
        rao = int(amount * pow(10, 9))
        return Balance(rao)

    @staticmethod
    def from_rao(amount: int):
        """ Given rao (int), return Balance object with rao(int) and tao(float), where rao = int(tao*pow(10,9))
        """
        return Balance(amount)
