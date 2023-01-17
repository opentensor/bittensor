# The MIT License (MIT)
# Copyright © 2023 Opentensor Foundation

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

class ChainError(BaseException):
    r""" Base error for any chain related errors.
    """
    pass


class ChainConnectionError(ChainError):
    r""" Error for any chain connection related errors.
    """
    pass


class ChainTransactionError(ChainError):
    r""" Error for any chain transaction related errors.
    """
    pass


class ChainQueryError(ChainError):
    r""" Error for any chain query related errors.
    """
    pass


class StakeError(ChainTransactionError):
    r""" Error raised when a stake transaction fails.
    """
    pass


class UnstakeError(ChainTransactionError):
    r""" Error raised when an unstake transaction fails.
    """
    pass


class TransferError(ChainTransactionError):
    r""" Error raised when a transfer transaction fails.
    """
    pass


class RegistrationError(ChainTransactionError):
    r""" Error raised when a neuron registration transaction fails.
    """
    pass


class NotRegisteredError(ChainTransactionError):
    r""" Error raised when a neuron is not registered, and the transaction requires it to be.
    """
    pass


class NotDelegateError(StakeError):
    r""" Error raised when a hotkey you are trying to stake to is not a delegate.
    """
    pass