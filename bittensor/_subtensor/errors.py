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