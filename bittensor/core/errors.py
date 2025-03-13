from typing import Optional, TYPE_CHECKING

from async_substrate_interface.errors import (
    SubstrateRequestException,
    StorageFunctionNotFound,
    BlockNotFound,
    ExtrinsicNotFound,
)

if TYPE_CHECKING:
    from bittensor.core.synapse import Synapse

# redundant aliases
SubstrateRequestException = SubstrateRequestException
StorageFunctionNotFound = StorageFunctionNotFound
BlockNotFound = BlockNotFound
ExtrinsicNotFound = ExtrinsicNotFound


class _ChainErrorMeta(type):
    _exceptions: dict[str, Exception] = {}

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)

        mcs._exceptions.setdefault(cls.__name__, cls)

        return cls

    @classmethod
    def get_exception_class(mcs, exception_name):
        return mcs._exceptions[exception_name]


class MaxSuccessException(Exception):
    """Raised when the POW Solver has reached the max number of successful solutions."""


class MaxAttemptsException(Exception):
    """Raised when the POW Solver has reached the max number of attempts."""


class ChainError(SubstrateRequestException, metaclass=_ChainErrorMeta):
    """Base error for any chain related errors."""

    @classmethod
    def from_error(cls, error):
        try:
            error_cls = _ChainErrorMeta.get_exception_class(
                error["name"],
            )
        except KeyError:
            return cls(error)
        else:
            return error_cls(" ".join(error["docs"]))


class ChainConnectionError(ChainError):
    """Error for any chain connection related errors."""


class ChainTransactionError(ChainError):
    """Error for any chain transaction related errors."""


class ChainQueryError(ChainError):
    """Error for any chain query related errors."""


class DelegateTakeTooHigh(ChainTransactionError):
    """
    Delegate take is too high.
    """


class DelegateTakeTooLow(ChainTransactionError):
    """
    Delegate take is too low.
    """


class DelegateTxRateLimitExceeded(ChainTransactionError):
    """
    A transactor exceeded the rate limit for delegate transaction.
    """


class HotKeyAccountNotExists(ChainTransactionError):
    """
    The hotkey does not exist.
    """


class NonAssociatedColdKey(ChainTransactionError):
    """
    Request to stake, unstake or subscribe is made by a coldkey that is not associated with the hotkey account.
    """


class StakeError(ChainTransactionError):
    """Error raised when a stake transaction fails."""


class UnstakeError(ChainTransactionError):
    """Error raised when an unstake transaction fails."""


class IdentityError(ChainTransactionError):
    """Error raised when an identity transaction fails."""


class NominationError(ChainTransactionError):
    """Error raised when a nomination transaction fails."""


class TakeError(ChainTransactionError):
    """Error raised when an increase / decrease take transaction fails."""


class TransferError(ChainTransactionError):
    """Error raised when a transfer transaction fails."""


class RegistrationError(ChainTransactionError):
    """Error raised when a neuron registration transaction fails."""


class NotRegisteredError(ChainTransactionError):
    """Error raised when a neuron is not registered, and the transaction requires it to be."""


class NotDelegateError(StakeError):
    """Error raised when a hotkey you are trying to stake to is not a delegate."""


class MetadataError(ChainTransactionError):
    """Error raised when metadata commitment transaction fails."""


class InvalidRequestNameError(Exception):
    """This exception is raised when the request name is invalid. Usually indicates a broken URL."""


class SynapseException(Exception):
    def __init__(
        self, message="Synapse Exception", synapse: Optional["Synapse"] = None
    ):
        self.message = message
        self.synapse = synapse
        super().__init__(self.message)


class UnknownSynapseError(SynapseException):
    """This exception is raised when the request name is not found in the Axon's forward_fns dictionary."""


class SynapseParsingError(Exception):
    """This exception is raised when the request headers are unable to be parsed into the synapse type."""


class NotVerifiedException(SynapseException):
    """This exception is raised when the request is not verified."""


class BlacklistedException(SynapseException):
    """This exception is raised when the request is blacklisted."""


class PriorityException(SynapseException):
    """This exception is raised when the request priority is not met."""


class PostProcessException(SynapseException):
    """This exception is raised when the response headers cannot be updated."""


class RunException(SynapseException):
    """This exception is raised when the requested function cannot be executed. Indicates a server error."""


class InternalServerError(SynapseException):
    """This exception is raised when the requested function fails on the server. Indicates a server error."""


class SynapseDendriteNoneException(SynapseException):
    def __init__(
        self,
        message="Synapse Dendrite is None",
        synapse: Optional["Synapse"] = None,
    ):
        self.message = message
        super().__init__(self.message, synapse)
