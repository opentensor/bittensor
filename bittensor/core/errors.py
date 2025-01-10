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


class ChainError(SubstrateRequestException):
    """Base error for any chain related errors."""


class ChainConnectionError(ChainError):
    """Error for any chain connection related errors."""


class ChainTransactionError(ChainError):
    """Error for any chain transaction related errors."""


class ChainQueryError(ChainError):
    """Error for any chain query related errors."""


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
