# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from bittensor.core.synapse import Synapse


class SubstrateRequestException(Exception):
    pass


class StorageFunctionNotFound(ValueError):
    pass


class BlockNotFound(Exception):
    pass


class ExtrinsicNotFound(Exception):
    pass


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
