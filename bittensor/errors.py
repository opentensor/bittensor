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
    r"""Base error for any chain related errors."""

    pass


class ChainConnectionError(ChainError):
    r"""Error for any chain connection related errors."""

    pass


class ChainTransactionError(ChainError):
    r"""Error for any chain transaction related errors."""

    pass


class ChainQueryError(ChainError):
    r"""Error for any chain query related errors."""

    pass


class StakeError(ChainTransactionError):
    r"""Error raised when a stake transaction fails."""

    pass


class UnstakeError(ChainTransactionError):
    r"""Error raised when an unstake transaction fails."""

    pass


class IdentityError(ChainTransactionError):
    r"""Error raised when an identity transaction fails."""

    pass


class NominationError(ChainTransactionError):
    r"""Error raised when a nomination transaction fails."""

    pass


class TakeError(ChainTransactionError):
    r"""Error raised when a increase / decrease take transaction fails."""

    pass


class TransferError(ChainTransactionError):
    r"""Error raised when a transfer transaction fails."""

    pass


class RegistrationError(ChainTransactionError):
    r"""Error raised when a neuron registration transaction fails."""

    pass


class NotRegisteredError(ChainTransactionError):
    r"""Error raised when a neuron is not registered, and the transaction requires it to be."""

    pass


class NotDelegateError(StakeError):
    r"""Error raised when a hotkey you are trying to stake to is not a delegate."""

    pass


class KeyFileError(Exception):
    """Error thrown when the keyfile is corrupt, non-writable, non-readable or the password used to decrypt is invalid."""

    pass


class MetadataError(ChainTransactionError):
    r"""Error raised when metadata commitment transaction fails."""

    pass


class InvalidRequestNameError(Exception):
    r"""This exception is raised when the request name is invalid. Ususally indicates a broken URL."""

    pass


class UnknownSynapseError(Exception):
    r"""This exception is raised when the request name is not found in the Axon's forward_fns dictionary."""

    pass


class SynapseParsingError(Exception):
    r"""This exception is raised when the request headers are unable to be parsed into the synapse type."""

    pass


class NotVerifiedException(Exception):
    r"""This exception is raised when the request is not verified."""

    pass


class BlacklistedException(Exception):
    r"""This exception is raised when the request is blacklisted."""

    pass


class PriorityException(Exception):
    r"""This exception is raised when the request priority is not met."""

    pass


class PostProcessException(Exception):
    r"""This exception is raised when the response headers cannot be updated."""

    pass


class RunException(Exception):
    r"""This exception is raised when the requested function cannot be executed. Indicates a server error."""

    pass


class InternalServerError(Exception):
    r"""This exception is raised when the requested function fails on the server. Indicates a server error."""

    pass


class SynapseDendriteNoneException(Exception):
    def __init__(self, message="Synapse Dendrite is None"):
        self.message = message
        super().__init__(self.message)
