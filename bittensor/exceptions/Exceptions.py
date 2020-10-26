class EmptyTensorException(Exception):
    """ Raised when tensor included in the response is unexpectedly empty """
    pass


class ResponseShapeException(Exception):
    """ Raised when a response message has an improper shape """
    pass


class RequestShapeException(Exception):
    """ Raised when a request message has an improper shape """
    pass


class SerializationException(Exception):
    """ Raised when message serialization fails """
    pass


class NonExistentSynapseException(Exception):
    """ Raised when the called synapse is not in the local synapse set """
    pass

class RemoteIPException(Exception):
    """ Raised when a failure occurs trying to set a remote IP """
    pass
