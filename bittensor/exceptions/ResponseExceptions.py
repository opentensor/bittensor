class EmptyTensorException (Exception):
    """ Raised when tensor included in the response is unexpectedly empty """
    pass

class ResponseShapeException (Exception):
    """ Raised when a response message has an improper shape """
    pass

class RequestShapeException (Exception):
    """ Raised when a request message has an improper shape """
    pass

