class EmptyTensorException (Exception):
    """ Raised when tensor included in the response is unexpectedly empty """
    pass

class ResponseShapeException (Exception):
    """ Raised when tensor included in the response has an incorrect shape """
    pass