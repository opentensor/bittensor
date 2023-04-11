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

import time
import bittensor
from dataclasses import dataclass

@dataclass
class BittensorCall(object):
    """ CallState object.
        CallState is a dataclass that holds the state of a call to a receptor.
    """
    # The hotkey of the receptor.
    hotkey: str = ''
    # The version of the caller
    version: int = 0
    # The timeout for the call.
    timeout: float = 0.0
    # The start time of the call.
    start_time: float = 0.0
    # The end time of the call.
    end_time: float = 0.0
    # The request code, filled while preprocessing the request.
    request_code: bittensor.proto.ReturnCode = bittensor.proto.ReturnCode.Success
    # The request message, filled while preprocessing the request.
    request_message: str = 'Success'
    # The response code, filled after the call is made.
    response_code: bittensor.proto.ReturnCode = bittensor.proto.ReturnCode.Success
    # The response message, filled after the call is made.
    response_message: str = 'Success'
    # The request proto, filled while preprocessing the request.
    request_proto: object = None
    # The response proto, filled after the call is made.
    response_proto: object = None
    
    def __init__(
            self, 
            timeout: float = bittensor.__blocktime__
        ):
        self.timeout = timeout
        self.start_time = time.time()

    def get_inputs_shape(self):
        raise NotImplementedError('process_forward_response_proto not implemented for this call type.')
    
    def get_outputs_shape(self):
        raise NotImplementedError('process_forward_response_proto not implemented for this call type.')
