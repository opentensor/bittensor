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

import bittensor 
def code_to_string( code: bittensor.proto.ReturnCode ) -> str:
    if code == 0: 
	    return 'Success'
    elif code == 1:
        return 'Timeout'
    elif code == 2:
        return 'Backoff'
    elif code == 3:
        return 'Unavailable'
    elif code == 4:
        return 'NotImplemented'
    elif code == 5:
        return 'EmptyRequest'
    elif code == 6:
        return 'EmptyResponse'
    elif code == 7:
        return 'InvalidResponse'
    elif code == 8:
        return 'InvalidRequest'
    elif code == 9:
        return 'RequestShapeException'
    elif code == 10:
        return 'ResponseShapeException'
    elif code == 11:
        return 'RequestSerializationException'
    elif code == 12:
        return 'ResponseSerializationException'
    elif code == 13:
        return 'RequestDeserializationException'
    elif code == 14:
        return 'ResponseDeserializationException'
    elif code == 15:
        return 'NotServingSynapse'
    elif code == 16:
        return 'NucleusTimeout'
    elif code == 17:
        return 'NucleusFull'
    elif code == 18:
        return 'RequestIncompatibleVersion'
    elif code == 19:
        return 'ResponseIncompatibleVersion'
    elif code == 20:
        return 'SenderUnknown'
    elif code == 21:
        return 'UnknownException'
    else:
        return 'UnknownCode'

def code_to_color( code: bittensor.proto.ReturnCode ) -> str:
    if code == 0: 
	    return 'bold green'
    elif code == 1:
        return 'dim yellow'
    elif code == 2:
        return 'black'
    elif code == 3:
        return 'underline red'
    elif code == 4:
        return 'red'
    elif code == 5:
        return 'black'
    elif code == 6:
        return 'red'
    elif code == 7:
        return 'red'
    elif code == 8:
        return 'red'
    elif code == 9:
        return 'red'
    elif code == 10:
        return 'red'
    elif code == 11:
        return 'red'
    elif code == 12:
        return 'red'
    elif code == 13:
        return 'red'
    elif code == 14:
        return 'red'
    elif code == 15:
        return 'red'
    elif code == 16:
        return 'yellow'
    elif code == 17:
        return 'yellow'
    elif code == 18:
        return 'red'
    elif code == 19:
        return 'red'
    elif code == 20:
        return 'red'
    elif code == 21:
        return 'red'
    else:
        return 'red'
