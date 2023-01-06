""" utils for rpc log, convert return code to string with color for the log
"""
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

from loguru import logger
import bittensor 

logger = logger.opt(colors=True)

def code_to_string( code: 'bittensor.proto.ReturnCode' ) -> str:
    """ Return code -> string
    """
    if code == 0: 
        return 'NoReturn'
    elif code == 1: 
        return 'Success'
    elif code == 2:
        return 'Timeout'
    elif code == 3:
        return 'Backoff'
    elif code == 4:
        return 'Unavailable'
    elif code == 5:
        return 'NotImplemented'
    elif code == 6:
        return 'EmptyRequest'
    elif code == 7:
        return 'EmptyResponse'
    elif code == 8:
        return 'InvalidResponse'
    elif code == 9:
        return 'InvalidRequest'
    elif code == 10:
        return 'RequestShapeException'
    elif code == 11:
        return 'ResponseShapeException'
    elif code == 12:
        return 'RequestSerializationException'
    elif code == 13:
        return 'ResponseSerializationException'
    elif code == 14:
        return 'RequestDeserializationException'
    elif code == 15:
        return 'ResponseDeserializationException'
    elif code == 16:
        return 'NotServingNucleus'
    elif code == 17:
        return 'NucleusTimeout'
    elif code == 18:
        return 'NucleusFull'
    elif code == 19:
        return 'RequestIncompatibleVersion'
    elif code == 20:
        return 'ResponseIncompatibleVersion'
    elif code == 21:
        return 'SenderUnknown'
    elif code == 22:
        return 'UnknownException'
    elif code == 23:
        return 'Unauthenticated'
    elif code == 24:
        return 'BadEndpoint'
    else:
        return 'UnknownCode'

def code_to_loguru_color( code: 'bittensor.proto.ReturnCode' ) -> str:
    """ Return code -> loguru color
    """
    if code == 0: 
        return 'red'
    elif code == 1: 
        return 'green'
    elif code == 2:
        return 'yellow'
    elif code == 3:
        return 'yellow'
    elif code == 4:
        return 'red'
    elif code == 5:
        return 'red'
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
        return 'red'
    elif code == 17:
        return 'yellow'
    elif code == 18:
        return 'yellow'
    elif code == 19:
        return 'red'
    elif code == 20:
        return 'red'
    elif code == 21:
        return 'red'
    elif code == 22:
        return 'red'
    else:
        return 'red'

def code_to_synapse( code: 'bittensor.proto.Synapse.SynapseType'):
    """Return Code -> Synapse Type"""
    if code == 1:
        return 'text_last_hidden_state'
    elif code == 2:
        return 'text_causal_lm'
    elif code == 3:
        return 'text_seq_2_seq'
    elif code == 4:
        return 'text_causal_lm_next'
    else:
        return 'Null'