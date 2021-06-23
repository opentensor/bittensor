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
import torch
from loguru import logger
logger = logger.opt(colors=True)

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
        return 'NotServingNucleus'
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
        return 'dim yellow'
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

def code_to_loguru_color( code: bittensor.proto.ReturnCode ) -> str:
    if code == 0: 
	    return 'green'
    elif code == 1:
        return 'yellow'
    elif code == 2:
        return 'yellow'
    elif code == 3:
        return 'red'
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

def rpc_log( axon: bool, forward: bool, is_response: bool, code:int, pubkey: str, inputs, outputs, message:str ):
    if axon:
        log_msg = '<white>Axon</white>     '
    else:
        log_msg = '<white>Dendrite</white> '
    if forward:
        log_msg += "<green>Forward</green> "
    else:
        log_msg += "<green>Backward</green>"
    if is_response:
        log_msg += " <green>Response</green> <--- "
    else:
        log_msg += " <green>Request</green>  ---> "
    if is_response:
        if axon:
            log_msg += "<white>to:  </white><blue>{}</blue> ".format( pubkey )
        else:
            log_msg += "<white>from:</white><blue>{}</blue> ".format( pubkey )
    else:
        if axon:
            log_msg += "<white>from:</white><blue>{}</blue> ".format( pubkey )
        else:
            log_msg += "<white>to:  </white><blue>{}</blue> ".format( pubkey )
    log_msg += "<white>code:</white>"
    code_color = code_to_loguru_color( code )
    code_string = code_to_string( code )
    code_str = "<" + code_color + ">" + code_string + "</" + code_color + ">"
    log_msg += code_str
    if inputs != None:
        if isinstance(inputs, list):
            log_msg += " <white>inputs:</white>{}".format( [list(inp.shape) for inp in inputs] )
        else:
            log_msg += " <white>inputs:</white>{}".format( list(inputs.shape) )
    if outputs != None:
        log_msg += " <white>outputs:</white>{}".format( list(outputs.shape))
    if message != None:
        log_msg += " <white>message:</white><" + code_color + ">" + message + "</" + code_color + ">"
    logger.debug( log_msg )
