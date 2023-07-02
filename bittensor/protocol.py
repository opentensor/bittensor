""" Create and init Axon, whcih services Forward and Backward requests from other neurons.
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

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

from enum import Enum
from pydantic import BaseModel
from typing import Dict, Optional, Tuple, Union, List, Callable

class ReturnCode(Enum):
    SUCCESS = 0
    BLACKLIST = 1
    TIMEOUT = 2
    NOTVERIFIED = 3
    UNKNOWN = 4

class BaseRequest( BaseModel ):
    name: str = 'default'
    timeout: float = 12

    # Reciever items
    receiver_hotkey: Optional[ str ] = None

    # Sender Signature items.
    sender_nonce: Optional[ str ] = None
    sender_uuid: Optional[ str ] = None
    sender_hotkey: Optional[ str ] = None
    sender_signature: Optional[ str ]  = None 

class BaseResponse( BaseModel ):
    name: str = 'default'

    return_code: int = ReturnCode.SUCCESS.value
    return_message: str = "Success"

    # Sender Signature items.
    sender_nonce: Optional[ str ] = None
    sender_uuid: Optional[ str ] = None
    sender_hotkey: Optional[ str ] = None
    sender_signature: Optional[ str ] = None 

    # Reciever Signature items.
    reciever_nonce: Optional[ str ] = None
    reciever_uuid: Optional[ str ] = None
    reciever_hotkey: Optional[ str ] = None
    reciever_signature: Optional[ str ] = None 



