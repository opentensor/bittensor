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

import asyncio
import bittensor

from enum import Enum
from typing import Union, List, Tuple, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel

class ReturnCode(Enum):
    SUCCESS = 0
    BLACKLIST = 1
    TIMEOUT = 2
    NOTVERIFIED = 3
    UNKNOWN = 4

class BaseRequest( BaseModel ):
    hotkey: str
    signature: Optional[ str ]  = None 
    priority: Optional[ float ] = None

class BaseResponse( BaseModel ):
    hotkey: str
    signature: Optional[ str ] = None
    return_code: int = ReturnCode.SUCCESS
    return_message: str = "Success"

class synapse( ABC ):
    name: str

    @abstractmethod
    def forward( self, call: BaseRequest ) -> BaseResponse:
        ...

    def verify( self, request: BaseRequest ) -> bool:
        return self.axon.verify( request )

    def priority( self, request: BaseRequest ) -> float:
        return 1.0

    def blacklist( self, request: BaseRequest ) -> bool: 
        return False

    def __init__(self, axon: 'bittensor.axon'):
        self.axon = axon
        self.axon.router.add_api_route(f"{self.name}/forward", self.apply, methods=["GET", "POST"])
        self.axon.attach( self )

    def apply( self, request: BaseRequest ) -> BaseResponse:
        try:
            if not self.verify( request ):
                return_code = ReturnCode.NOTVERIFIED
                return_message = 'Failed Signature verification'

            elif self.blacklist( request ):
                return_code = ReturnCode.BLACKLIST
                return_message = 'Blacklisted'

            else:
                request.priority = self.priority( request )
                future = self.axon.thread_pool.submit( self.forward, priority = request.priority )
                response = future.result( timeout = request.timeout )
                return_code = ReturnCode.SUCCESS
                return_message = 'Success'

        # Catch timeouts
        except asyncio.TimeoutError:
            return_code = ReturnCode.TIMEOUT
            return_message = 'GRPC request timeout after: {}s'.format( request.timeout )

        # Catch unknown exceptions.
        except Exception as e:
            return_code = ReturnCode.UNKNOWN
            return_message = str(e)

        # Finally return the call.
        finally:
            self.axon.sign( response )
            response.return_code = return_code
            response.return_message = return_message
            return response
        
   