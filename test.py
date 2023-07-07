import pydantic
import bittensor as bt
from fastapi import Request
bt.debug()

class Forward( bt.BaseRequest ):
    input: int = pydantic.Field(..., allow_mutation=False)
    output: int = None

    class Config:
        validate_assignment = True

def forward( request: Forward ) -> Forward:
    request.input += 1
    return request

def verify( request: Request ):
    pass

def blacklist( request: Request ):
    return False

axon = bt.axon()
axon.attach( forward, verify_fn = verify, blacklist_fn = blacklist )
axon.start()
import time; time.sleep(1)
d  = bt.dendrite()
resp = await d( [ axon ], Forward( input = 1 ) )
print (resp)