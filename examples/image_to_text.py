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
from typing import List, Dict, Union, Tuple

bittensor.logging( bittensor.logging.config() )

class ImageToTextSynapse( bittensor.ImageToTextSynapse ):
    def priority( self, forward_call: "bittensor.SynapseCall" ) -> float:
        return 0.0

    def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
        return False
    
    def forward( self, image: bytes ) -> str:
        return "this is what was in the image represented by bytes =" + str(image)

# Create a mock wallet.
wallet = bittensor.wallet( config = bittensor.wallet.config() ).create_if_non_existent()
axon = bittensor.axon( wallet = wallet, port = 9090, ip = "127.0.0.1" )
image_to_text = bittensor.image_to_text( axon = axon.info(), keypair = wallet.hotkey )
axon.attach( ImageToTextSynapse() )

# Start the server and then exit after 50 seconds.
axon.start()
image = b"the bytes of an image of a dog with blue eyes."
print ( 'image =', image )
print( 'text =', image_to_text( image ).text )
time.sleep(50)
axon.stop()