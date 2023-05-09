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
import torch
import bittensor
from typing import List, Dict, Union, Tuple

bittensor.logging( bittensor.logging.config() )

class TextToSpeechSynapse( bittensor.TextToSpeechSynapse ):

    def priority( self, forward_call: "bittensor.SynapseCall" ) -> float:
        return 0.0

    def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
        return False
    
    def forward( self, text: str ) -> bytes:
        return b'these bytes dont really represent speech'

class TextToImageSynapse( bittensor.TextToImageSynapse ):

    def priority( self, forward_call: "bittensor.SynapseCall" ) -> float:
        return 0.0

    def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
        return False
    
    def forward( self, text: str ) -> bytes:
        return b'these bytes dont really represent an image'
    
class SpeechToTextSynapse( bittensor.SpeechToTextSynapse ):

    def priority( self, forward_call: "bittensor.SynapseCall" ) -> float:
        return 0.0

    def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
        return False
    
    def forward( self, speech: bytes ) -> str:
        return "this is what was said."

class ImageToTextSynapse( bittensor.ImageToTextSynapse ):

    def priority( self, forward_call: "bittensor.SynapseCall" ) -> float:
        return 0.0

    def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
        return False
    
    def forward( self, image: bytes ) -> str:
        return "this is a dog with blue eyes"
    
class TextPromptingSynapse( bittensor.TextPromptingSynapse ):
    def priority(self, forward_call: "bittensor.SynapseCall") -> float:
        return 0.0

    def blacklist(self, forward_call: "bittensor.SynapseCall") -> Union[ Tuple[bool, str], bool ]:
        return False

    def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str:
        pass

    def forward(self, messages: List[Dict[str, str]]) -> str:
        return "hello im a chat bot."

    def multi_forward(self, messages: List[Dict[str, str]]) -> List[ str ]:
        return ["hello im a chat bot.", "my name is bob" ]


# Create a mock wallet.
wallet = bittensor.wallet().create_if_non_existent()
axon = bittensor.axon( wallet = wallet, port = 9090, ip = "127.0.0.1" )
text_to_speech = bittensor.text_to_speech( axon = axon.info(), keypair = wallet.hotkey )
speech_to_text = bittensor.speech_to_text( axon = axon.info(), keypair = wallet.hotkey )
text_to_image = bittensor.text_to_image( axon = axon.info(), keypair = wallet.hotkey )
image_to_text = bittensor.image_to_text( axon = axon.info(), keypair = wallet.hotkey )
text_prompting = bittensor.text_prompting( axon = axon.info(), keypair = wallet.hotkey )
axon.attach( ImageToTextSynapse() )
axon.attach( TextToImageSynapse() )
axon.attach( TextToSpeechSynapse() )
axon.attach( SpeechToTextSynapse() )
axon.attach( TextPromptingSynapse() )

# Start the server and then exit after 50 seconds.
axon.start()
print( 'text = ', image_to_text( b"the bytes of an image of a dog with blue eyes." ).text )
print( 'text = ', speech_to_text( b"lala I am singing a song (this does not actually represent speech.)" ).speech )
print( 'image = ', text_to_image( "a dog with blue eyes" ).image )
print( 'speech = ', text_to_speech( "lalala I am singing a song." ).speech )
print( 'completion = ', text_prompting( "what is the capital of Texas?" ).completion )
time.sleep(100)
axon.stop()