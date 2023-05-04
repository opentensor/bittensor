
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")

import torch
import bittensor
from diffusers import StableDiffusionPipeline

class TextToImageMiner( bittensor.BasePromptingMiner ):
    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--stt.model_name', type=str, help='Name/path of the ASR model to load', default="openai/whisper-large" )
        parser.add_argument( '--stt.chunk_length_s', type=int, help='Audio chunk length in seconds', default=30 )
        parser.add_argument( '--stt.device', type=str, help='Device to load model', default="cuda:0" )

    def __init__(self):
        super( SpeechToTextMiner, self ).__init__()
        print ( self.config )
        
        bittensor.logging.info( 'Loading ' + str(self.config.stt.model_name))
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.config.stt.model_name,
            chunk_length_s=self.config.stt.chunk_length_s,
            device=self.config.stt.device,
        )
        bittensor.logging.info( 'Model loaded!' )

    def forward(self, filepath: str) -> str:
        audio = self.load_audio(filepath)
        return self.generate_transcription(audio)

    def generate_transcription(self, audio: np.array) -> str:
        prediction = self.pipe(audio, return_timestamps=True)

        if isinstance(prediction, dict):
            if 'text' in prediction.keys():
                return prediction['text']
        elif isinstance(prediction, list):
            processor = self.pipe.tokenizer
            transcriptions = []
            for chunk in prediction:
                predicted_ids = torch.tensor(chunk["token_ids"]).unsqueeze(0).to(self.config.stt.device)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
                transcriptions.append(transcription[0])
            return transcriptions


if __name__ == "__main__":
    bittensor.utils.version_checking()
    SpeechToTextMiner().run()



# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import argparse
import bittensor

from diffusers import StableDiffusionPipeline
from typing import List, Dict, Union, Tuple, Optional

def config():       
    parser = argparse.ArgumentParser( description = 'Stable Diffusion Text to Image Miner' )
    parser.add_argument( '--model_name', type=str, help='Name of the diffusion model to use.', default = "runwayml/stable-diffusion-v1-5" )
    parser.add_argument( '--device', type=str, help='Device to load model', default="cuda:0" )
    bittensor.base_miner_neuron.add_args( parser )
    return bittensor.config( parser )

def main( config ):

    # --- Build the base miner
    base_miner = bittensor.base_miner_neuron( config = config )

    # --- Build diffusion pipeline ---
    pipe = StableDiffusionPipeline.from_pretrained( config.model_name, torch_dtype=torch.float16).to( config.device )

    # --- Build Synapse ---
    class SDTextToImageSynapse( bittensor.TextToImageSynapse ):

        def priority( self, forward_call: "bittensor.SynapseCall" ) -> float: 
            return base_miner.priority( forward_call )

        def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
            return base_miner.blacklist( forward_call )
        
        def forward( self, text: str ) -> bytes:
            image = pipe( text ).images[0] 
            return image
        
    # --- Attach the synapse to the miner ----
    base_miner.axon.attach( SDTextToImageSynapse() )

    # --- Run Miner ----
    base_miner.run()

if __name__ == "__main__":
    bittensor.utils.version_checking()
    main( config() )





