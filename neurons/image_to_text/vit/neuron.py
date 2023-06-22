# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import io
import time
import base64
import argparse
import bittensor
import numpy as np
from PIL import Image
from typing import List, Dict, Union, Tuple, Optional
from transformers import pipeline

class VITImageToTextMiner( bittensor.BaseImageToTextMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):       
        parser.add_argument( '--model_name', type=str, help = 'model name to pull from hugging face.', default = "nlpconnect/vit-gpt2-image-captioning" )
        parser.add_argument( '--device', type=str, help='Device to load model', default="cuda:0" )
        parser.add_argument( '--netuid', type=int, help='Unique ID of the Image to Text network.', default=16 )

    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
        self.image_to_text = pipeline( "image-to-text", model = self.config.model_name, device=self.config.device )

    def forward( self, image: str ) -> str:
        # Convert base64 bytestring back into bytes
        image_bytes = base64.b64decode(image)

        # Convert bytes back into numpy array
        image_bytes_io = io.BytesIO(image_bytes)
        pixel_array = np.load(image_bytes_io, allow_pickle=True)
        pil_image = Image.fromarray(pixel_array)

        # Perform image to text generation
        generation = self.image_to_text( pil_image )
        return generation[0]['generated_text']

if __name__ == "__main__":
    bittensor.utils.version_checking()
    with VITImageToTextMiner() as miner:
        while True:
            time.sleep( 1 )
