# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import bittensor
from typing import List, Dict, Union, Tuple, Optional
from transformers import pipeline

def config():       
    parser = argparse.ArgumentParser( description = 'Captioning Image to Text miner.' )
    parser.add_argument( '--model_name', type=str, help = 'model name to pull from hugging face.', default = "nlpconnect/vit-gpt2-image-captioning" )
    parser.add_argument( '--device', type=str, help='Device to load model', default="cuda:0" )
    bittensor.base_miner_neuron.add_args( parser )
    return bittensor.config( parser )

def main( config ):

    config.netuid = 16

    # --- Build the base miner
    base_miner = bittensor.base_miner_neuron( netuid = 16, config = config )

    # --- Build image to text pipeline ---
    image_to_text = pipeline( "image-to-text", model = config.model_name ).to( config.device )

    # --- Build Synapse ---
    class CaptioningImageToTextSynapse( bittensor.ImageToTextSynapse ):

        def priority( self, forward_call: "bittensor.SynapseCall" ) -> float: 
            return base_miner.priority( forward_call )

        def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
            return base_miner.blacklist( forward_call )
        
        def forward( self, image: bytes ) -> str:
            return image_to_text( image )
        
    # --- Attach the synapse to the miner ----
    base_miner.axon.attach( CaptioningImageToTextSynapse() )

    # --- Run Miner ----
    base_miner.run()

if __name__ == "__main__":
    bittensor.utils.version_checking()
    main( config() )





