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

import torch
import argparse
import bittensor
import base64
from io import BytesIO

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from typing import List, Dict, Union, Tuple, Optional

def config():       
    parser = argparse.ArgumentParser( description = 'Stable Diffusion Text to Image Miner' )
    parser.add_argument( '--neuron.model_name', type=str, help='Name of the diffusion model to use.', default = "stabilityai/stable-diffusion-2-1" )
    parser.add_argument( '--device', type=str, help='Device to load model', default="cuda:0" )
    bittensor.base_miner_neuron.add_args( parser )
    return bittensor.config( parser )

def main( config ):
    bittensor.trace()

    # --- Build the base miner
    base_miner = bittensor.base_miner_neuron( netuid = 14, config = config )

    # --- Build diffusion pipeline ---
    # lpw_stable_diffusion is used to increase CLIP token length from 77 only works for text2img
    text2img = StableDiffusionPipeline.from_pretrained( config.neuron.model_name, custom_pipeline="lpw_stable_diffusion", safety_checker=None, torch_dtype=torch.float16).to( config.device )
    img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
    inpaint = StableDiffusionInpaintPipeline(**text2img.components)

    # --- Build Synapse ---
    class StableDiffusion( bittensor.TextToImageSynapse ):

        def priority( self, forward_call: "bittensor.SynapseCall" ) -> float: 
            # return base_miner.priority( forward_call )
            return 0.0

        def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
            # return base_miner.blacklist( forward_call )
            return False
        
        def forward( self, text: str, image: str, height: int, width: int, num_images_per_prompt: int, num_inference_steps: int, guidance_scale: float, strength: float, negative_prompt: str, seed: int ) -> List[str]:
            
            use_image = False
            
            if num_images_per_prompt >= 4: # TODO: specify in config 
                return "Stable Diffusion only supports num_images_per_prompt <= 4"
            
            if image != "":
                # check if image is valid base64
                try:
                    base64.b64decode(image)
                    use_image = True
                except Exception as e:
                    pass

            if(seed == -1):
                seed = torch.randint(1000000000, (1,)).item()

            generator = torch.Generator(device=config.device).manual_seed(seed)

            if use_image:
                # turn image from base64 to PIL image
                # Load bytes into a PIL image
                image_bytes = base64.b64decode(image)
                processed_image = Image.open(BytesIO(image_bytes))

                images = img2img( 
                    text,
                    processed_image,
                    num_images_per_prompt = num_images_per_prompt,
                    num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    strength=strength,
                    negative_prompt = negative_prompt,
                    generator=generator
                    
                    # safety_checker = None,
                )
            else:
                images = text2img( 
                    text,
                    height = height,
                    width = width,
                    num_images_per_prompt = num_images_per_prompt,
                    num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    negative_prompt = negative_prompt,
                    generator=generator,
                    # safety_checker = None,
                )

            buffered = BytesIO()
            images.images[0].save(buffered, format="PNG")
            # images.images[0].show()
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return image_base64
        
    # --- Attach the synapse to the miner ----
    base_miner.axon.attach( StableDiffusion() )

    # --- Run Miner ----
    base_miner.run()

if __name__ == "__main__":
    bittensor.utils.version_checking()
    main( config() )





