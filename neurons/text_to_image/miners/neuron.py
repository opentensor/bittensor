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
import argparse
import bittensor
import base64
from io import BytesIO

from abc import ABC, abstractmethod
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from typing import List, Dict, Union, Tuple, Optional


class StableDiffusionTextToImageMiner( bittensor.BaseTextToImageMiner, ABC ):

    @classmethod
    def check_config( cls, config: "bittensor.Config" ):
        pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--neuron.model_name', type=str, help='Name of the diffusion model to use.', default = "stabilityai/stable-diffusion-2-1" )
        parser.add_argument( '--max_images_per_prompt', type=int, help='Maximum number of images per prompt', default = 4 )
        parser.add_argument( '--image_format', type=str, help='Image format to use', default = "PNG" )
        parser.add_argument( '--device', type=str, help='Device to load model', default="cuda:0" )

    def __init__( self, *args, **kwargs ):
        super( StableDiffusionTextToImageMiner, self ).__init__( *args, **kwargs )
        # --- Build diffusion pipeline ---
        # lpw_stable_diffusion is used to increase CLIP token length from 77 only works for text2img
        self.text2img = StableDiffusionPipeline.from_pretrained( 
            self.config.neuron.model_name, 
            custom_pipeline="lpw_stable_diffusion", 
            safety_checker=None, 
            torch_dtype=torch.float16
        ).to( self.config.device )
        self.img2img = StableDiffusionImg2ImgPipeline( **self.text2img.components )
        self.inpaint = StableDiffusionInpaintPipeline( **self.text2img.components )
    
    def forward(self, 
                text: str, 
                image: str, 
                height: int, 
                width: int,
                num_images_per_prompt: int, 
                num_inference_steps: int, 
                guidance_scale: float, 
                strength: float, 
                negative_prompt: str, 
                seed: int
        ) -> List[str]:
        
        use_image = False
        
        if num_images_per_prompt >= self.config.max_images_per_prompt: # TODO: specify in config 
            return "Stable Diffusion only supports num_images_per_prompt <= 4"
        
        if image != "":
            # check if image is valid base64
            try:
                base64.b64decode( image )
                use_image = True
            except Exception as e:
                pass

        if( seed == -1 ):
            seed = torch.randint( 1000000000, (1,) ).item()

        generator = torch.Generator( device=self.config.device ).manual_seed( seed )

        if use_image:
            # turn image from base64 to PIL image
            # Load bytes into a PIL image
            image_bytes = base64.b64decode( image )
            processed_image = Image.open( BytesIO( image_bytes ) )

            images = self.img2img( 
                text,
                processed_image,
                num_images_per_prompt=num_images_per_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                negative_prompt=negative_prompt,
                generator=generator
                
                # safety_checker = None,
            )
        else:
            images = self.text2img( 
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
        images.images[0].save( buffered, format=self.config.image_format )
        image_base64 = base64.b64encode( buffered.getvalue() ).decode( 'utf-8' )

        return image_base64


if __name__ == "__main__":
    bittensor.utils.version_checking()
    with StableDiffusionTextToImageMiner() as miner:
        while True:
            time.sleep( 1 )




