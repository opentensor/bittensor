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
import re
from collections import defaultdict
from safetensors.torch import load_file
import copy
import os


from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from typing import List, Dict, Union, Tuple, Optional

def config():       
    parser = argparse.ArgumentParser( description = 'Stable Diffusion Text to Image Miner' )
    parser.add_argument( '--neuron.model_name', type=str, help='Name of the diffusion model to use.', default = "prompthero/openjourney-v4" )
    parser.add_argument( '--device', type=str, help='Device to load model', default="cuda:0" )
    bittensor.base_miner_neuron.add_args( parser )
    return bittensor.config( parser )

def extract_lora_models(prompt):
    pattern = r"<lora:(\w+):([\d.]+)>"
    matches = re.findall(pattern, prompt)
    lora_models = [(model, float(version)) for model, version in matches]
    clean_prompt = re.sub(pattern, "", prompt)
    return lora_models, clean_prompt.strip()

def get_project_root():
    # Get the current file's directory
    current_path = os.path.abspath(__file__)
    print(current_path, os.path.basename(current_path))
    # Traverse upwards until the "bittensor" folder is the last folder in the path
    if "bittensor" in current_path:
        # Get the index where the target string starts in the basename
        # find all occurences of the target string in the basename "bittensor"
        split = current_path.split("/")
        indices = [i for i, a in enumerate(split) if a == "bittensor"]
        # Get the index of the last occurence of the target string
        index = indices[-1]

        # recombine the path
        return "/".join(split[:-(len(split)-index)])+"/bittensor/"


        # Slice the path up to the index where the target string starts
        # return current_path[:index]+"bittensor/"
    else:
        raise Exception("Could not find project root directory")
        
def main( config ):
    bittensor.trace()

    def load_lora_weights(pipeline, checkpoint_path, multiplier, device = config.device, dtype=torch.float16):
        LORA_PREFIX_UNET = "lora_unet"
        LORA_PREFIX_TEXT_ENCODER = "lora_te"
        # load LoRA weight from .safetensors
        state_dict = load_file(os.path.join(get_project_root(),'neurons/text_to_image/loras/',checkpoint_path+".safetensors"), device=device)

        updates = defaultdict(dict)
        for key, value in state_dict.items():
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

            layer, elem = key.split('.', 1)
            updates[layer][elem] = value

        # directly update weight in diffusers model
        for layer, elems in updates.items():

            if "text" in layer:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = pipeline.text_encoder
            else:
                layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = pipeline.unet

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            # get elements for this layer
            weight_up = elems['lora_up.weight'].to(dtype)
            weight_down = elems['lora_down.weight'].to(dtype)
            alpha = elems['alpha']
            if alpha:
                alpha = alpha.item() / weight_up.shape[1]
            else:
                alpha = 1.0

            # update weight
            if len(weight_up.shape) == 4:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            else:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

        return pipeline

    # --- Build the base miner
    base_miner = bittensor.base_miner_neuron( netuid = 14, config = config )

    # --- Build diffusion pipeline ---
    # lpw_stable_diffusion is used to increase CLIP token length from 77 only works for text2img
    text2img = StableDiffusionPipeline.from_pretrained( config.neuron.model_name, custom_pipeline="lpw_stable_diffusion", safety_checker=None, torch_dtype=torch.float16).to( config.device )
    img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
    inpaint = StableDiffusionInpaintPipeline(**text2img.components)

    # --- Load Textual Inversions ---
    inversions_path = os.path.join(get_project_root(),'neurons/text_to_image/inversions/')
    files = os.listdir(inversions_path)
    for file in files:
        if file.endswith(".pt") or file.endswith(".safetensors"):
            text2img.load_textual_inversion(os.path.join(inversions_path, file))
            # only need to load the textual inversion for text2img as it shares components
            bittensor.logging.trace(f"Loaded textual inversion {file}")

    
    # Create backup default dicts to load after lora generations
    text2img_text_encoder_state_dict = copy.deepcopy(text2img.text_encoder.state_dict())
    text2img_unet_state_dict = copy.deepcopy(text2img.unet.state_dict())
    img2img_text_encoder_state_dict = copy.deepcopy(img2img.text_encoder.state_dict())
    img2img_unet_state_dict = copy.deepcopy(img2img.unet.state_dict())
    inpaint_text_encoder_state_dict = copy.deepcopy(inpaint.text_encoder.state_dict())
    inpaint_unet_state_dict = copy.deepcopy(inpaint.unet.state_dict())
    

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
            t2i = text2img
            i2i = img2img
            
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

            # extract out lora models from prompt
            lora_models, text = extract_lora_models(text)
            negative_lora_models, negative_prompt = extract_lora_models(negative_prompt)
            

            # loop all negative lora models, math abs them, then make them negative
            for i in range(len(negative_lora_models)):
                negative_lora_models[i] = (negative_lora_models[i][0], -abs(negative_lora_models[i][1]))

            if use_image:
                # turn image from base64 to PIL image
                # Load bytes into a PIL image
                image_bytes = base64.b64decode(image)
                processed_image = Image.open(BytesIO(image_bytes))


                # load lora weights
                for model, weight in lora_models:
                    i2i = load_lora_weights(i2i, model, weight)

                # load negative lora weights
                for model, weight in negative_lora_models:
                    i2i = load_lora_weights(i2i, model, weight)

                images = i2i( 
                    prompt  = text,
                    image = processed_image,
                    num_images_per_prompt = num_images_per_prompt,
                    num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    strength = strength,
                    negative_prompt = negative_prompt,
                    generator = generator
                    
                    # safety_checker = None,
                )

                # reset lora weights
                img2img.text_encoder.load_state_dict(img2img_text_encoder_state_dict)
                img2img.unet.load_state_dict(img2img_unet_state_dict)
            else:

                try:
                    bittensor.logging.trace("loading lora weights")
                    # load lora weights
                    for model, weight in lora_models:
                        t2i = load_lora_weights(t2i, model, weight)
                    
                    # load negative lora weights
                    for model, weight in negative_lora_models:
                        t2i = load_lora_weights(t2i, model, weight)
                except Exception as e:
                    bittensor.logging.error('Error loading lora weights: {}', e)
                    return "Error loading lora weights: {}".format(e)

                images = t2i( 
                    prompt = text,
                    height = height,
                    width = width,
                    num_images_per_prompt = num_images_per_prompt,
                    num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    negative_prompt = negative_prompt,
                    generator=generator,
                    # safety_checker = None,
                )

                # reset lora weights
                text2img.text_encoder.load_state_dict(text2img_text_encoder_state_dict)
                text2img.unet.load_state_dict(text2img_unet_state_dict)

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





