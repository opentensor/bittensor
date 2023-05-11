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
from typing import List, Dict, Union, Tuple, Optional
import base64
from io import BytesIO


from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import BACKENDS_MAPPING
from diffusers.utils.import_utils import is_opencv_available
import numpy as np
import tempfile


def export_to_video(video_frames: List[np.ndarray], output_video_path: str = None, fps: int = 8) -> str:
    if is_opencv_available():
        import cv2
    else:
        raise ImportError(BACKENDS_MAPPING["opencv"][1].format("export_to_video"))
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    return output_video_path


def config():       
    parser = argparse.ArgumentParser( description = 'Template Embdding miner.' )
    parser.add_argument('--neuron.model_name', default="damo-vilab/text-to-video-ms-1.7b", type=str, help='Name of the model to run.')
    parser.add_argument('--neuron.optimize', default=True, type=bool, help='Optimize for GPU memory.')
    bittensor.base_miner_neuron.add_args( parser )
    return bittensor.config( parser )

def main( config ):
    print ( config )
    
    # --- Build the base miner
    base_miner = bittensor.base_miner_neuron( netuid = 15, config = config )
    pipe = DiffusionPipeline.from_pretrained(config.neuron.model_name, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if config.neuron.optimize:
        # optimize for GPU memory
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

    # --- Build the synapse ---
    class ModelscopeTextToVideo( bittensor.TextToVideoSynapse ):

        def priority( self, forward_call: "bittensor.SynapseCall" ) -> float: 
            # return base_miner.priority( forward_call )
            return 0.0
        
        def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
            # return base_miner.blacklist( forward_call )
            return False
        
        def forward( self, text: List[str], num_inference_steps: int, num_frames: int, fps: int ) -> Union[ torch.FloatTensor, List[float] ]:
            test_text = {
                'text': text,
            }
            video_frames = pipe(text, num_inference_steps=num_inference_steps, num_frames=num_frames).frames
            output_file = export_to_video(video_frames, 'tmp.mp4', fps=fps) #TODO: no file, just pipe the bytes, see: https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/testing_utils.py
            buffered = BytesIO()

            with open(output_file, 'rb') as f:
                content = f.read()
                video = base64.b64encode(content).decode('utf-8')
            
            return video

    # --- Attach the synapse to the base miner ---
    base_miner.attach( ModelscopeTextToVideo() )

    # --- Run the miner continually until a Keyboard break ---
    with base_miner: 
        while True: 
            time.sleep( 1 )

if __name__ == "__main__":
    bittensor.utils.version_checking()
    main( config() )




