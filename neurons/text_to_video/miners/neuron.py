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
from typing import List, Union
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


class ModelscopeTextToVideoMiner( bittensor.BaseTextToVideoMiner ):

    @classmethod
    def check_config( clf, config: "bittensor.Config" ):
        pass

    @classmethod
    def add_args( cls, parser: "argparse.ArgumentParser"):
        parser.add_argument('--neuron.model_name', default="damo-vilab/text-to-video-ms-1.7b", type=str, help='Name of the model to run.')
        parser.add_argument('--neuron.optimize', default=True, type=bool, help='Optimize for GPU memory.')

    def __init__( self, *args, **kwargs ):
        super().__init__(*args, **kwargs)
        self.pipe = DiffusionPipeline.from_pretrained(self.config.neuron.model_name, torch_dtype=torch.float16, variant="fp16")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        if self.config.neuron.optimize:
            # optimize for GPU memory
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()

    def forward( self, text: str, num_inference_steps: int, num_frames: int, fps: int ) -> Union[ torch.FloatTensor, List[float] ]:

        video_frames = self.pipe(text, num_inference_steps=num_inference_steps, num_frames=num_frames).frames

        # TODO: no file, just pipe the bytes, see: https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/testing_utils.py
        output_file = export_to_video(video_frames, 'tmp.mp4', fps=fps) 
        buffered = BytesIO()

        with open(output_file, 'rb') as f:
            content = f.read()
            buffered.write(content)
            video = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return video


if __name__ == "__main__":
    # --- Run the miner continually until a Keyboard break ---
    bittensor.utils.version_checking()
    with ModelscopeTextToVideoMiner() as miner: 
        while True: 
            time.sleep( 1 )
