import os
import glob
import time
import torch
import argparse
import base64
import bittensor
from io import BytesIO
import soundfile as sf
from tortoise import utils, api
from typing import Optional, Union, Tuple


def config():       
    parser = argparse.ArgumentParser( description = 'Tortoise Miner' )
    parser.add_argument( '--device', type=str, help='Device to load model', default="cuda" )
    parser.add_argument( '--default_clips_path', type=str, help='Path to WAV file(s) for latent conditioning.', default=None)
    bittensor.base_miner_neuron.add_args( parser )
    return bittensor.config( parser )

def main( config ):
    bittensor.trace()
    print ( config )

    # --- Build the base miner
    base_miner = bittensor.base_miner_neuron( netuid = 13, config = config )

    # --- Load default reference clips ---
    if config.default_clips_path is not None and os.path.isdir(config.default_clips_path):
        default_clips_path = glob.glob(os.path.join(config.default_clips_path, "*.wav"))
    elif os.path.isfile(config.default_clips_path) and config.default_clips_path.endswith(".wav"):
        default_clips_path = [config.default_clips_path]
    else:
        raise ValueError("Invalid default_clips_path: {}".format(config.default_clips_path))

    # --- Build text to speech pipeline ---
    default_reference_clips = [utils.audio.load_audio(p, 22050) for p in default_clips_path]
    model = api.TextToSpeech()

    # --- Build the synapse ---
    class TortoiseTextToSpeechSynapse( bittensor.TextToSpeechSynapse ):

        def priority( self, forward_call: "bittensor.SynapseCall" ) -> float: 
            return 0.0

        def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
            return False
        
        def forward( self, text: str, reference_clips: Optional[torch.FloatTensor] = None ) -> bytes:

            # Perform the generation
            pcm_audio = model.tts_with_preset(
                text, 
                voice_samples=reference_clips or default_reference_clips, 
                preset='fast'
            )
            # Convert audio to bytes and write to buffer
            audio_buffer = BytesIO()
            sf.write( audio_buffer, pcm_audio.numpy().squeeze(), 22050, format='WAV' )

            # Convert buffer to base64 string
            audio_base64 = base64.b64encode( audio_buffer.getvalue() ).decode('utf-8')
            return audio_base64
        
    # --- Attach the synapse to the base miner ---
    text_to_speech_synapse = TortoiseTextToSpeechSynapse()
    base_miner.axon.attach( text_to_speech_synapse )

    # --- Run miner continually until Keyboard break ---
    with base_miner: 
        while True: 
            time.sleep( 1 )

if __name__ == "__main__":
    bittensor.utils.version_checking()
    main( config() )
