import torch
import argparse
import bittensor
import librosa
from transformers import pipeline
import numpy as np

# NOTE: Requires transformers>=4.28.1
# NOTE: Requires librosa 

# TODO: Write a BaseAudioMiner? Or some abstract base class... (BasePromptingMiner is a placeholder)
class SpeechToTextMiner( bittensor.BasePromptingMiner ):
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

    @staticmethod
    def load_audio(path, resample=True):
        y, sr = librosa.load(path, sr=None)
        if resample:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        return y


def test(path='neurons/audio/transcription/miner/urgency.mp3'):
    audio_transcriber = SpeechToTextMiner()
    audio_data = audio_transcriber.load_audio(path)
    transcription = audio_transcriber.generate_transcription(audio_data)
    print(transcription)


if __name__ == "__main__":
    bittensor.utils.version_checking()
    SpeechToTextMiner().run()
