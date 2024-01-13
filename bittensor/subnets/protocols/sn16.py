# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# (developer): ETG development team
# Copyright © 2023 ETG

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
import numpy as np
import torch
from typing import Optional, List
import bittensor as bt
import pydantic

class TextToSpeech(bt.Synapse):
    """
    TextToSpeech class inherits from bt.Synapse.
    It is used to convert text to speech.
    """
    class Config:
        """
        Pydantic model configuration class for Prompting. This class sets validation of attribute assignment as True.
        validate_assignment set to True means the pydantic model will validate attribute assignments on the class.
        """

        validate_assignment = True



    # Required request input, filled by sending dendrite caller.
    text_input: Optional[str] = None
    model_name: Optional[str] = None
    clone_input: Optional[List] = None
    # Here we define speech_output as an Optional PyTorch tensor instead of bytes.
    speech_output: Optional[List] = None

    completion: str = None


    def deserialize(self) -> List:
        """
        Deserialize the speech_output into a PyTorch tensor.
        """
        # If speech_output is a tensor, just return it
        # if isinstance(self.speech_output, List):
          # print(" Deserialize the speech_output into a PyTorch tensor.",self)
        return self
        # raise TypeError("speech_output is not a tensor")


class TextToMusic(bt.Synapse):
    """
    TextToMusic class inherits from bt.Synapse.
    It is used to convert text to music.
    """
    # Required request input, filled by sending dendrite caller.
    text_input: Optional[str] = None

    # Here we define music_output as an Optional PyTorch tensor instead of bytes.
    music_output: Optional[List] = None

    completion: str = None


    def deserialize(self) -> List:
        """
        Deserialize the music_output into a PyTorch tensor.
        """
        # If music_output is a tensor, just return it
        if isinstance(self.music_output, List):
          print(" Deserialize the music_output into a PyTorch tensor.",self)
          return self
        raise TypeError("music_output is not a tensor")

    

class VoiceClone(bt.Synapse):
    """
    VoiceClone class inherits from bt.Synapse.
    It is used to clone a voice.
    """
    class Config:
        """
        Pydantic model configuration class for Prompting. This class sets validation of attribute assignment as True.
        validate_assignment set to True means the pydantic model will validate attribute assignments on the class.
        """

        validate_assignment = True

    text_input: Optional[str] = None
    clone_input: Optional[List] = None
    clone_output: Optional[List] = []
    sample_rate: Optional[int] = None
    completion: Optional[List] = None
    hf_voice_id: Optional[str] = None

    def deserialize(self) -> "VoiceClone":
        """
        Return the clone_output as bytes.
        This method would need to be further implemented based on how the bytes data is intended to be used.
        """
        return self
