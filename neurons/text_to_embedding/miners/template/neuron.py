# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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
from typing import List

class TemplateEmbeddingMiner( bittensor.BaseEmbeddingMiner ):
    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--device', type=str, help='Device to load model', default="cuda:0" )

    def __init__( self, *args, **kwargs ):
        super( TemplateEmbeddingMiner, self ).__init__( *args, **kwargs )

    def forward( self, text: List[str] ) -> torch.FloatTensor:
        # Returns a list of 0s of length 2048 for each input text string.
        return torch.FloatTensor( [ [ 0 for _ in range( 2048 ) ] for _ in text ] ).to( self.config.device )


if __name__ == "__main__":
    bittensor.utils.version_checking()
    with TemplateEmbeddingMiner():
        while True:
            time.sleep( 1 )
