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

from multiprocessing.sharedctypes import Value
import bittensor
import argparse
import pytest

def test_loaded_config():
    with pytest.raises(NotImplementedError):
        bittensor.Config(loaded_config=True)

def test_strict():
    parser = argparse.ArgumentParser()
    parser.add_argument("arg", help="Dummy Args")
    parser.add_argument("--cov", help="Dummy Args")
    parser.add_argument("--cov-append", action='store_true', help="Dummy Args")
    parser.add_argument("--cov-config",  help="Dummy Args")
    bittensor.dendrite.add_args( parser )
    bittensor.logging.add_args( parser )
    bittensor.wallet.add_args( parser )
    bittensor.subtensor.add_args( parser )
    bittensor.metagraph.add_args( parser )
    bittensor.dataset.add_args( parser )
    bittensor.axon.add_args( parser )
    bittensor.wandb.add_args( parser )
    bittensor.config( parser, strict=False)
    bittensor.config( parser, strict=True)

def construct_config():
    defaults = bittensor.Config()
    bittensor.subtensor.add_defaults( defaults )
    bittensor.dendrite.add_defaults( defaults )
    bittensor.axon.add_defaults( defaults )
    bittensor.wallet.add_defaults( defaults )
    bittensor.dataset.add_defaults( defaults )
    bittensor.logging.add_defaults( defaults )
    bittensor.wandb.add_defaults( defaults )
    
    return defaults

def test_to_defaults():
    config = construct_config()
    config.to_defaults()

if __name__  == "__main__":
    test_loaded_config()
    test_strict()
    test_to_defaults()