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
from sys import prefix
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

def test_prefix():
    # Test the use of prefixes to instantiate all of the bittensor objects.
    parser = argparse.ArgumentParser()
    bittensor.dendrite.add_args( parser)
    bittensor.dendrite.add_args( parser, prefix = 'second' )

    bittensor.logging.add_args( parser )
    bittensor.logging.add_args( parser, prefix = 'second' )

    bittensor.wallet.add_args( parser )
    bittensor.wallet.add_args( parser, prefix = 'second' )

    bittensor.subtensor.add_args( parser )
    bittensor.subtensor.add_args( parser, prefix = 'second'  )

    bittensor.metagraph.add_args( parser )
    bittensor.metagraph.add_args( parser, prefix = 'second' )

    bittensor.dataset.add_args( parser )
    bittensor.dataset.add_args( parser, prefix = 'second' )

    bittensor.axon.add_args( parser )
    bittensor.axon.add_args( parser, prefix = 'second' )

    bittensor.wandb.add_args( parser )
    bittensor.wandb.add_args( parser, prefix = 'second' )

    config_non_strict = bittensor.config( parser, strict=False)
    config_strict = bittensor.config( parser, strict=True)

    bittensor.dendrite( config_strict )
    bittensor.dendrite( config_non_strict )
    bittensor.dendrite( config_strict.second )
    bittensor.dendrite( config_non_strict.second )

    bittensor.axon( config_strict )
    bittensor.axon( config_non_strict )
    bittensor.axon( config_strict.second )
    bittensor.axon( config_non_strict.second )

    bittensor.dataset( config_strict )
    bittensor.dataset( config_non_strict )
    bittensor.dataset( config_strict.second )
    bittensor.dataset( config_non_strict.second )

    bittensor.axon( config_strict )
    bittensor.axon( config_non_strict )
    bittensor.axon( config_strict.second )
    bittensor.axon( config_non_strict.second )

    bittensor.metagraph( config_strict )
    bittensor.metagraph( config_non_strict )
    bittensor.metagraph( config_strict.second )
    bittensor.metagraph( config_non_strict.second )

    bittensor.wallet( config_strict )
    bittensor.wallet( config_non_strict )
    bittensor.wallet( config_strict.second )
    bittensor.wallet( config_non_strict.second )

    bittensor.logging( config_strict )
    bittensor.logging( config_non_strict )
    bittensor.logging( config_strict.second )
    bittensor.logging( config_non_strict.second )

    bittensor.wandb( config_strict )
    bittensor.wandb( config_non_strict )
    bittensor.wandb( config_strict.second )
    bittensor.wandb( config_non_strict.second )


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