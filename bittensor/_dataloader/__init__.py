

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


import argparse
import bittensor
import copy

from . import dataloader_impl

class dataloader:

    def __new__(
            cls,
            block_size: int = 20,
            batch_size: int = 10,
            max_corpus_size:int = 1e+6,
            num_workers: int = 0
        ):
        assert batch_size > 0, 'Batch size must be larger than 0'
        assert block_size > 0, 'Block size must be larger than 0'
        assert max_corpus_size > 0, 'max_corpus_size must be larger than 0'
        assert num_workers >= 0, 'num_workers must be equal to or larger than 0'
        return dataloader_impl.GenesisTextDataloader(
            block_size = block_size,
            batch_size = batch_size,
            max_corpus_size = max_corpus_size,
            num_workers = num_workers
        )

