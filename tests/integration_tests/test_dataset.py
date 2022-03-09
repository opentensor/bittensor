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

import bittensor
from unittest.mock import MagicMock

logging = bittensor.logging()

def test_construct_text_corpus():
    # text corpus for the train set
    dataset = bittensor.dataset(max_corpus_size = 1000, save_dataset = True)
    dataset.construct_text_corpus()
    dataset.close()

def test_next():
    dataset = bittensor.dataset(max_corpus_size = 1000)
    next(dataset)
    next(dataset)
    next(dataset)
    dataset.close()

def test_mock():
    dataset = bittensor.dataset(_mock=True)
    next(dataset)
    next(dataset)
    next(dataset)
    dataset.close()

def test_mock_function():
    dataset = bittensor.dataset.mock()
    next(dataset)
    next(dataset)
    next(dataset)
    dataset.close()

def test_fail_IPFS_server():
    dataset = bittensor.dataset()
    next(dataset)
    dataset.requests_retry_session = MagicMock(return_value = None)
    assert dataset.construct_text_corpus() == []
    
    

if __name__ == "__main__":
    test_fail_IPFS_server()