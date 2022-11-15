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
from . import constant
from unittest.mock import MagicMock
logging = bittensor.logging()

def test_construct_text_corpus():
    for run_generator in [True, False]: 
        dataset = bittensor.dataset(num_batches = constant.dataset.num_batches, save_dataset = True, dataset_name = constant.dataset.dataset_name, run_generator=run_generator)
        dataset.construct_text_corpus()
        dataset.close()

def test_change_data_size():
    # only for run_generator is False
    data_sizes = [(10,1000), (15, 2000),(30, 3000), (25,4000)]
    dataset = bittensor.dataset(num_batches = constant.dataset.num_batches, dataset_name = constant.dataset.dataset_name, run_generator=False, no_tokenizer=False)
    for data_size in data_sizes:
        dataset.set_data_size(*data_size)
        sample_dict = next(dataset)
        for k,v in sample_dict.items():
            v.shape[0] == data_size[0]
        
    dataset = bittensor.dataset(num_batches = constant.dataset.num_batches, dataset_name = constant.dataset.dataset_name, run_generator=False, no_tokenizer=True)

    for data_size in data_sizes:
        raw_text_sample = next(dataset)
        len(raw_text_sample)  == data_size[1]
    
    dataset.close() 

def test_next_tokenized_sample():
    batch_size = 10
    sequence_length = 128
    block_size = 1000
    num_batches = 10
    for run_generator in [True, False]:
        

        dataset = bittensor.dataset (
            block_size = block_size,
            batch_size = batch_size,
            sequence_length = sequence_length,
            num_batches=num_batches,
            run_generator = run_generator,
            no_tokenizer=False
        )


        input = next(dataset)
        assert input['input_ids'].shape[0] == input['attention_mask'].shape[0] == batch_size
        assert input['input_ids'].shape[1] == input['attention_mask'].shape[1] == sequence_length
        dataset.close()


def test_next_raw_sample():
    batch_size = 10
    sequence_length = 128
    block_size = 1000
    num_batches = 10
    for run_generator in [True, False]:
        dataset = bittensor.dataset (
            block_size = block_size,
            batch_size = batch_size,
            sequence_length = sequence_length,
            num_batches=num_batches,
            run_generator = run_generator,
            no_tokenizer = True
        )

        input = next(dataset)
        assert len(input) == batch_size
        for i in range(len(input)):
            assert len(input[i].split()) == sequence_length

        dataset.close()


def test_mock():
    dataset = bittensor.dataset(_mock=True, dataset_name = constant.dataset.dataset_name)
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
    dataset = bittensor.dataset(num_batches = constant.dataset.num_batches, dataset_name = constant.dataset.dataset_name)
    dataset.requests_retry_session = MagicMock(return_value = None)
    next(dataset)
    next(dataset)
    next(dataset)
    dataset.close()

if __name__ == "__main__":
    test_change_data_size()