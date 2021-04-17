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
import torch
import random

class TextCorpus():
    def __init__(
            self,             
            dataset,
            tokenizer, 
            block_size: int, 
        ):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.dataset = dataset

    def __len__(self):
        return len( self.dataset ) - self.block_size

    def __getitem__(self, idx):
        """ Returns a batch of sentences from text dataset.
            Args:
                idx: index of data input
            Returns:
                x
        """
        chunk = self.dataset[idx:idx + self.block_size]['sentence']
        dix = []
        block_num=0
        while block_num < self.block_size:
            tokenized = self.tokenizer(chunk[block_num], padding=True, truncation=True)['input_ids']
            for t in tokenized:
                if block_num < self.block_size:
                    dix.append(t)
                    block_num += 1
        x = torch.tensor(dix, dtype=torch.long)
        return x


class MLMCorpus():
    def __init__(
            self,             
            dataset,
            tokenizer, 
            collator,
        ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.collator = collator

    def next_batch( self, batch_size:int ) -> dict:
        """ Returns a random batch from text dataset with 50 percent NSP.            
            Returns:
                tensor_batch torch.Tensor (batch_size, sequence_length): List of tokenized sentences.
                labels torch.Tensor (batch_size, sequence_length)
        """
        batch_text = []
        for _ in range( batch_size ):
            batch_text.append(self.dataset[random.randint(0, len(self.dataset))]['sentence'])

        # Tokenizer returns a dict { 'input_ids': list[], 'attention': list[] }
        # but we need to convert to List [ dict ['input_ids': ..., 'attention': ... ]]
        # annoying hack...
        tokenized = self.tokenizer( batch_text )
        tokenized = [dict(zip(tokenized,t)) for t in zip(*tokenized.values())]

        # Produces the masked language model inputs aw dictionary dict {'inputs': tensor_batch, 'labels': tensor_batch}
        # which can be used with the Bert Language model. 
        collated_batch =  self.collator(tokenized)
        return {'inputs': collated_batch['input_ids'], 'labels': collated_batch['labels']}

class NSPCorpus:
    def __init__(
            self,             
            dataset,
            tokenizer, 
        ):
        self.tokenizer = tokenizer
        self.dataset = dataset

    def next_batch( self, batch_size:int ) -> dict:
        """ Returns a random batch from text dataset with 50 percent NSP.            
            Returns:
                inputs List[str]: List of sentences.
                targets torch.Tensor(batch_size): 1 if random next sentence, otherwise 0.
        """
        batch_inputs = []
        batch_next = []
        batch_labels = []
        for _ in range( batch_size ):
            if random.random() > 0.5:
                pos = random.randint(0, len(self.dataset))
                batch_inputs.append(self.dataset[pos]['sentence'])
                batch_next.append(self.dataset[pos + 1]['sentence'])
                batch_labels.append(0)
            else:
                while True:
                    pos_1 = random.randint(0, len(self.dataset))
                    pos_2 = random.randint(0, len(self.dataset))
                    batch_inputs.append(self.dataset[pos_1]['sentence'])
                    batch_next.append(self.dataset[pos_2]['sentence'])
                    batch_labels.append(1)
                    if (pos_1 != pos_2) and (pos_1 != pos_2 - 1):
                        break
        tokenized_inputs = self.tokenizer(batch_inputs, text_pair = batch_next, return_tensors='pt', padding=True)
        return {'inputs': tokenized_inputs['input_ids'], 'attention_mask': tokenized_inputs['attention_mask'], 'targets': torch.tensor(batch_labels, dtype=torch.long)}