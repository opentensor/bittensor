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
import time
import torch
import argparse
import bittensor
from tqdm import tqdm
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForMaskedLM
from typing import List, Dict, Union, Tuple, Optional
from datasets import load_dataset


def config():       
    parser = argparse.ArgumentParser( description = 'Template Embdding Validator.' )
    bittensor.base_validator.add_args( parser )
    bittensor.dataset.add_args( parser )
    return bittensor.config( parser )

def main( config ):
    print ( config )
    
    # --- Build the base miner ---
    base_validator = bittensor.base_validator( netuid = config.netuid, config = config )
                                       
    # --- Build the dataset ---
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')

    # --- Build embedding model ---
    class EmbeddingDecoder(nn.Module):
        def __init__(self, model_name):
            super(EmbeddingDecoder, self).__init__()
            self.bert = BertModel.from_pretrained( model_name )
            self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
            self.tokenizer = BertTokenizer.from_pretrained( model_name )

        def forward(self, embeddings: torch.FloatTensor ):
            # Run the bert model.
            outputs = self.bert( inputs_embeds = embeddings )

            # Get the hidden states from the last layer
            last_hidden_state = outputs.last_hidden_state

            # Pass each hidden state through a linear layer to predict the token id
            token_logits = self.linear( last_hidden_state )

            # Return the logits
            return token_logits
        
    # --- Build the decoder
    print ('load model')
    decoder = EmbeddingDecoder('bert-base-uncased')

    # Initialize the optimizer
    optimizer = torch.optim.Adam( decoder.parameters(), lr=1e-4 )

    # --- Run the miner continually until a Keyboard break ---
    batch_size = 1
    for i in tqdm( range(0, len(dataset['train']), batch_size) ):

        # --- Get the next batch of text ---
        batch = dataset['train'][i:i+batch_size]
        text = ''.join(batch['text'])

        # --- Tokenize the text ---
        input_ids = decoder.tokenizer.encode( text, add_special_tokens=True )
        print ('input_ids', input_ids)

        # --- Get the embeddings for the text ---
        embeddings = torch.randn( 1, decoder.bert.config.hidden_size ).repeat(1, len(input_ids), 1) 
        print ('embeddings', embeddings)

        # --- Run the model ---
        token_logits = decoder( embeddings )
        print ('token_logits', token_logits)

        # --- Calculate the loss ---
        loss = nn.CrossEntropyLoss()(token_logits.view(-1, decoder.bert.config.vocab_size), torch.tensor([input_ids]).view(-1) )
        loss.backward()
        print (loss)

        # --- Update the weights ---
        optimizer.step()

        # --- Zero the gradients ---
        optimizer.zero_grad()

    
if __name__ == "__main__":
    bittensor.utils.version_checking()
    main( config() )






