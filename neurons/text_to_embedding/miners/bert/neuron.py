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
from typing import List, Dict, Union, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling( model_output, attention_mask ):
    token_embeddings = model_output[ 0 ]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze( -1 ).expand( token_embeddings.size() ).float()
    return torch.sum( token_embeddings * input_mask_expanded, 1 ) / torch.clamp( input_mask_expanded.sum( 1 ), min=1e-9 )

class BertEmbeddingMiner( bittensor.BaseMinerNeuron ):

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--model_name', default='bert-base-cased', choices=['bert-base-cased', 'bert-base-uncased', 'sentence-transformers/all-MiniLM-L6-v2'], type=str, help='Name of the model to use for embedding' )
        parser.add_argument( '--device', type=str, help='Device to load model', default="cuda:0" )

    def __init__( self, *args, **kwargs ):
        super( BertEmbeddingMiner, self ).__init__( *args, **kwargs )
        self.tokenizer = AutoTokenizer.from_pretrained( self.config.model_name )  
        self.model = AutoModel.from_pretrained( self.config.model_name ).to( self.config.device )

    def forward( self, text: List[str] ) -> torch.FloatTensor:
        encoded_input = self.tokenizer( text, padding=True, truncation=True, return_tensors='pt' ).to( self.config.device )
        with torch.no_grad(): model_output = self.model( **encoded_input )
        sentence_embeddings = mean_pooling( model_output, encoded_input['attention_mask'] )
        sentence_embeddings = F.normalize( sentence_embeddings, p=2, dim=1 )
        return sentence_embeddings


if __name__ == "__main__":
    bittensor.utils.version_checking()
    with BertEmbeddingMiner():
        while True:
            time.sleep( 1 )