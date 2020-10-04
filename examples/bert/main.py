from bittensor import bittensor_pb2
import bittensor

import os, sys
import argparse
import math
import time

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

import transformers
from datasets import load_dataset, list_metrics, load_metric
from transformers import DataCollatorForNextSentencePrediction
from transformers import BertTokenizer

class BertMLMSynapse(bittensor.Synapse):
    """ An bittensor endpoint trained on wiki corpus.
    """
    def __init__(self, config):
        super(BertMLMSynapse, self).__init__(config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config = config
        
        self._bert_config = transformers.modeling_bert.BertConfig(hidden_size=256, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512, is_decoder=False)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.embeddings = transformers.modeling_bert.BertEmbeddings(self._bert_config)

        self.encoder = transformers.modeling_bert.BertEncoder(self._bert_config)

        self.pooler = transformers.modeling_bert.BertPooler(self._bert_config)

        self.student_encoder = transformers.modeling_bert.BertEncoder(self._bert_config)

        self.student_pooler = transformers.modeling_bert.BertPooler(self._bert_config)

        self.nsp = transformers.modeling_bert.BertOnlyNSPHead(self._bert_config) 

        self.nsp_loss_fct = torch.nn.CrossEntropyLoss()

    def forward_text(self, inputs: List[str]):
        return self.forward(inputs = inputs, next_inputs = None, labels = None, network = None) ['student_y']
        
    def forward(        self, 
                        inputs: List[str], 
                        next_inputs: List[str] = None, 
                        labels: torch.Tensor = None, 
                        network: torch.Tensor = None):
                
        # Tokenize the list of strings.
        if labels is not None:
            tokenized = self.tokenizer(inputs, next_inputs, return_tensors='pt', padding=True)
        else:
            tokenized = self.tokenizer(inputs, return_tensors='pt', padding=True)

        # Embed representations.
        embedding = self.embeddings(input_ids=tokenized['input_ids'], token_type_ids=tokenized['token_type_ids'])

        # Encode embeddings.
        encoding = self.encoder(embedding) #, attention_mask=tokenized['attention_mask'])

        # Pool encodings
        pooled = self.pooler(encoding[0])

        # Encode embeddings using distillation model.
        student_encoding = self.student_encoder (embedding) #, attention_mask=tokenized['attention_mask'])

        # Pool distilled encodings
        student_pooled = self.student_pooler(student_encoding[0])

        # Joined output.
        student_y = pooled + student_pooled

        # Compute distillation loss for student network.
        if network is not None:
            network_y = pooled + network

            student_distillation_loss = F.mse_loss(student_pooled, network_y) 
        else:
            student_distillation_loss = None
            network_y = None

        # Compute NSP loss for student outputs.
        if labels is not None:
            student_prediction = self.nsp(student_y)

            student_target_loss = self.nsp_loss_fct(student_prediction.view(-1, 2), labels)
        else:
            student_target_loss = None
            
        # Compute NSP loss for network outputs.
        if network is not None and labels is not None:
            network_prediction = self.nsp(network_y)

            network_target_loss = self.nsp_loss_fct(network_prediction.view(-1, 2), labels)
        else:
            network_target_loss = None
    
        return {
            'student_y': student_y,
            'network_y': network_y,
            'network_target_loss': network_target_loss,
            'student_target_loss': student_target_loss,
            'student_distillation_loss': student_distillation_loss
        }
            
def main(hparams):
    # Args
    config = bittensor.Config( hparams )
    
    # Build Synapse
    model = BertMLMSynapse(config)
    
    # Dataset
    dataset = load_dataset('bookcorpus')

    # Build and start the metagraph background object.
    # The metagraph is responsible for connecting to the blockchain
    # and finding the other neurons on the network.
    metagraph = bittensor.Metagraph( config )
    metagraph.subscribe( model ) # Adds the synapse to the metagraph.
    metagraph.start() # Starts the metagraph gossip threads.
    
    # Build and start the Axon server.
    # The axon server serves the synapse objects 
    # allowing other neurons to make queries through a dendrite.
    axon = bittensor.Axon( config )
    axon.serve( model ) # Makes the synapse available on the axon server.
    axon.start() # Starts the server background threads. Must be paired with axon.stop().
    
    # Build the dendrite and router. 
    # The dendrite is a torch object which makes calls to synapses across the network
    # The router is responsible for learning which synapses to call.
    dendrite = bittensor.Dendrite( config )
    router = bittensor.Router(x_dim = 256, key_dim = 100, topk = 10)
    
    # Optimizer.
    lr = 3.0 # learning rate
    params = list(router.parameters()) + list(model.parameters())
    optimizer = torch.optim.SGD(params, lr=lr)
    
    def train(dataset, transformer, epoch):
        model.train()  # Turn on the train mode.
        optimizer.zero_grad() # Zero out lingering gradients.

        batch_size = 10
        inputs = dataset['train'][0: batch_size]['text']
        next_inputs = dataset['train'][1: batch_size+1]['text']
        labels = torch.ones(batch_size, dtype=torch.long)
        
        # Get routing context
        context = model.forward_text( inputs )
        
        # Query the remote network.
        # Flatten mnist inputs for routing.
        synapses = metagraph.get_synapses( 1000 ) # Returns a list of synapses on the network (max 1000).
        requests, scores = router.route( synapses, context, inputs ) # routes inputs to network.
        responses = dendrite.forward_text( synapses, requests ) # Makes network calls.
        network = router.join( responses ) # Joins responses based on scores..
        
        # Compute full pass and get loss.
        output = model.forward(inputs, next_inputs, labels, network)
        
        loss = output['student_target_loss'] + output['student_distillation_loss'] + output['network_target_loss']
        loss.backward()
        optimizer.step()
          
        # Set network weights.
        weights = metagraph.getweights(synapses).to(model.device)
        weights = (0.99) * weights + 0.01 * torch.mean(scores, dim=0)
        metagraph.setweights(synapses, weights)
      
    epoch = 0
    try:
        while True:
            train(dataset, model, epoch)
            epoch += 1
    except Exception as e:
        logger.exception(e)
        metagraph.stop()
        axon.stop()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    hparams = bittensor.Config.add_args(parser)
    hparams = parser.parse_args()
    main(hparams)