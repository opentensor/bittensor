"""BERT Next Sentence Prediction Synapse

This file demonstrates a bittensor.Synapse trained for Next Sentence Prediction.

Example:
        $ python examples/bert/main.py

"""

from bittensor import bittensor_pb2
import bittensor

import argparse
from datasets import load_dataset, list_metrics, load_metric
from loguru import logger
import os, sys
import math
import random
import time
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import DataCollatorForNextSentencePrediction
from transformers import BertTokenizer
from typing import List, Tuple, Dict, Optional

class BertNSPSynapse(bittensor.Synapse):
    """ An bittensor synapse endpoint traiing as BERT transformer using Next Sentence Prediction (NSP) 
    on the bookscorpus dataset.
    """

    def __init__(self, config: transformers.modeling_bert.BertConfig):
        super(BertNSPSynapse, self).__init__()                
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embeddings = transformers.modeling_bert.BertEmbeddings(self.config)
        self.encoder = transformers.modeling_bert.BertEncoder(self.config)
        self.pooler = transformers.modeling_bert.BertPooler(self.config)
        self.student_encoder = transformers.modeling_bert.BertEncoder(self.config)
        self.student_pooler = transformers.modeling_bert.BertPooler(self.config)
        self.nsp = transformers.modeling_bert.BertOnlyNSPHead(self.config) 
        self.nsp_loss_fct = torch.nn.CrossEntropyLoss()
        self.device

    def forward_text(self, inputs: List[str]):
        """ Forward pass inputs and labels through the NSP BERT module.

            Args:
                inputs (List[str]): batch_size length list of text sentences.
            
            Returns:
                local_output torch.Tensor(n, config.hidden_dim): (Required) Output encoding of inputs 
                    produced by using the local student distillation model as context.
        """
        return self.forward(sentences = inputs, 
                            next_sentences = None, 
                            next_sentence_labels = None, 
                            network = None) ['local_output']
        
    def forward(    self, 
                    sentences: List[str], 
                    next_sentences: List[str] = None, 
                    next_sentence_labels: torch.Tensor = None, 
                    network: torch.Tensor = None):
    
        r""" Forward pass inputs and labels through the NSP BERT module.

            Args:
                sentences (:obj:`List[str]` of shape :obj:`(batch_size)`, `required`): 
                    Batch_size length list of text sentences.

                next_sentences (:obj:`List[str]` of shape :obj:`(batch_size)`, `optional`): 
                    Batch_size length list of (potential) next sentences.

                next_sentence_labels (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
                    Labels for computing the next sequence prediction (classification) loss. 
                    Indices should be in ``[0, 1]``:
                        - 0 indicates sequence B is a continuation of sequence A,
                        - 1 indicates sequence B is a random sequence.

                network (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.hidden_size)`, `optional`):
                    response context from a bittensor dendrite query. 

            Returns:
                dictionary with { 
                    loss  (:obj:`List[str]` of shape :obj:`(batch_size)`, `required`):
                        Total loss acumulation to be used by loss.backward()

                    local_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.hidden_dim)`, `required`):
                        Output encoding of inputs produced by using the local student distillation model as 
                        context rather than the network. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Next sentence prediction loss computed using the local_output and with respect to the passed labels.

                    network_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.hidden_dim)`, `optional`): 
                        Output encoding of inputs produced by using the network inputs as context to the local model rather than 
                        the student.

                    network_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):  
                        Next sentence prediction loss computed using the network_output and with respect to the passed labels.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss produced by the student with respect to the network context.
                }
        """

        # Return vars.
        loss = torch.tensor(0.0)
        local_output = None
        network_output = None
        network_target_loss = None
        local_target_loss = None
        distillation_loss = None
                    
        # Tokenize inputs: dict
        #  tokenized = dict {
        #       'input_ids': torch.Tensor(batch_size, max_sequence_len),
        #       'token_type_ids': torch.Tensor(batch_size, max_sequence_len),
        #       'attention_mask': torch.Tensor(batch_size, max_sequence_len)
        # }
        if next_sentence_labels is not None:
            # During training we tokenize both sequences and return token_type_ids which tell
            # the model which token belongs to which sequence.
            # i.e tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]),
            tokenized = self.tokenizer(sentences, next_sentences, return_tensors='pt', padding=True).to(self.device)
        else:
            # During inference we only tokenize the inputs, padding them to the longest sequence len.
            tokenized = self.tokenizer(sentences, return_tensors='pt', padding=True).to(self.device)

        # Embed tokens into a common dimension.
        # embedding = torch.Tensor(batch_size, max_sequence_len, config.hidden_size)
        embedding = self.embeddings(input_ids=tokenized['input_ids'], token_type_ids=tokenized['token_type_ids'])

        # Bert transformer returning the last hidden states from the transformer model.
        # encoding = List [
        #   hidden_states = torch.Tensor(batch_size, max_sequence_len, config.hidden_size), 
        # ]
        #import pdb; pdb.set_trace()
        encoding = self.encoder(embedding)

        # Pooling, "pool" the model by simply taking the hidden state corresponding
        # to the first token. first_token_tensor = encoding[:, 0]. Applies a dense linear
        # layer to the encoding for the first token. 
        # pooled = torch.Tensor (batch_size, config.hidden_size)
        pooled = self.pooler(encoding[0])
        #import pdb; pdb.set_trace()
        # Student transformer model which learns a mapping from the embedding to the network inputs
        # student_pooled = torch.Tensor (batch_size, config.hidden_size)
        student_encoding = self.student_encoder (embedding.detach())
        student_pooled = self.student_pooler(student_encoding[0])
        if network is not None:
            # Distillation loss between student_pooled and network inputs.
            distillation_loss = F.mse_loss(student_pooled, network)
            loss = loss + distillation_loss

        # Output from the forward pass using only the local and student models.
        # local_ouput = torch.Tensor ( batch_size, config.hidden_size)
        local_output = pooled + student_pooled
        if next_sentence_labels is not None:
            # Compute the NSP loss by projecting the output to torch.Tensor(2)
            # logit(1) > logit(0) if next_inputs are the real next sequences.
            local_prediction = self.nsp(local_output).to(self.device)
            local_prediction = F.softmax(local_prediction, dim=1)
            local_target_loss = self.nsp_loss_fct(local_prediction.view(-1, 2), next_sentence_labels.to(self.device))
            loss = loss + local_target_loss
            
        # Compute NSP loss for network outputs. Only run this if we have passed network inputs.
        if network is not None and next_sentence_labels is not None:
            # Compute the NSP loss by projecting the network_output to torch.Tensor(2)
            # logit(1) > logit(0) if next_inputs are the real next sequences.
            network_output = pooled + network
            network_prediction = self.nsp(network_output).to(self.device)
            network_prediction = F.softmax(network_prediction, dim=1)
            network_target_loss = self.nsp_loss_fct(network_prediction.view(-1, 2), next_sentence_labels.to(self.device))
            loss = loss + network_target_loss
    
        return {
            'loss': loss,
            'local_output': local_output,
            'local_target_loss': local_target_loss,
            'network_output': network_output,
            'network_target_loss': network_target_loss,
            'distillation_loss': distillation_loss
        }

def nsp_batch(data, batch_size):
    """ Returns a random batch from text dataset with 50 percent NSP.

        Args:
            data: (List[dict{'text': str}]): Dataset of text inputs.
            batch_size: size of batch to create.
        
        Returns:
            batch_inputs List[str]: List of sentences.
            batch_next List[str]: List of (potential) next sentences 
            batch_labels torch.Tensor(batch_size): 1 if random next sentence, otherwise 0.
    """
    batch_inputs = []
    batch_next = []
    batch_labels = []
    for _ in range(batch_size):
        if random.random() > 0.5:
            pos = random.randint(0, len(data))
            batch_inputs.append(data[pos]['text'])
            batch_next.append(data[pos + 1]['text'])
            batch_labels.append(0)
        else:
            while True:
                pos_1 = random.randint(0, len(data))
                pos_2 = random.randint(0, len(data))
                batch_inputs.append(data[pos_1]['text'])
                batch_next.append(data[pos_2]['text'])
                batch_labels.append(1)
                if (pos_1 != pos_2) and (pos_1 != pos_2 - 1):
                    break
    return batch_inputs, batch_next, torch.tensor(batch_labels, dtype=torch.long)
            
def main(hparams):
    # Args
    config = bittensor.Config( hparams )
    batch_size = 128
    epoch_size = 100
    hidden_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset: 74 million sentence pulled from books.
    dataset = load_dataset('bookcorpus')

    # Build Synapse
    model_config = transformers.modeling_bert.BertConfig(hidden_size=hidden_size, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512, is_decoder=False)
    model = BertNSPSynapse(model_config)
    model.to(device)

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
    router = bittensor.Router(x_dim = hidden_size, key_dim = 100, topk = 10)
    
    # Optimizer.
    lr = 3.0 # learning rate
    params = list(router.parameters()) + list(model.parameters())
    optimizer = torch.optim.SGD(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    def train(dataset, model, epoch):
        model.train()  # Turn on the train mode.
        optimizer.zero_grad() # Zero out lingering gradients.

        step = 0
        while step < epoch_size:
            # Next batch.
            sentences, next_sentences, next_sentence_labels = nsp_batch(dataset['train'], batch_size)

            # Get routing context
            context = model.forward_text( sentences )
            
            # Query the remote network.
            # Flatten mnist inputs for routing.
            synapses = metagraph.get_synapses( 1000 ) # Returns a list of synapses on the network (max 1000).
            requests, _ = router.route( synapses, context, sentences ) # routes inputs to network.
            responses = dendrite.forward_text( synapses, requests ) # Makes network calls.
            network = router.join( responses ) # Joins responses based on scores..
            
            # Compute full pass and get loss.
            output = model.forward(sentences, next_sentences, next_sentence_labels, network)
            
            loss = output['local_target_loss']
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            logger.info('Train Step: {} [{}/{} ({:.0f}%)]\t Network Loss: {:.6f}\t Local Loss: {:.6f}\t Distilation Loss: {:.6f}'.format(
                 epoch, step, epoch_size, step/epoch_size, output['network_target_loss'].item(), output['local_target_loss'].item(), output['distillation_loss'].item()))
      
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