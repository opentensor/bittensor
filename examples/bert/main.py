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
from transformers import DataCollatorForLanguageModeling
from transformers import BertTokenizer

class BertMLMSynapse(bittensor.Synapse):
    """ An bittensor endpoint trained on wiki corpus.
    """
    def __init__(self, config):
        super(BertMLMSynapse, self).__init__(config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config = config
        
        # Bert config
        self._bert_config = transformers.modeling_bert.BertConfig(hidden_size=256, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512, is_decoder=False)
        
        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # The collator accepts a list [ dict{'input_ids, ...; } ] where the internal dict 
        # is produced by the tokenizer.
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
 
        # Bert embeddings
        self.embeddings = transformers.modeling_bert.BertEmbeddings(self._bert_config)

        # Bert encoder
        self._encoder = transformers.modeling_bert.BertEncoder(self._bert_config)

        # Bert dist encoder
        self.dist_encoder = transformers.modeling_bert.BertEncoder(self._bert_config)
        
        # joiner
        self.joiner = nn.Linear(self._bert_config.hidden_size, self._bert_config.hidden_size, bias=False)
        
        # Bert pooler
        self._pooler = transformers.modeling_bert.BertPooler(self._bert_config)

        # Bert Masked MLM Head.
        self._mlm_head = transformers.modeling_bert.BertOnlyMLMHead(self._bert_config)        
        
        # Loss for masked language modelling.
        self._mlm_loss = torch.nn.CrossEntropyLoss()
        
    def forward_text(self, x: List[str], net = None):
        
        # Response
        return_dict = {}
        
        # Tokenize the list of strings.
        x_tokenized = self.tokenizer(x)

        # Tokenizer returns a dict { 'input_ids': list[], 'attention': list[] }
        # but we need to convert to List [ dict ['input_ids': ..., 'attention': ... ]]
        # annoying hack
        x_tokenized = [dict(zip(x_tokenized,t)) for t in zip(*x_tokenized.values())]
        
        # Produces the masked language model inputs dictionary 
        # {'inputs': tensor_batch, 'labels': tensor_batch}
        # which can be used with the Bert Language model. 
        x_tokenized, x_masked = self.data_collator(x_tokenized)
        
        # Embeds each token into a uniform space.
        x_embed = self._embeddings(x_tokenized)
                  
        # Encodes sequence into a pooled representation. 
        x_dist = self._dist_encoder(x_embed)
        
        # y_dist
        y_dist = self._dist_pooler(x_dist)
        
        # x encoding.
        x_encod = self._encoder(x_embed, hidden_states = y_dist)
        
        # Produce masked language predictions.
        x_masked_prediction = self._masked_language_head(x_encod)
        
        # Calculate the masked prediction scores.
        loss = self.loss_mlm(x_masked_prediction.view(-1, self._bert_config.vocab_size), x_masked.view(-1))
        return_dict['loss_student'] = loss
        
        # Pools encoded inputs into network shape.
        y = self._pooler(x_encod)
        return_dict['y_student'] = y
        
        if net is not None:
            # Distilled network trained on dist output.
            loss_dist = F.mse_loss(y_dist, net) 
            return_dict['loss_distill'] = loss_dist

            x_encod_net = self._encoder(x_embed, hidden_states = net)
        
            # Produce masked language predictions.
            x_masked_prediction_net = self._masked_language_head(x_embed)
        
            # Calculate the masked prediction scores.
            loss_net = self.loss_mlm(x_masked_prediction.view(-1, self._bert_config.vocab_size), x_masked.view(-1))
            return_dict['loss_network'] = loss_net
        
            # Pools encoded inputs into network shape.
            y_net = self._pooler(x_encod)
            return_dict['y_network'] = y_net
            
        return return_dict
            
def main(hparams):
    
    # Args
    batch_size = 50
    eval_batch_size = 20
    log_interval = 10
    config = bittensor.Config( hparams )
    
    # Log/data/model paths.
    trial_id =  'cola-' + str(time.time()).split('.')[0]
    data_path = "~/data/CoLA/"
    log_dir = 'data/' + trial_id + '/logs/'
    model_path = 'data/' + trial_id + '/model.torch'

    # Build Synapse
    model = BertMLMSynapse(config)
    
    # Dataset
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')

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
    router = bittensor.Router(x_dim = batch_size, key_dim = 100, topk = 10)
    
    # Optimizer.
    criterion = nn.CrossEntropyLoss()  # loss function
    lr = 3.0 # learning rate
    params = list(router.parameters()) + list(model.parameters())
    optimizer = torch.optim.SGD(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    def train(dataset, transformer, epoch):
        model.train()  # Turn on the train mode
        total_loss = 0.0
        global_step = 0
        start_time = time.time()
        
        optimizer.zero_grad()

        # Here we produce a list of strings batch_size long
        batch_size = 10
        batch = dataset['train'][0: batch_size]['text']
        
        # encode the string inputs.
        context = model.forward_text( batch ) ['y_student']
        
        # Query the remote network.
        # Flatten mnist inputs for routing.
        synapses = metagraph.get_synapses( 1000 ) # Returns a list of synapses on the network (max 1000).
        requests, scores = router.route( synapses, context, batch ) # routes inputs to network.
        responses = dendrite.forward_image( synapses, requests ) # Makes network calls.
        net = router.join( responses ) # Joins responses based on scores..
        
        output = model.forward_text(batch, net)
        
        loss = output['loss_distill'] + output['loss_student'] + output['loss_network']
        loss.backward()
        optimizer.step()
        global_step += 1
          
        # Set network weights.
        weights = metagraph.getweights(synapses).to(model.device)
        weights = (0.99) * weights + 0.01 * torch.mean(scores, dim=0)
        metagraph.setweights(synapses, weights)
      
    global_step = 0
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