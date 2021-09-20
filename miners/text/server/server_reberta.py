#!/bin/python3
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
""" The Exodus base client.

Example:
    $ python miners/text/template_client.py

"""
import argparse
import bittensor
import torch
import time
import wandb
import datetime
from qqdm import qqdm
from transformers import RobertaConfig, RobertaModel
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
import os
import torch.nn.functional as F

class server(torch.nn.Module):
    def __init__(self, pretrained,pre_dimension,final_dim ):
        super(server, self).__init__()
        self.pretrained = pretrained
        self.final_dim =  final_dim
        self.pre_dimension = pre_dimension
        self.mapping = torch.nn.Linear( pre_dimension, final_dim)
        self.decoder = torch.nn.Linear( final_dim, bittensor.__vocab_size__ , bias=False)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        
    def forward(self, inputs,target):
        pre_hidden = self.pretrained(inputs).last_hidden_state
        down= F.interpolate(pre_hidden.unsqueeze(1),size=[20,768])
        #padding_l = (self.final_dim-self.pre_dimension)//2
        #padding_r = (self.final_dim-self.pre_dimension) - padding_l
        #encoded_hidden = F.pad(down.squeeze(1).detach(), (padding_l, padding_r),  "constant", 0)
        encoded_hidden = self.mapping(down.squeeze(1).detach())
        decoded_targets = self.decoder(encoded_hidden)
        
        shift_logits = decoded_targets[..., :-1, :].contiguous()
        shift_labels = target[..., 1:].contiguous()     
        loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) ) 
        return loss, decoded_targets
    
    def encode_forward(self,inputs):
        pre_hidden = self.pretrained(inputs).last_hidden_state
        padding_l = (self.final_dim-self.pre_dimension)//2
        padding_r = (self.final_dim-self.pre_dimension) - padding_l
        encoded_hidden = F.pad(pre_hidden, (padding_l, padding_r),  "constant", 0)
        return encoded_hidden


def remapping(input, old_token,new_token):
    new_data = []
    for i in range(input.shape[0]):
        decoded = old_token.decode(input[i]) 
        hugging = new_token(decoded)
        new_data += [torch.LongTensor(hugging.input_ids)]
    new_data = pad_sequence(new_data,batch_first=True)
    return new_data


def config ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--miner.learning_rate', type=float, help='Training initial learning rate.', default=0.05)
    parser.add_argument('--miner.momentum', type=float, help='optimizer momentum.', default=0.8)
    parser.add_argument('--miner.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0)
    parser.add_argument('--miner.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--miner.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='template_miner')
    bittensor.wallet.add_args( parser )
    bittensor.axon.add_args( parser )
    bittensor.subtensor.add_args( parser )
    bittensor.logging.add_args( parser )
    bittensor.wandb.add_args(parser)
    return bittensor.config( parser )

def main( config ):
    print (config)

    # Init bittensor logging.
    bittensor.logging( config = config )

    # Load/Create our bittensor wallet.
    wallet = bittensor.wallet( config = config ).create()


    # Instantiate the model we are going to serve on the network.
    # Miner training device.
    device = torch.device( device = config.miner.device)
    pretrained = RobertaModel.from_pretrained("roberta-base").to( device )
    hidd_dimension = pretrained.config.hidden_size

    gp_server = server(pretrained,hidd_dimension,bittensor.__network_dim__)
    # Create our optimizer.
    optimizer = torch.optim.SGD(
        [ {"params": gp_server.parameters()} ],
        lr = config.miner.learning_rate,
        momentum = config.miner.momentum,
    )
    scheduler= torch.optim.lr_scheduler.StepLR(optimizer,
        step_size= 1.0,
        gamma=0.95
    )
    # Define our forward function.
    def forward_text ( pubkey, inputs_x ):
        return gp_server.encode_forward( inputs_x.to(device) )

    # Define our backward function.
    def backward_text ( pubkey:str, inputs_x, grads_dy ):
        with torch.enable_grad():
            outputs_y = gp_server.encode_forward( inputs_x.to(device) )
            torch.autograd.backward (
                tensors = [ outputs_y.to(device) ],
                grad_tensors = [ grads_dy.to(device) ]
            )
            optimizer.step() # Applies accumulated gradients.
            optimizer.zero_grad() 

    # Training Data

    dataload = bittensor.dataloader()
    full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.miner.name ))
    # --- Init Wandb.
    bittensor.wandb(
        config = config,
        cold_pubkey = wallet.coldkeypub,
        hot_pubkey = wallet.hotkey.public_key,
        root_dir = full_path
    )

    # --- Run Forever.
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    for epoch in range(10000):
        print("epoch:",epoch)
        epoch_loss = 0
        epoch_batches = dataload.dataloader(epoch_length=100)
        for iteration, inputs in enumerate(epoch_batches):
            optimizer.zero_grad()
            new_data = remapping(inputs,bittensor.tokenizer(),tokenizer)
            loss, _ = gp_server( new_data,inputs )
            loss.backward()
            clip_grad_norm_(gp_server.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            if iteration % 10 == 1:
                print(loss.item())

        print("Epoch Loss:",epoch_loss/100)
        

if __name__ == "__main__":
    main( config() )