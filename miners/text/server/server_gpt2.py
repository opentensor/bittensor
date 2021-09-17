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
from transformers import GPT2Model
from torch.nn.utils import clip_grad_norm_
import os
import torch.nn.functional as F

class server(torch.nn.Module):
    def __init__(self, pretrained,pre_dimension,final_dim ):
        super(server, self).__init__()
        self.pretrained = pretrained
        self.final_dim =  final_dim
        self.pre_dimension = pre_dimension
        #self.mapping = torch.nn.Linear( pre_dimension, final_dim)
        self.decoder = torch.nn.Linear( final_dim, bittensor.__vocab_size__ , bias=False)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        
    def forward(self, inputs):
        pre_hidden = self.pretrained(inputs).last_hidden_state
        padding_l = (self.final_dim-self.pre_dimension)//2
        padding_r = (self.final_dim-self.pre_dimension) - padding_l
        encoded_hidden = F.pad(pre_hidden, (padding_l, padding_r),  "constant", 0)
        decoded_targets = self.decoder(encoded_hidden)
        
        shift_logits = decoded_targets[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()     
        loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) ) 
        return loss, decoded_targets
    
    def encode_forward(self,inputs):
        pre_hidden = self.pretrained(inputs).last_hidden_state
        padding_l = (self.final_dim-self.pre_dimension)//2
        padding_r = (self.final_dim-self.pre_dimension) - padding_l
        encoded_hidden = F.pad(pre_hidden, (padding_l, padding_r),  "constant", 0)
        return encoded_hidden


def config ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--miner.learning_rate', type=float, help='Training initial learning rate.', default=0.1)
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

    # Load/Sync/Save our metagraph.
    metagraph = bittensor.metagraph ( 
        subtensor = bittensor.subtensor( config = config )
    ).load().sync().save()

    # Instantiate the model we are going to serve on the network.
    # Miner training device.
    device = torch.device( device = config.miner.device)
    pretrained = GPT2Model.from_pretrained("gpt2").to( device )
    hidd_dimension = pretrained.config.n_embd

    gp_server = server(pretrained,hidd_dimension,bittensor.__network_dim__)
    # Create our optimizer.
    optimizer = torch.optim.SGD(
        [ {"params": gp_server.parameters()} ],
        lr = config.miner.learning_rate,
        momentum = config.miner.momentum,
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

    # Create our axon server and subscribe it to the network.
    axon = bittensor.axon (
        wallet = wallet,
        forward_text = forward_text,
        backward_text = backward_text,
    ).start().subscribe()

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
    for epoch in range(100):
        print("epoch:",epoch)
        epoch_loss = 0
        epoch_batches = dataload.dataloader(epoch_length=100)
        for iteration, inputs in enumerate(epoch_batches):
            optimizer.zero_grad()
            loss, _ = gp_server( inputs )
            loss.backward()
            clip_grad_norm_(gp_server.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            if iteration % 10 == 1:
                print("iteration:",loss.item())

        print("Epoch Loss:",epoch_loss/100)
        uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
        wandb_data = {
            'Epoch': epoch,
            'loss': epoch_loss/100,
            'stake': metagraph.S[ uid ].item(),
            'rank': metagraph.R[ uid ].item(),
            'incentive': metagraph.I[ uid ].item(),
            'axon QPS': axon.stats.qps.value
        } 
        wandb.log( wandb_data )

if __name__ == "__main__":
    main( config() )