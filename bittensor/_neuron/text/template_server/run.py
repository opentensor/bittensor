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
import bittensor
import sys
import torch
import time
import wandb
import pandas
import datetime
from threading import Lock

def serve( config, server ):
    config.to_defaults()

    # Create Subtensor connection
    subtensor = bittensor.subtensor(config = config)

    # Load/Create our bittensor wallet.
    wallet = bittensor.wallet( config = config ).create().register()

    # Load/Sync/Save our metagraph.
    metagraph = bittensor.metagraph ( 
        subtensor = bittensor.subtensor( config = config )
    ).load().sync().save()

    # Instantiate the model we are going to serve on the network.
    # Miner training device.
    model = server

    # Create our optimizer.
    optimizer = torch.optim.SGD(
        [ {"params": model.parameters()} ],
        lr = config.neuron.learning_rate,
        momentum = config.neuron.momentum,
    )
    mutex = Lock()

    def forward_text ( inputs_x ):
        r""" Single threaded version of the Forward function that is called when the axon recieves a forward request from other peers
        """ 
        return model.encode_forward( inputs_x )


    def backward_text ( inputs_x, grads_dy ):
        r"""Single threaded backwards function that is called when the axon recieves a backwards request from other peers.
            Updates the server parameters with gradients through the chain.             
        """
        with mutex:
            with torch.enable_grad():
                with torch.autograd.set_detect_anomaly(True):
                    outputs_y = model.encode_forward( inputs_x )
                    torch.autograd.backward (
                        tensors = [ outputs_y ],
                        grad_tensors = [ grads_dy ]
                        )
                    optimizer.step()
                    optimizer.zero_grad()

    # Create our axon server and subscribe it to the network.
    axon = bittensor.axon (
        wallet = wallet,
        forward_text = forward_text,
        backward_text = backward_text,
    ).start().serve(subtensor=subtensor)

    if config.wandb.api_key != 'default':
        # --- Init Wandb.
        bittensor.wandb(
            config = config,
            cold_pubkey = wallet.coldkeypub.ss58_address,
            hot_pubkey = wallet.hotkey.ss58_address,
            root_dir = config.neuron.full_path
        )

    # --- Run Forever.
    while True:
        
        current_block = subtensor.get_current_block()
        end_block = current_block + config.neuron.blocks_per_epoch
        while end_block >= current_block:
            time.sleep( bittensor.__blocktime__ )
            current_block = subtensor.get_current_block()

        metagraph.sync().save()
        my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
       
        if config.wandb.api_key != 'default':
            wandb_data = {
                'stake': metagraph.S[ my_uid ].item(),
                'rank': metagraph.R[ my_uid ].item(),
                'trust': metagraph.I[ my_uid ].item(),
                'consensus': metagraph.C[ my_uid ].item(),
                'incentive': metagraph.I[ my_uid ].item(),
                'emission': metagraph.E[ my_uid ].item(),
            } 
            df = pandas.concat( [
                bittensor.utils.indexed_values_to_dataframe( prefix = 'w_i_{}'.format(nn.uid), index = metagraph.uids, values = metagraph.W[:, uid] ),
                axon.to_dataframe( metagraph = metagraph ),
            ], axis = 1)
            df['uid'] = df.index
            wandb_info_axon = axon.to_wandb()                
            wandb.log( { **wandb_data, **wandb_info_axon }, step = current_block )
            wandb.log( { 'stats': wandb.Table( dataframe = df ) }, step = current_block )

            for uid_i, val in enumerate(metagraph.W[:,my_uid].tolist()):
                if uid_i > 0:
                    wandb_data[ '{}/w_{}_{}'.format(uid_i, uid_i, my_uid) ] = val
            axon_wandb = axon.to_wandb( metagraph )
            wandb.log( { **wandb_data, **axon_wandb } )

        