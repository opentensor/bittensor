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
from loguru import logger; logger = logger.opt(colors=True)

def serve( config, model ):
    config.to_defaults()

    # Create Subtensor connection
    subtensor = bittensor.subtensor(config = config)

    # Load/Create our bittensor wallet.
    wallet = bittensor.wallet( config = config ).create().register()

    # Load/Sync/Save our metagraph.
    metagraph = bittensor.metagraph ( 
        subtensor = bittensor.subtensor( config = config )
    ).load().sync().save()

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
        if config.neuron.training:
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

    last_set_block = subtensor.get_current_block()


    # --- Run Forever.
    while True:
        
        current_block = subtensor.get_current_block()
        end_block = current_block + config.neuron.blocks_per_epoch
        while end_block >= current_block:
            time.sleep( bittensor.__blocktime__ )
            current_block = subtensor.get_current_block()

        nn = subtensor.neuron_for_pubkey(wallet.hotkey.ss58_address)
        uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
        wandb_data = {
            'stake': nn.stake,
            'rank': nn.rank,
            'trust': nn.trust,
            'consensus': nn.consensus,
            'incentive': nn.incentive,
            'emission': nn.emission,
        }
        bittensor.__console__.print('[green]Current Status:[/green]', wandb_data)
        if config.wandb.api_key != 'default':

            df = pandas.concat( [
                bittensor.utils.indexed_values_to_dataframe( prefix = 'w_i_{}'.format(nn.uid), index = metagraph.uids, values = metagraph.W[:, uid] ),
                axon.to_dataframe( metagraph = metagraph ),
            ], axis = 1)
            df['uid'] = df.index
            wandb_info_axon = axon.to_wandb()                
            wandb.log( { **wandb_data, **wandb_info_axon }, step = current_block )
            wandb.log( { 'stats': wandb.Table( dataframe = df ) }, step = current_block )

        if current_block - last_set_block > config.neuron.blocks_per_set_weights:
            try: 
                last_set_block = current_block
                # Set self weights to maintain activity.
                chain_weights = torch.zeros(metagraph.n)
                chain_weights [ uid ] = 1 
                did_set = subtensor.set_weights(
                    uids=metagraph.uids,
                    weights = chain_weights,
                    wait_for_inclusion = False,
                    wallet = wallet,
                )
                
                if did_set:
                    logger.success('Successfully set weights on the chain')
                else:
                    logger.error('Failed to set weights on chain. (Timeout)')
            except Exception as e:
                logger.error('Failure setting weights on chain with error: {}', e)
