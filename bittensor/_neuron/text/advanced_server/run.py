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
""" Advanced server neuron.

Example:
    $ python miners/text/advanced_server/main.py

"""
import bittensor
import torch
import wandb
import datetime
import traceback
import sys
import os

from loguru import logger; logger = logger.opt(colors=True)
from torch.nn.utils import clip_grad_norm_
from datetime import datetime,timedelta
from threading import Lock
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def serve( config, server):
    config.to_defaults()

    # Create Subtensor connection
    subtensor = bittensor.subtensor(config = config)

    # Load/Create our bittensor wallet.
    wallet = bittensor.wallet( config = config ).create().register()

    # Load/Sync/Save our metagraph.
    metagraph = bittensor.metagraph ( 
        subtensor = subtensor
    ).load().sync().save()

    # Instantiate the model we are going to serve on the network.
    # Creating a threading lock for updates to the model
    mutex = Lock()
    gp_server = server
    
    # Create our optimizer.
    optimizer = torch.optim.SGD(
        [ {"params": gp_server.parameters()} ],
        lr = config.server.learning_rate,
        momentum = config.server.momentum,
    )
    
    timecheck = {}
    # Define our forward function.
    def forward_text ( inputs_x ):
        r""" Forward function that is called when the axon recieves a forward request from other peers
            Args:
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.

            Returns:
                outputs (:obj:`torch.FloatTensor`):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """ 
        return gp_server.encode_forward( inputs_x )

    # Define our backward function.
    def backward_text (inputs_x, grads_dy ):
        r"""Backwards function that is called when the axon recieves a backwards request from other peers.
            Updates the server parameters with gradients through the chain.

            Args:
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs from previous forward call.
                grads_dy ( :obj:`torch.Tensor`, `required`):
                    torch grads of forward output.
                    
        """
        # -- normalized grads -- 
        grads_dy = grads_dy/(grads_dy.sum() + 0.00001)
        
        with mutex:
            outputs_y = gp_server.encode_forward( inputs_x )
            with torch.autograd.set_detect_anomaly(True):
                torch.autograd.backward (
                    tensors = [ outputs_y ],
                    grad_tensors = [ grads_dy ],
                    retain_graph=True
                )
            logger.info('Backwards axon gradient applied')

        gp_server.backward_gradients += inputs_x.size(0)
       
    def priority(pubkey:str, request_type:bittensor.proto.RequestType, inputs_x) -> float:
        r"""Calculates the priority on requests based on stake and size of input

            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                request_type ( bittensor.proto.RequestType, `required`):
                    the request type ('FORWARD' or 'BACKWARD').
        """        
        uid = metagraph.hotkeys.index(pubkey)
        priority = metagraph.S[uid].item()/ sys.getsizeof(inputs_x)

        return priority

    def blacklist(pubkey:str, request_type:bittensor.proto.RequestType) -> bool:
        r"""Axon security blacklisting, used to blacklist message from low stake members
            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                request_type ( bittensor.proto.RequestType, `required`):
                    the request type ('FORWARD' or 'BACKWARD').
        """

        # Check for stake
        def stake_check():
            uid =metagraph.hotkeys.index(pubkey)
            if metagraph.S[uid].item() < config.server.blacklist.stake:
                return True
            else:
                return False

        # Check for time
        def time_check():
            current_time = datetime.now()
            if pubkey in timecheck.keys():
                prev_time = timecheck[pubkey]
                if current_time - prev_time >= timedelta(seconds=config.server.blacklist.time):
                    timecheck[pubkey] = current_time
                    return False
                else:
                    return True
            else:
                timecheck[pubkey] = current_time
                return False

        # Black list or not
        if stake_check() or time_check():
            return True
        else: 
            return False
            

    # Create our axon server
    axon = bittensor.axon (
                wallet = wallet,
                forward_text = forward_text,
                backward_text = backward_text,
                blacklist= blacklist,
                priority = priority
            ) 

    # Training Data
    dataset = bittensor.dataset(config=config)

    # load our old model
    if config.server.restart != True:
        gp_server.load(config.server.full_path)

    if config.wandb.api_key != 'default':
        # --- Init Wandb.
        bittensor.wandb(
            config = config,
            cold_pubkey = wallet.coldkeypub.ss58_address,
            hot_pubkey = wallet.hotkey.ss58_address,
            root_dir = config.server.full_path
        )

    # -- Main Training loop --
    try:
        # --  serve axon to the network.
        axon.start().serve(subtensor=subtensor)

        # --- creating our chain weights
        chain_weights =torch.zeros(metagraph.n)
        uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
        chain_weights[uid] = 1 

        while True:
            # --- Run 
            dataloader = iter(dataset.dataloader(epoch_length=100))
            current_block = subtensor.get_current_block()
            end_block = current_block + config.server.blocks_per_epoch
            interation = 0
            # --- Training step.
            while end_block >= current_block:
                if current_block != subtensor.get_current_block():
                    loss, _ = gp_server( next( dataloader ) )
                    if interation > 0 : 
                        losses += loss
                    else:
                        losses = loss
                    interation += 1
                    current_block = subtensor.get_current_block()
            
            #Custom learning rate
            if gp_server.backward_gradients > 0:
                optimizer.param_groups[0]['lr'] =  1/(gp_server.backward_gradients)
            else:
                optimizer.param_groups[0]['lr'] =  0.1
            gp_server.backward_gradients = 0

            # --- Update parameters
            if interation != 0:
                with mutex:
                    logger.info('Backpropagation Started')
                    losses.backward()
                    clip_grad_norm_(gp_server.parameters(), 1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    logger.info('Backpropagation Successful: Model updated')

            # --- logging data
            wandb_data = {
                'block': end_block,
                'loss': losses.item()/interation,
                'stake': metagraph.S[ uid ].item(),
                'rank': metagraph.R[ uid ].item(),
                'incentive': metagraph.I[ uid ].item(),
            } 

            # wandb syncing and update metagraph
            metagraph.sync().save()
            chain_weights =torch.zeros(metagraph.n)
            chain_weights[uid] = 1 

            if config.wandb.api_key != 'default':
                wandb.log( wandb_data )
            logger.info(wandb_data)

            # save the model
            gp_server.save(config.server.full_path)

            # --- setting weights
            try: 
                did_set = subtensor.timeout_set_weights(
                    timeout=10,
                    uids=metagraph.uids,
                    weights = chain_weights,
                    wait_for_inclusion = True,
                    wallet = wallet,
                )
                
                if did_set:
                    logger.success('Successfully set weights on the chain')
                else:
                    logger.error('Failed to set weights on chain. (Timeout)')
            except Exception as e:
                logger.error('Failure setting weights on chain with error: {}', e)

    except KeyboardInterrupt:
        # --- User ended session ----
        axon.stop()
    except Exception as e:
        # --- Unknown error ----
        logger.exception('Unknown exception: {} with traceback {}', e, traceback.format_exc())

