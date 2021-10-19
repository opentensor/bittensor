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
    $ python miners/text/server/template_client.py

"""
import argparse
import traceback
from bittensor._metagraph.metagraph_impl import Metagraph
from logging import Logger, raiseExceptions
from loguru import logger; logger = logger.opt(colors=True)
import bittensor
import torch
import time
import wandb
import datetime
from qqdm import qqdm
from transformers import AutoModel,AutoTokenizer,AutoConfig
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import concurrent
from threading import Thread, Lock
from nuclei.server import server
import sys
import os
import torch.nn.functional as F
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def main( config ):

    # Create Subtensor connection
    subtensor = bittensor.subtensor(config = config)

    # Load/Create our bittensor wallet.
    wallet = bittensor.wallet( config = config ).create()

    # Load/Sync/Save our metagraph.
    metagraph = bittensor.metagraph ( 
        subtensor = subtensor
    ).load().sync().save()

    # Instantiate the model we are going to serve on the network.
    # Creating a threading lock for updates to the model
    mutex = Lock()
    gp_server = server(config=config)
    
    # Create our optimizer.
    optimizer = torch.optim.SGD(
        [ {"params": gp_server.parameters()} ],
        lr = config.server.learning_rate,
        momentum = config.server.momentum,
    )
    threadpool = bittensor.prioritythreadpool(config=config)

    # Define our forward function.
    def forward_text (pubkey, inputs_x ):
        r""" Forward function that is called when the axon recieves a forward request from other peers
            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.

            Returns:
                outputs (:obj:`torch.FloatTensor`):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """ 
        def call(inputs):
            return gp_server.encode_forward( inputs )
        uid = metagraph.hotkeys.index(pubkey)
        priority = metagraph.S[uid].item()/ sys.getsizeof(inputs_x)
        future = threadpool.submit(call,inputs=inputs_x.to(gp_server.device),priority=priority)
        try:
            return future.result(timeout= gp_server.config.server.forward_timeout)
        except concurrent.futures.TimeoutError :
            raise TimeoutError('TimeOutError')
        except Exception as e:
            logger.error('Error found: {}, with message {}'.format(repr(e), e))

    # Define our backward function.
    def backward_text (pubkey:str, inputs_x, grads_dy ):
        r"""Backwards function that is called when the axon recieves a backwards request from other peers.
            Updates the server parameters with gradients through the chain.

            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs from previous forward call.
                grads_dy ( :obj:`torch.Tensor`, `required`):
                    torch grads of forward output.
                    
        """
        def call(input,grad,mutex):
            with mutex:
                outputs_y = gp_server.encode_forward( input )
                if gp_server.outputs_cache == None:
                    gp_server.outputs_cache = outputs_y
                    gp_server.gradients_cache = grad
                else:
                    gp_server.outputs_cache = torch.cat((gp_server.outputs_cache, outputs_y),0)
                    gp_server.gradients_cache = torch.cat((gp_server.gradients_cache, grad),0)

                if gp_server.outputs_cache.size(0) >= 30:
                    with torch.autograd.set_detect_anomaly(True):
                        torch.autograd.backward (
                            tensors = [ gp_server.outputs_cache ],
                            grad_tensors = [ gp_server.gradients_cache ],
                            retain_graph=True
                        )
                    gp_server.outputs_cache = None
                    gp_server.gradients_cache = None  
                    logger.info('Backwards axon gradient applied')
                    
        uid = metagraph.hotkeys.index(pubkey)
        priority = metagraph.S[uid].item()/ sys.getsizeof(inputs_x)
        

        try:
            future = threadpool.submit(call, input=inputs_x.to( gp_server.device ), grad=grads_dy.to( gp_server.device ),mutex=mutex, priority=priority)
        except concurrent.futures.TimeoutError :
            raise TimeoutError('TimeOutError')
        except Exception as e:
            logger.error('Error found: {}, with message {}'.format(repr(e), e))

    def blacklist(pubkey:str) -> bool:
        r"""Axon security blacklisting, used to blacklist message from low stake members
        Currently, this is not turned on.
        """
        uid =metagraph.hotkeys.index(pubkey)
        if metagraph.S[uid].item() < config.server.blacklist:
            return True
        else:
            return False

    # Create our axon server
    axon = bittensor.axon (
                wallet = wallet,
                forward_text = forward_text,
                backward_text = backward_text,
                blacklist= blacklist,
            ) 

    # Training Data
    dataset = bittensor.dataset(config=config)
    full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.server.name ))
    bittensor.logging( config = config,logging_dir = full_path)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    # load our old model
    if config.server.restart != True:
        gp_server.load(full_path)

    # --- Init Wandb.
    bittensor.wandb(
        config = config,
        cold_pubkey = wallet.coldkeypub.ss58_address,
        hot_pubkey = wallet.hotkey.ss58_address,
        root_dir = full_path
    )

    # --- creating our chain weights
    chain_weights =torch.zeros(metagraph.n)
    uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
    chain_weights[uid] = 1 

    # -- Main Training loop --
    try:
        # --  subscribe axon to the network.
        axon.start().subscribe()

        while True:
            # --- Run 
            dataset = iter(dataset.dataloader(epoch_length=config.server.blocks_per_epoch))
            current_block = subtensor.get_current_block()
            end_block = current_block + 10
            interation = 0
            # --- Training step.
            while end_block >= current_block:
                if current_block != subtensor.get_current_block():
                    loss, _ = gp_server( next( dataset ) )
                    if interation > 0 : 
                        losses += loss
                    else:
                        losses = loss
                    interation += 1
                    current_block = subtensor.get_current_block()

            # --- Update parameters
            if interation != 0:
                with mutex:
                    logger.info('Backpropagation Started')
                    if gp_server.outputs_cache != None:
                        torch.autograd.backward (
                            tensors = [ gp_server.outputs_cache ],
                            grad_tensors = [ gp_server.gradients_cache ],
                            retain_graph=True
                        )

                    gp_server.outputs_cache = None
                    gp_server.gradients_cache = None

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
            wandb.log( wandb_data )
            logger.info(wandb_data)

            # save the model
            gp_server.save(full_path)

            # --- setting weights
            try: 
                did_set = subtensor.timeout_set_weights(
                    timeout=10,
                    uids=metagraph.uids,
                    weights = chain_weights,
                    wait_for_inclusion = True,
                    wallet = wallet,
                )
            except Exception as e:
                logger.error('Failure setting weights on chain with error: {}', e)

    except KeyboardInterrupt:
        # --- User ended session ----
        axon.stop()
    except Exception as e:
        # --- Unknown error ----
        logger.exception('Unknown exception: {} with traceback {}', e, traceback.format_exc())

if __name__ == "__main__":
    main( server.config() )