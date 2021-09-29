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
import sys
import torch
import time
import wandb
import datetime
from transformers import BertModel, BertConfig

def config ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--miner.learning_rate', type=float, help='Training initial learning rate.', default=1)
    parser.add_argument('--miner.momentum', type=float, help='optimizer momentum.', default=0.8)
    parser.add_argument('--miner.timeout', type=int, help='Number of seconds to wait for axon request', default=2)
    parser.add_argument('--miner.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0)
    parser.add_argument('--miner.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
    bittensor.wallet.add_args( parser )
    bittensor.axon.add_args( parser )
    bittensor.subtensor.add_args( parser )
    bittensor.logging.add_args( parser )
    bittensor.prioritythreadpool.add_args( parser )
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
    model = BertModel( 
        BertConfig (
            vocab_size = bittensor.__vocab_size__,
            hidden_size = bittensor.__network_dim__,
            num_hidden_layers = 8,
            num_attention_heads = 8,
            intermediate_size = 3072,
            hidden_act = "gelu",
            hidden_dropout_prob = 0.1,
            attention_probs_dropout_prob = 0.1,
            max_position_embeddings = 512,
            type_vocab_size = 2,
            initializer_range = 0.02,
            layer_norm_eps = 1e-12,
            pad_token_id = 0,
            gradient_checkpointing = False,
            position_embedding_type = "absolute",
            use_cache = True,
        )
    ).to( device )

    # Create our optimizer.
    optimizer = torch.optim.SGD(
        [ {"params": model.parameters()} ],
        lr = config.miner.learning_rate,
        momentum = config.miner.momentum,
    )

    # Priority thread pool 
    thread_pool = bittensor.prioritythreadpool(
        config = config
    )

    # Define our forward function.
    def forward_text ( pubkey, inputs_x, modality ):
        def run_call( inputs_x ):
            return model( inputs_x ).last_hidden_state
        # Make call based on stake priority.
        priority = metagraph.S[ metagraph.hotkeys.index(pubkey) ] / sys.getsizeof(inputs_x)
        future = thread_pool.submit( run_call, inputs_x = inputs_x.to( device ), priority = priority )
        return future.result(timeout = config.miner.timeout )


    # Define our backward function.
    def backward_text ( pubkey:str, inputs_x, grads_dy, modality ):
        def run_call( inputs_x, grads_dy ):
            with torch.enable_grad():
                outputs_y = model( inputs_x ).last_hidden_state
                torch.autograd.backward (
                    tensors = [ outputs_y ],
                    grad_tensors = [ grads_dy ]
                )
                optimizer.step() # Applies accumulated gradients.
                optimizer.zero_grad() 
        # Make call based on stake priority.
        priority = metagraph.S[ metagraph.hotkeys.index(pubkey) ] / sys.getsizeof(inputs_x)
        thread_pool.submit( run_call, input=inputs_x.to( device ), grad=grads_dy.to( device ), priority = priority )

    # Create our axon server and subscribe it to the network.
    axon = bittensor.axon (
        wallet = wallet,
        forward_text = forward_text,
        backward_text = backward_text,
    ).start().subscribe()

    # --- Init Wandb.
    with wandb.init (
            config = config, 
            name = datetime.datetime.now().strftime("%Y-%m-%d:%H-%M"),
            project = wallet.coldkeypub[:8],
            group = wallet.hotkey.ss58_address[:8],
            save_code = True
        ):

        # --- Run Forever.
        while True:
            metagraph.sync().save()
            uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
            wandb_data = {
                'stake': metagraph.S[ uid ].item(),
                'rank': metagraph.R[ uid ].item(),
                'incentive': metagraph.I[ uid ].item(),
                'axon QPS': axon.stats.qps.value
            } 
            for uid_i, val in enumerate(metagraph.W[:,uid].tolist()):
                wandb_data[ 'w_{},{}'.format(uid_i, uid) ] = val
            wandb.log( wandb_data )
            time.sleep( 10 * bittensor.__blocktime__ )

if __name__ == "__main__":
    main( config() )