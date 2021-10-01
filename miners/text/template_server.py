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
import datetime
from nuclei.server import server

def main( config ):
    # Create Subtensor connection
    subtensor = bittensor.subtensor(config = config)

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
    model = server(config=config,model_name='bert-base-uncased',pretrained=True)


    # Create our optimizer.
    optimizer = torch.optim.SGD(
        [ {"params": model.parameters()} ],
        lr = config.server.learning_rate,
        momentum = config.server.momentum,
    )

    def forward_text (pubkey, inputs_x ):
        r""" Single threaded version of the Forward function that is called when the axon recieves a forward request from other peers
        """ 
        return model.encode_forward( inputs_x )


    def backward_text ( pubkey:str, inputs_x, grads_dy ):
        r"""Single threaded backwards function that is called when the axon recieves a backwards request from other peers.
            Updates the server parameters with gradients through the chain.             
        """
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
    ).start().subscribe()

    # --- Init Wandb.
    with wandb.init (
            config = config, 
            name = datetime.datetime.now().strftime("%Y-%m-%d:%H-%M"),
            project = wallet.coldkeypub.ss58_address[:8],
            group = wallet.hotkey.ss58_address[:8],
        ):

        # --- Run Forever.
        while True:
            end_block = subtensor.get_current_block() + config.server.blocks_per_epoch
            while end_block >= subtensor.get_current_block():
                time.sleep( bittensor.__blocktime__ )
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
            

if __name__ == "__main__":
    main( server.config() )