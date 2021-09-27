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
from transformers import BertModel, BertConfig
from server.pretrained_template import server

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
    model = server(config=config,model_name='bert-base-uncased',pretrained=False)


    # Create our optimizer.
    optimizer = torch.optim.SGD(
        [ {"params": model.parameters()} ],
        lr = config.server.learning_rate,
        momentum = config.server.momentum,
    )

    # Create our axon server and subscribe it to the network.
    model.start(
        wallet = wallet,
        optimizer= optimizer,
        metagraph=metagraph,
        single_thread = True
    )

    # --- Init Wandb.
    with wandb.init (
            config = config, 
            name = datetime.datetime.now().strftime("%Y-%m-%d:%H-%M"),
            project = wallet.coldkeypub[:8],
            group = wallet.hotkey.ss58_address[:8],
        ):

        # --- Run Forever.
        while True:
            metagraph.sync().save()
            uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
            wandb_data = {
                'stake': metagraph.S[ uid ].item(),
                'rank': metagraph.R[ uid ].item(),
                'incentive': metagraph.I[ uid ].item(),
                'axon QPS': model.axon.stats.qps.value
            } 
            for uid_i, val in enumerate(metagraph.W[:,uid].tolist()):
                wandb_data[ 'w_{},{}'.format(uid_i, uid) ] = val
            wandb.log( wandb_data )
            time.sleep( 10 * bittensor.__blocktime__ )

if __name__ == "__main__":
    main( server.config() )