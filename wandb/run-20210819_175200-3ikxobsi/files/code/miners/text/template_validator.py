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
""" The Exodus base validator

Example:
    $ python miners/text/template_validator.py

"""
import argparse
import bittensor
import torch
import time
import wandb
import datetime
import torch.nn.functional as F
from qqdm import qqdm, format_str
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Validator Model.
class Validator( torch.nn.Module ):
    def __init__(self, config, device, dendrite, metagraph ):
        super(Validator, self).__init__()
        self.chain_weights = torch.nn.Parameter(torch.ones( [0], requires_grad=True))
        self.layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout )
        self.encoder = TransformerEncoder( self.layers, config.nucleus.nlayers ).to(device)
        self.decoder = torch.nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__ , bias=False).to(device)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.dendrite = dendrite
        self.metagraph = metagraph
        self.device = device

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):    
        r""" Add custom params to the parser.
        """
        parser.add_argument('--nucleus.nhid', type=int, help='the dimension of the feedforward network model in nn.TransformerEncoder', default=200)
        parser.add_argument('--nucleus.nhead', type=int, help='the number of heads in the multiheadattention models', default=2)
        parser.add_argument('--nucleus.nlayers', type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2)
        parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.2)
        parser.add_argument('--nucleus.topk', type=int, help='the number of peers queried during each remote forward call', default=20)
        parser.add_argument('--nucleus.punishment', type=float, help='The punishment on the chain weights that do not respond ', default=0.001 )

    def init_weights(self):
        initrange = 0.1
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward ( self, inputs ):
        # --- Forward pass.
        outputs = self.decoder ( self.encoder( self.remote( inputs ) ) )

        # --- Loss
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()     
        loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) )
        return outputs, loss

    def remote ( self, inputs ):
        # ---- Topk Weights ---- (TODO: check if the gaussians are enough disrupt the chain weights)
        real_topk = min( config.miner.topk, self.metagraph.n.item() ) 
        noise = torch.normal( 0, torch.std( self.chain_weights ).item()+0.0000001, size=( self.chain_weights.size())).to( self.device )
        topk_weights, topk_uids = torch.topk( self.chain_weights + noise, real_topk, dim=0 ) 

        # ---- Query network ----
        responses, return_ops = self.dendrite.forward_text ( 
            endpoints = self.metagraph.endpoints[ topk_uids ], 
            inputs = inputs
        )

        # ---- Join based on weights ----
        joining_uids= torch.where(return_ops==0)[0]
        joining_weights = F.softmax( topk_weights[(return_ops == 0)], dim = 0 )
        output = torch.zeros( (inputs.shape[0], inputs.shape[1], bittensor.__network_dim__)).to( self.device )
        for index, joining_weight in enumerate( joining_weights ): 
            output += responses[joining_uids[index]].to( self.device ) * joining_weight

        # ---- Punish peers with non-successful return ops ----
        with torch.no_grad():
            self.chain_weights[topk_uids[(return_ops != 0)]] -= config.nucleus.punishment
            self.chain_weights[ self.chain_weights < -1 ] = -1 #lower bound for chain weights 

        return output

def config ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--miner.topk', type=int, help='the number of peers queried during each remote forward call', default=20)
    parser.add_argument('--miner.learning_rate', type=float, help='Training initial learning rate.', default=1)
    parser.add_argument('--miner.momentum', type=float, help='optimizer momentum.', default=0.8)
    parser.add_argument('--miner.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
    bittensor.wallet.add_args( parser )
    bittensor.dendrite.add_args( parser )
    bittensor.subtensor.add_args( parser )
    bittensor.logging.add_args( parser )
    bittensor.dataloader.add_args( parser )
    Validator.add_args( parser )
    return bittensor.config( parser )

def main( config ):
    print (config)

    # Init bittensor logging.
    bittensor.logging( config = config )

    # Load/Create our bittensor wallet.
    wallet = bittensor.wallet( config = config ).create()
    if wallet.get_balance().rao > 0:
        wallet.add_stake()

    # Connect to the chain.
    subtensor = bittensor.subtensor( config = config )

    # Load/Sync/Save our metagraph.
    metagraph = bittensor.metagraph ( subtensor = subtensor ).load().sync().save()

    # Create Dendrite.
    dendrite = bittensor.dendrite( config = config )

    # Genesis dataset.
    dataset = bittensor.dataloader ( config = config )

    # Build Device.
    device = torch.device( device = config.miner.device)

    # Instantiate validator model.
    validator = Validator( config = config, dendrite = dendrite, metagraph = metagraph, device = device )
    validator.to( device )

    # Create our optimizer.
    optimizer = torch.optim.SGD(
        [ {"params": validator.parameters() } ],
        lr = config.miner.learning_rate,
        momentum = config.miner.momentum,
    )

    # --- Init Wandb.
    with wandb.init (
            config = config, 
            name = datetime.datetime.now().strftime("%Y-%m-%d:%H-%M"),
            project = wallet.coldkeypub[:20],
            group = wallet.hotkey.ss58_address[:20],
            save_code = True
        ):
        wandb.watch( validator, log = 'all', log_freq = 10 )

        # --- Run Forever.
        while True:

            # --- Run epoch.
            epoch_batches = dataset.dataloader( config.miner.epoch_length )
            progress_bar = qqdm(enumerate(epoch_batches), total=len(epoch_batches), desc=format_str('blue', f'Epoch Progress'))
            for _, (inputs) in progress_bar:
                _, loss = validator( inputs.to( device ) )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad() 

            # --- Checkpoint + Set mechanism weights.
            real_topk = min( config.miner.n_topk_chain_weights, metagraph.n.item() ) 
            topk_weights, topk_uids = torch.topk( validator.chain_weights, k = real_topk )
            normalized_topk_weights = torch.nn.functional.normalize( topk_weights - torch.min( topk_weights ), p = 1, dim = 0)
            did_set = subtensor.set_weights(
                uids = topk_uids,
                weights = normalized_topk_weights,
                wait_for_inclusion = True,
                wallet = wallet,
            )            
            metagraph.sync().save()

            # --- Log
            metagraph.sync().save()
            uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
            wand_data = {
                'Stake': metagraph.S[ uid ].item(),
                'Rank': metagraph.R[ uid ].item(),
                'Incentive': metagraph.I[ uid ].item(),
            } 
            for uid_j, val in enumerate(metagraph.W[uid,:].tolist()):
                wand_data[ 'w_\{{},{}\}'.format( uid, uid_j ) ] = val
            wandb.log( wand_data )
            time.sleep( 10 * bittensor.__blocktime__ )

if __name__ == "__main__":
    main( config() )