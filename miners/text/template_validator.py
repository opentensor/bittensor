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
    $ python miners/text/template_validator.py --logging.debug

"""
import argparse
import yaml
from types import SimpleNamespace
import bittensor
import math
import torch
import wandb
import datetime
import os
from termcolor import colored
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from qqdm import qqdm, format_str
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from loguru import logger; logger = logger.opt(colors=True)

def config ():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--config', type=str, help='If set, defaults are overridden by passed file.')
    parser.add_argument('--miner.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='template_miner')
    parser.add_argument('--miner.resume', action='store_true', help='resume previous trial.', default=False)
    parser.add_argument('--miner.topk', type=int, help='the number of peers queried during each remote forward call', default=20)
    parser.add_argument('--miner.learning_rate', type=float, help='Training initial learning rate.', default=1)
    parser.add_argument('--miner.learning_rate_chain', type=float, help='Training initial learning rate.', default=1)
    parser.add_argument('--miner.momentum', type=float, help='optimizer momentum.', default=0.8)
    parser.add_argument('--miner.blocks_per_epoch', type=int, help='Blocks per epoch', default=30)
    parser.add_argument('--miner.n_topk_chain_weights', type=int, help='Maximum number of weights to submit to chain', default=100 )
    parser.add_argument('--miner.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--miner.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0)
    parser.add_argument('--nucleus.topk', type=int, help='the number of peers queried during each remote forward call', default=20)
    parser.add_argument('--nucleus.noise_multiplier', type=float, help='Noise standard deviation multiplier. Increases query exploration.', default=1.0)
    parser.add_argument('--nucleus.nhid', type=int, help='the dimension of the feedforward network model in nn.TransformerEncoder', default=200)
    parser.add_argument('--nucleus.nhead', type=int, help='the number of heads in the multiheadattention models', default=2)
    parser.add_argument('--nucleus.nlayers', type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2)
    parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.2)
    parser.add_argument('--nucleus.punishment', type=float, help='the punishment for those not responding', default=0)
    parser.add_argument('--wandb.project', type=str, help='''Optionally pass wandb project name for use_wandb''', default='default')
    parser.add_argument('--wandb.run_group', type = str, help='''Optionally pass wandb group name for use_wandb''', default='default')
    
    bittensor.wallet.add_args( parser )
    bittensor.dendrite.add_args( parser )
    bittensor.subtensor.add_args( parser )
    bittensor.logging.add_args( parser )
    bittensor.dataset.add_args( parser )
    return bittensor.config( parser )

def main( config ):

    print (config)
    
    save_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.miner.name ))

    # Init bittensor logging.
    bittensor.logging ( config = config )

    # Load/Create our bittensor wallet.
    wallet = bittensor.wallet ( config = config ).create_if_non_existent()

    # Connect to the chain.
    subtensor = bittensor.subtensor ( config = config )

    # Subscribe validator.
    subtensor.subscribe (
        wallet = wallet,
        ip = bittensor.external_ip(),
        port = 8080,
        modality = 0,
        wait_for_inclusion = True,
        wait_for_finalization = False 
    )

    # Load/Sync/Save our metagraph.
    metagraph = bittensor.metagraph ( subtensor = subtensor ).load().sync().save()
    
    uid = metagraph.hotkeys.index ( wallet.hotkey.ss58_address )

    # Create Dendrite.
    dendrite = bittensor.dendrite ( config = config )

    # Load genesis dataset.
    dataset = bittensor.dataset ( config = config )

    # Build Device.
    device = torch.device ( device = config.miner.device )

    # Instantiate validator model.
    class Validator( torch.nn.Module ):

        def __init__(self, config ):
            super(Validator, self).__init__()
            self.layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout )
            self.encoder = TransformerEncoder( self.layers, config.nucleus.nlayers )
            self.decoder = torch.nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__ , bias=False)
            self.loss_fct = torch.nn.CrossEntropyLoss()
            self.chain_weights = torch.nn.Parameter(torch.ones( [ metagraph.n.item() ] , requires_grad=True))
            self.noise_offset = 0.0000001

        def forward ( self, inputs ):
            # Apply model.
            remote_hidden = self.remote( inputs.to( device ) )
            encoded_hidden = self.encoder( remote_hidden )
            decoded_targets = self.decoder ( encoded_hidden )

            # Compute loss.
            shift_logits = decoded_targets[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()     
            self.loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) )
            return self.loss, decoded_targets

        def remote ( self, inputs ):

            # ---- Get active peers and their weights ---- 
            active_uids = torch.where(metagraph.active > 0)[0]
            active_chain_weights = self.chain_weights[active_uids]

            # ---- Topk Weights ---- (TODO: check if the gaussians are enough disrupt the chain weights)
            real_topk = min( config.nucleus.topk, metagraph.n.item(), len(active_uids))
            noise = torch.normal( 0, torch.std(active_chain_weights).item()+self.noise_offset, size=( active_chain_weights.size())).to( config.miner.device )
            topk_weights, topk_idx = torch.topk(active_chain_weights + noise , real_topk, dim=0)
            topk_uids = active_uids[topk_idx]

            # ---- Query network ----
            responses, return_ops, query_times = dendrite.forward_text ( 
                endpoints = metagraph.endpoints[ topk_uids ], 
                inputs = inputs
            )

            # ---- Join based on weights ----
            joining_uids = torch.where(return_ops== bittensor.proto.ReturnCode.Success)[0]
            joining_weights = F.softmax( topk_weights[(return_ops == bittensor.proto.ReturnCode.Success)], dim = 0 )
            output = torch.zeros( (inputs.shape[0], inputs.shape[1], bittensor.__network_dim__)).to( device )
            for index, joining_weight in enumerate( joining_weights ): 
                output += responses[joining_uids[index]].to( device ) * joining_weight

            # ---- Punish peers with non-successful return ops ----
            with torch.no_grad():
                self.chain_weights[topk_uids[(return_ops != bittensor.proto.ReturnCode.Success)]] -= config.nucleus.punishment
                self.chain_weights[ self.chain_weights < -1 ] = -1 # lower bound for chain weights 

            return output

    # Create validator model.
    validator = Validator( config = config ).to( device )
    
    optimizer = torch.optim.SGD(
        [ {'params': validator.chain_weights, 'lr': config.miner.learning_rate_chain} ],
        lr = config.miner.learning_rate,
        momentum = config.miner.momentum,
    )

    # Create wandb for telemetry.
    wandb.init (
        config = config, 
        name = datetime.datetime.now().strftime("%Y-%m-%d:%H-%M"),
        project = wallet.coldkeypub.ss58_address[:8] if not config.wandb.project else config.wandb.project,
        group = wallet.hotkey.ss58_address[:8] if not config.wandb.run_group else config.wandb.run_group,
        dir = save_path,
        resume = config.miner.resume,
        save_code = True
    )
    wandb.watch( validator, log = 'all', log_freq = 10 )

    # Optionally resume.
    if config.miner.resume:
        try:
            validator.load_state_dict( torch.load("{}/validator.torch".format( save_path ))['validator'], strict=False )
        except Exception as e:
            logger.error('Error reloading model: {} '.format(e))
    torch.save( { 'validator': validator.state_dict() }, "{}/validator.torch".format( save_path ))

    # --- Run Forever.
    epoch = 0
    global_step = 0
    best_loss = math.inf

    while True:
    
        # --- Sync + reshape.      
        metagraph.sync().save()
        chain_growth = metagraph.n.item() - torch.numel( validator.chain_weights )
        validator.chain_weights = torch.nn.Parameter(torch.cat( [validator.chain_weights, torch.ones([chain_growth], dtype=torch.float32, requires_grad=True)])).to(device)

        # --- Run epoch.
        start_block = subtensor.get_current_block() + 1
        end_block = start_block + config.miner.blocks_per_epoch
        blocks = [ block for block in range(start_block, end_block) ]
        progress = qqdm( blocks, total=len(blocks), desc=format_str('white', f'Epoch'))

        # --- Reset the epoch logs
        total_epoch_loss = 0
        batch_count = 0
        
        for block in progress:
            
            # --- Training step.
            while block >= subtensor.get_current_block():
                loss, _ = validator( next( dataset ) )
                loss.backward()
                clip_grad_norm_(validator.parameters(), config.miner.clip_gradients)
                optimizer.step()
                optimizer.zero_grad() 
                global_step += 1
                batch_count += 1
                total_epoch_loss += loss.item()

            # Take topk chain weights.
            real_topk = min( config.miner.n_topk_chain_weights, metagraph.n.item() ) 
            topk_weights, topk_uids = torch.topk( F.softmax( validator.chain_weights ), k = real_topk )
            final_weights = torch.nn.functional.normalize( topk_weights - torch.min( topk_weights ), p = 1, dim = 0)

            # --- Step logs.
            info = { 
                'epoch': epoch,
                'global_step': global_step,
                'start': start_block,
                'current': block,
                'end': start_block + config.miner.blocks_per_epoch,
                'loss': colored('{:.4f}'.format(loss.item()), 'green'), 
                'best': colored('{:.4f}'.format(best_loss), 'green'), 
                'stake': colored('{:.4f}'.format(metagraph.S[ uid ].item()), 'green'),
                'dividends': colored('{:.4f}'.format(metagraph.S[ uid ].item()), 'green')
            }
            
            for weight, uid_j in list(zip(final_weights.tolist(), topk_uids.tolist())):
                if (validator.chain_weights.grad != None) and (validator.chain_weights.grad[ uid_j ] < 0):
                    color = 'green'
                else:
                    color = 'red'
                if weight > 0.001: 
                    info[ str(uid_j) ] = colored('{:.4f}'.format( weight ), color)

            progress.set_infos( info )
        
        # --- End of epoch
        # --- Set mechanism weights.
        subtensor.set_weights (
            uids = topk_uids,
            weights = final_weights,
            wait_for_inclusion = False,
            wallet = wallet,
        )    

        # --- Log.
        metagraph.sync().save()
        epoch_loss = total_epoch_loss / batch_count
        
        wandb_data = {
            'stake': metagraph.S[ uid ].item(),
            'dividends': metagraph.D[ uid ].item(),
            'epoch_loss': epoch_loss
        } 

        norm_weights = F.softmax( validator.chain_weights.detach() )
        
        for uid_j in topk_uids.tolist():
            uid_str = str(uid_j).zfill(3)
            wandb_data[ f'peer_norm_weight uid:{uid_str}' ] = norm_weights[uid_j]
            wandb_data[ f'peer_wo_norm_weight uid:{uid_str}' ] = validator.chain_weights[uid_j]
        
        wandb_data_dend = dendrite.to_wandb()
        wandb.log( {**wandb_data, **wandb_data_dend} )
        
        # --- Save.
        if best_loss > epoch_loss : 
            best_loss = epoch_loss
            torch.save( { 'validator': validator.state_dict() }, "{}/validator.torch".format( save_path ))
        epoch += 1


if __name__ == "__main__":
    main( config() )