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
import bittensor
import math
import torch
import wandb
import pandas
from termcolor import colored
from functools import partial

from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from qqdm import qqdm, format_str
from loguru import logger; logger = logger.opt(colors=True)
from ..neuron_utilities import update_metagraph_peerweight

def run( config , validator, subtensor, wallet, metagraph, dataset, device, uid, dendrite):
    print(config)
    config.to_defaults()
    validator = validator.to(device)

    if config.wandb.api_key != 'default':
        # Create wandb for telemetry.
        bittensor.wandb(
            config = config,
            cold_pubkey = wallet.coldkeypub.ss58_address,
            hot_pubkey = wallet.hotkey.ss58_address,
            root_dir = config.neuron.full_path
        )

    # Optionally resume.
    if config.neuron.no_restart != True:
        try:
            validator.load_state_dict( torch.load("{}/validator.torch".format( config.neuron.full_path ))['validator'], strict=False )
        except Exception as e:
            logger.error('Error reloading model: {} '.format(e))
    torch.save( { 'validator': validator.state_dict() }, "{}/validator.torch".format( config.neuron.full_path ))

    # --- Run Forever.
    epoch = 0
    global_step = 0
    best_loss = math.inf
    ema_score_decay = 0.995
    ema_scores = torch.nn.Parameter(torch.ones(metagraph.n.item()).to(device), requires_grad = False)

    while True:
        # --- Sync + reshape.      
        update_metagraph_peerweight(metagraph, validator, device)

        optimizer = torch.optim.SGD(
            validator.parameters(),
            lr = config.neuron.learning_rate,
            momentum = config.neuron.momentum,
        )
        
        chain_growth = max(0, metagraph.n.item() - torch.numel( ema_scores ))
        ema_scores = torch.nn.Parameter(torch.cat([ema_scores, torch.zeros([chain_growth], dtype=torch.float32, requires_grad=False, device = device)]))

        # --- Run epoch.
        start_block = subtensor.get_current_block() + 1
        end_block = start_block + config.neuron.blocks_per_epoch
        blocks = [ block for block in range(start_block, end_block) ]
        progress = qqdm( blocks, total=len(blocks), desc=format_str('white', f'Epoch'))
        progress.set_bar = partial(progress.set_bar,  element='#')

        # --- Reset the epoch logs
        total_epoch_score = torch.zeros(metagraph.n.item(), device = device)
        total_epoch_loss = 0
        batch_count = 0
        k = min( config.neuron.n_topk_peer_weights,metagraph.n.item() )

        for block in progress:
            
            # --- Training step.
            current_block = subtensor.get_current_block()
            while block >= current_block:

                inputs = next( dataset )
                loss, _ = validator( inputs )
                val_score = validator.scores(loss,inputs)
                
                scores = torch.nn.functional.normalize ( torch.relu( val_score ), p=1, dim = 0 )
                loss.backward()
                clip_grad_norm_(validator.parameters(), config.neuron.clip_gradients)
                optimizer.step()
                optimizer.zero_grad() 
                global_step += 1
                batch_count += 1
                total_epoch_score += scores.detach()
                total_epoch_loss += loss.item()
                ema_scores = F.relu(ema_score_decay * ema_scores.detach() + (1 - ema_score_decay) * scores.detach())
                current_block = subtensor.get_current_block()

            # --- Step logs.
            info = {
                'Step': colored('{}'.format(global_step), 'red'),
                'Epoch': colored('{}'.format(epoch), 'yellow'),
                'Best-loss': colored('{:.4f}'.format(best_loss), 'green'),            
                'Loss': colored('{:.4f}'.format(loss.item()), 'blue'),            
                'nPeers': colored(metagraph.n.item(), 'red'),
                'Stake(\u03C4)': colored('{:.3f}'.format(metagraph.S[uid].item()), 'yellow'),
                'Rank(\u03C4)': colored('{:.3f}'.format(metagraph.R[uid].item()), 'green'),
                'Incentive(\u03C4/block)': colored('{:.6f}'.format(metagraph.I[uid].item()), 'blue'),
                'Dividends': colored('{:.4f}'.format(metagraph.D[ uid ].item()), 'red'),
                'Current Block': colored('{}'.format(block), 'yellow')
            }
            topk_scores, topk_idx = bittensor.unbiased_topk(ema_scores, k, dim=0)
            progress.set_infos( info )
        
        # --- End of epoch
        inactive_uids = torch.where(metagraph.active == 0)[0]
        ema_scores[inactive_uids] = 0
        # --- Set mechanism weights.
        
        topk_scores, topk_uids = bittensor.unbiased_topk( ema_scores, k, dim=0 )
        subtensor.set_weights(
            uids = topk_uids.detach().to('cpu'),
            weights = topk_scores.detach().to('cpu'),
            wallet = wallet,
        )

        # --- Log.
        epoch_loss = total_epoch_loss / batch_count
        epoch_score = total_epoch_score / batch_count
    
        # Push to wandb.
        if config.wandb.api_key != 'default':
            wandb_data = {
                'stake': metagraph.S[ uid ].item(),
                'dividends': metagraph.D[ uid ].item(),
                'epoch_loss': epoch_loss
            } 
            df = pandas.concat( [
                bittensor.utils.indexed_values_to_dataframe( prefix = 'fisher_ema_score', index = topk_uids, values = ema_scores ),
                bittensor.utils.indexed_values_to_dataframe( prefix = 'peer_weight', index = topk_uids, values = validator.peer_weights ),
                dendrite.to_dataframe( metagraph = metagraph )
            ], axis = 1)
            df['uid'] = df.index
            wandb_data_dend = dendrite.to_wandb()
            wandb.log( { **wandb_data, **wandb_data_dend }, step = current_block)
            wandb.log( { 'stats': wandb.Table( dataframe = df ) }, step = current_block)
            
        # --- Save.
        if best_loss > epoch_loss : 
            best_loss = epoch_loss
            torch.save( { 'validator': validator.state_dict() }, "{}/validator.torch".format( config.neuron.full_path ))
        epoch += 1

