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
""" The SGMOE base validator

Example:
    $ python miners/text/sgmoe_validator.py --logging.debug

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

def run( config , validator, subtensor, wallet, metagraph, dataset, device, uid, dendrite):
    print(config)
    config.to_defaults()
    validator = validator.to(device)
    optimizer = torch.optim.SGD(
        validator.parameters(),
        lr = config.neuron.learning_rate,
        momentum = config.neuron.momentum,
    )
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

    # --- last sync block 
    last_sync_block = subtensor.get_current_block()

    # --- Run Forever.
    epoch = 0
    global_step = 0
    best_loss = math.inf
    ema_score_decay = 0.995
    ema_scores = torch.nn.Parameter(torch.zeros_like(validator.peer_weights, device = device) * (1 / metagraph.n.item()), requires_grad = False)

    while True:

        # --- Sync + reshape.      
        chain_growth = max(0, metagraph.n.item() - torch.numel( validator.peer_weights ))
        validator.sync_with_chain_state()
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
        
        for block in progress:
            
            # --- Training step.
            current_block = subtensor.get_current_block()
            while block >= current_block:
                loss, _ = validator( next( dataset ) )
                val_score = validator.scores()
                scores = torch.nn.functional.normalize ( torch.relu( val_score ), p=1, dim = 0 )
                loss.backward()
                clip_grad_norm_(validator.parameters(), config.neuron.clip_gradients)
                optimizer.step()
                optimizer.zero_grad() 
                global_step += 1
                batch_count += 1
                total_epoch_score += scores.detach()
                total_epoch_loss += loss.item()
                ema_scores = (ema_score_decay * ema_scores) + (1 - ema_score_decay) * scores.detach()
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
            
            topk_scores, topk_idx = torch.topk(ema_scores, 5, dim=0)
            for idx, ema_score in zip(topk_idx, topk_scores) :
                color =  'green' if scores[idx] - ema_score > 0 else 'red'
                info[f'uid_{idx.item()}'] = colored('{:.4f}'.format(ema_score), color) 
            
            
            progress.set_infos( info )
        
        # --- End of epoch
        # --- Set mechanism weights.
        topk_scores, topk_uids = torch.topk( ema_scores.detach(), k = min(config.neuron.n_topk_peer_weights, metagraph.n.item()))
        subtensor.timeout_set_weights(
            timeout=10,
            uids = topk_uids.to('cpu'),
            weights = topk_scores.to('cpu'),
            wait_for_inclusion = True,
            wallet = wallet,
        )

        # --- Log.
        #metagraph.sync().save()
        epoch_loss = total_epoch_loss / batch_count
        epoch_score = total_epoch_score / batch_count
        active_uids = torch.where(metagraph.active > 0)[0]
                
        if config.wandb.api_key != 'default':
            wandb_data = {
                'stake': metagraph.S[ uid ].item(),
                'dividends': metagraph.D[ uid ].item(),
                'epoch_loss': epoch_loss,
                'STD in scores': torch.std(ema_scores[active_uids]).item(),
                
            } 
            df = pandas.concat( [
                bittensor.utils.indexed_values_to_dataframe( prefix = 'fisher_ema_score', index = topk_uids, values = ema_scores ),
                dendrite.to_dataframe( metagraph = metagraph )
            ], axis = 1)
            df['uid'] = df.index
            wandb_dendrite = dendrite.to_wandb()
            wandb.log( {**wandb_data, **wandb_dendrite}, step = current_block )
            wandb.log( { 'stats': wandb.Table( dataframe = df ) }, step = current_block )

        # --- Save.
        if best_loss > epoch_loss : 
            best_loss = epoch_loss
            torch.save( { 'validator': validator.state_dict() }, "{}/validator.torch".format( config.neuron.full_path ))

        if current_block - last_sync_block > config.neuron.metagraph_sync:
            metagraph.sync()
            last_sync_block = current_block

        epoch += 1

