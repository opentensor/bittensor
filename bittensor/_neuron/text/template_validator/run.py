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
from termcolor import colored
from functools import partial

from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from qqdm import qqdm, format_str
from loguru import logger; logger = logger.opt(colors=True)

def run( config , validator, subtensor, wallet, metagraph, dataset, device, uid, dendrite):
    print(config)
    config.to_defaults()

    optimizer = torch.optim.SGD(
        [ {'params': validator.peer_weights, 'lr': config.neuron.learning_rate_chain} ],
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

        wandb.watch( validator, log = 'all', log_freq = 50 )

    # Optionally resume.
    if config.neuron.resume:
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
    ema_scores = torch.ones_like( validator.peer_weights ) * (1 / metagraph.n.item()) 

    while True:
    
        # --- Sync + reshape.      
        metagraph.sync().save()
        chain_growth = metagraph.n.item() - torch.numel( validator.peer_weights )
        validator.peer_weights = torch.nn.Parameter(torch.cat( [validator.peer_weights, torch.ones([chain_growth], dtype=torch.float32, requires_grad=True)])).to(device)
        ema_scores = torch.nn.Parameter(torch.cat( [ema_scores, torch.ones([chain_growth], dtype=torch.float32, requires_grad=True)])).to(device)

        # --- Run epoch.
        start_block = subtensor.get_current_block() + 1
        end_block = start_block + config.neuron.blocks_per_epoch
        blocks = [ block for block in range(start_block, end_block) ]
        progress = qqdm( blocks, total=len(blocks), desc=format_str('white', f'Epoch'))
        progress.set_bar = partial(progress.set_bar,  element='#')

        # --- Reset the epoch logs
        total_epoch_score = torch.zeros(metagraph.n.item())
        total_epoch_loss = 0
        batch_count = 0
        
        for block in progress:
            
            # --- Training step.
            while block >= subtensor.get_current_block():
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
                ema_scores = ema_score_decay * ema_scores.detach() + (1 - ema_score_decay) * scores.detach()


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
            for uid, ema_score in zip(topk_idx, topk_scores) :
                color =  'green' if scores[uid] - ema_score > 0 else 'red'
                info[f'uid_{uid.item()}'] = colored('{:.4f}'.format(ema_score), color) 
            
            
            progress.set_infos( info )
        
        # --- End of epoch
        # --- Set mechanism weights.
        topk_scores, topk_uids = torch.topk( ema_scores.detach(), k = min(config.neuron.n_topk_peer_weights, metagraph.n.item())  )
        subtensor.set_weights (
            uids = topk_uids,
            weights = topk_scores,
            wallet = wallet,
            wait_for_inclusion = False,
        )    

        # --- Log.
        metagraph.sync().save()
        epoch_loss = total_epoch_loss / batch_count
        epoch_score = total_epoch_score / batch_count
        
        wandb_data = {
            'stake': metagraph.S[ uid ].item(),
            'dividends': metagraph.D[ uid ].item(),
            'epoch_loss': epoch_loss
        } 

        norm_weights = F.softmax( validator.peer_weights.detach(), dim=0 )
        
        for uid_j in topk_uids.tolist():
            uid_str = str(uid_j).zfill(3)
            wandb_data[ f'fisher_ema uid: {uid_str}' ] = ema_scores[uid_j]
            wandb_data[ f'fisher_epoch_score uid: {uid_str}' ] = epoch_score[uid_j]
            wandb_data[ f'peer_norm_weight uid:{uid_str}' ] = norm_weights[uid_j]
            wandb_data[ f'peer_wo_norm_weight uid:{uid_str}' ] = validator.peer_weights.detach()[uid_j]
        
        
        if config.wandb.api_key != 'default':
            wandb_data_dend = dendrite.to_wandb()
            wandb.log( {**wandb_data, **wandb_data_dend} )

        # --- Save.
        if best_loss > epoch_loss : 
            best_loss = epoch_loss
            torch.save( { 'validator': validator.state_dict() }, "{}/validator.torch".format( config.neuron.full_path ))
        epoch += 1

