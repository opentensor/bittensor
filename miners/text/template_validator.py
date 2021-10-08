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

def config ():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--miner.config', type=str, help='If set, defaults are overridden by passed file.')
    parser.add_argument('--miner.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='template_miner')
    parser.add_argument('--miner.resume', action='store_true', help='resume previous trial.', default=False)
    parser.add_argument('--miner.topk', type=int, help='the number of peers queried during each remote forward call', default=20)
    parser.add_argument('--miner.learning_rate', type=float, help='Training initial learning rate.', default=1)
    parser.add_argument('--miner.learning_rate_chain', type=float, help='Training initial learning rate.', default=1)
    parser.add_argument('--miner.momentum', type=float, help='optimizer momentum.', default=0.8)
    parser.add_argument('--miner.blocks_per_epoch', type=int, help='Blocks per epoch', default=30)
    parser.add_argument('--miner.n_topk_peer_weights', type=int, help='Maximum number of weights to submit to chain', default=100 )
    parser.add_argument('--miner.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--miner.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0)
    parser.add_argument('--nucleus.topk', type=int, help='the number of peers queried during each remote forward call', default=20)
    parser.add_argument('--nucleus.noise_multiplier', type=float, help='Noise standard deviation multiplier. Increases query exploration.', default=1.0)
    parser.add_argument('--nucleus.nhid', type=int, help='the dimension of the feedforward network model in nn.TransformerEncoder', default=200)
    parser.add_argument('--nucleus.nhead', type=int, help='the number of heads in the multiheadattention models', default=2)
    parser.add_argument('--nucleus.nlayers', type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2)
    parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.2)
    parser.add_argument('--wandb.project', type=str, help='''Optionally pass wandb project name for use_wandb''', default='default')
    parser.add_argument('--wandb.run_group', type = str, help='''Optionally pass wandb group name for use_wandb''', default='default')
    
    bittensor.wallet.add_args( parser )
    bittensor.dendrite.add_args( parser )
    bittensor.subtensor.add_args( parser )
    bittensor.logging.add_args( parser )
    bittensor.dataloader.add_args( parser )
    
    # ---- Loads config_file and updates defaults
    config_file_path = vars(parser.parse_known_args()[0])['miner.config']
    if config_file_path:
        config_file_path = os.path.expanduser(config_file_path)
        try:
            with open(config_file_path) as f:
                params_config = yaml.safe_load(f)
                print('Config File Detected at {} updating defaults'.format(config_file_path))
                parser.set_defaults(**params_config)
        except Exception as e:
            print('Error in loading: {} using default parser settings'.format(e))
    
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
    dataset = bittensor.dataloader ( config = config )

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
            self.peer_weights = torch.nn.Parameter(torch.ones( [ metagraph.n.item() ] , requires_grad=True))
            self.logs = SimpleNamespace()

        def forward ( self, inputs ):
            # Apply model.
            query_hidden = self.query( inputs.to( device ) )
            encoded_hidden = self.encoder( query_hidden )
            decoded_targets = self.decoder ( encoded_hidden )

            # Compute loss.
            shift_logits = decoded_targets[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()     
            self.loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) )
            return self.loss, decoded_targets

        def scores ( self ):
            # Computes salience scores for each peer in the network w.r.t the loss. 
            # We use a simplified fishers information score. score_i = hessian_ii * peer_weight_i^2
            peer_weights_d1 = torch.autograd.grad(self.loss, self.peer_weights, create_graph=True, retain_graph=True, allow_unused=True)[0]
            if peer_weights_d1 == None: return torch.ones_like( self.peer_weights ) * (1 / metagraph.n.item()) # None if no grad w.r.t the chain weights.
            peer_weights_d2 = torch.autograd.grad(peer_weights_d1.sum(), self.peer_weights, retain_graph=True, allow_unused=True )[0]
            validator_scores =  peer_weights_d2 * (self.peer_weights**2)/2  
            return validator_scores

        def query ( self, inputs ):

            # ---- Get active peers and their weights ---- 
            active_uids = torch.where(metagraph.active > 0)[0]
            active_peer_weights = self.peer_weights[active_uids]

            # ---- Topk Weights ---- (TODO: check if the gaussians are enough disrupt the chain weights)
            real_topk = min( config.nucleus.topk, metagraph.n.item(), len(active_uids))
            noise = torch.normal( 0, torch.std(active_peer_weights).item()+0.0000001, size=( active_peer_weights.size())).to( config.miner.device )
            topk_weights, topk_idx = torch.topk(active_peer_weights + noise , real_topk, dim=0)
            topk_uids = active_uids[topk_idx]

            # ---- Query network ----
            responses, return_ops, query_times = dendrite.forward_text ( 
                endpoints = metagraph.endpoints[ topk_uids ], 
                inputs = inputs
            )
            
            # ---- Join based on weights ----
            joining_uids = torch.where(return_ops==0)[0]
            joining_weights = F.softmax( topk_weights[(return_ops == 0)], dim = 0 )
            output = torch.zeros( (inputs.shape[0], inputs.shape[1], bittensor.__network_dim__)).to( device )
            for index, joining_weight in enumerate( joining_weights ): 
                output += responses[joining_uids[index]].to( device ) * joining_weight

            # ---- Punish peers with non-successful return ops ----
            with torch.no_grad():
                self.peer_weights[topk_uids[(return_ops != 0)]] -= config.nucleus.punishment
                self.peer_weights[ self.peer_weights < -1 ] = -1 # lower bound for chain weights 

            quested_peers = torch.zeros(metagraph.n.item())
            quested_peers[topk_uids] = 1
            
            responded_peers = torch.zeros(metagraph.n.item())
            responded_peers[topk_uids[joining_uids]] = 1
            
            if len(quested_peers) > len(self.logs.quested_peers_count):
                fill = torch.zeros(len(quested_peers) - len(self.logs.quested_peers_count))
                self.logs.quested_peers_count = torch.cat((self.logs.quested_peers_count, fill))
                self.logs.responded_peers_count = torch.cat((self.logs.responded_peers_count, fill))

            self.logs.quested_peers_count += quested_peers
            self.logs.responded_peers_count += responded_peers

            return output

    # Create validator model.
    validator = Validator( config = config ).to( device )
    
    optimizer = torch.optim.SGD(
        [ {'params': validator.peer_weights, 'lr': config.miner.learning_rate_chain} ],
        lr = config.miner.learning_rate,
        momentum = config.miner.momentum,
    )

    scheduler= torch.optim.lr_scheduler.StepLR(optimizer,
        step_size= 1.0,
        gamma=0.95
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
        except: 
            pass
    torch.save( { 'validator': validator.state_dict() }, "{}/validator.torch".format( save_path ))

    # --- Run Forever.
    epoch = 0
    global_step = 0
    best_loss = math.inf
    ema_score_decay = 0.97
    ema_scores = torch.ones_like( validator.peer_weights ) * (1 / metagraph.n.item()) 

    while True:
    
        # --- Sync + reshape.      
        metagraph.sync().save()
        chain_growth = metagraph.n.item() - torch.numel( validator.peer_weights )
        validator.peer_weights = torch.nn.Parameter(torch.cat( [validator.peer_weights, torch.ones([chain_growth], dtype=torch.float32, requires_grad=True)])).to(device)
        ema_scores = torch.nn.Parameter(torch.cat( [ema_scores, torch.ones([chain_growth], dtype=torch.float32, requires_grad=True)])).to(device)

        # --- Run epoch.
        start_block = subtensor.get_current_block() + 1
        end_block = start_block + config.miner.blocks_per_epoch
        blocks = [ block for block in range(start_block, end_block) ]
        progress = qqdm( blocks, total=len(blocks), desc=format_str('white', f'Epoch'))

        # --- Reset the epoch logs
        validator.logs.quested_peers_count = torch.zeros(0)
        validator.logs.responded_peers_count = torch.zeros(0)
        total_epoch_val_score = torch.zeros(metagraph.n.item())
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
                clip_grad_norm_(validator.parameters(), config.miner.clip_gradients)
                optimizer.step()
                optimizer.zero_grad() 
                global_step += 1
                batch_count += 1
                total_epoch_val_score += val_score
                total_epoch_score += scores
                total_epoch_loss += loss.item()


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
            
            for uid_i, score_i in enumerate(scores.tolist()):
                if score_i != 0:
                    color =  'green' if score_i - ema_scores[ uid_i ] > 0 else 'red'
                    info[ 'fi_' + str(uid_i) ] = colored('{:.4f}'.format( score_i ), color)
                    
                    weight_wo_norm = validator.peer_weights[uid_i]
                    info[ 'pw_' + str(uid_i) ] = colored('{:.4f}'.format( weight_wo_norm ), color)
            
            ema_scores = ema_score_decay * ema_scores + (1 - ema_score_decay) * scores
            
            progress.set_infos( info )
        
        # --- End of epoch
        # --- Set mechanism weights.
        topk_scores, topk_uids = torch.topk( ema_scores, k = min(config.miner.n_topk_peer_weights, metagraph.n.item())  )
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
        epoch_val_score = total_epoch_val_score / batch_count
        
        wandb_data = {
            'Stake': metagraph.S[ uid ].item(),
            'Dividends': metagraph.D[ uid ].item(),
            'Epoch_loss': epoch_loss
        } 

        norm_weights = F.softmax( validator.peer_weights.detach() )
        respond_rate = validator.logs.responded_peers_count / validator.logs.quested_peers_count
        
        for uid_j in topk_uids.tolist():
            uid_str = str(uid_j).zfill(3)
            wandb_data[ f'Fisher ema uid: {uid_str}' ] = ema_scores[uid_j]
            wandb_data[ f'Fisher epoch val score uid: {uid_str}' ] = epoch_val_score[uid_j]
            wandb_data[ f'Fisher epoch score uid: {uid_str}' ] = epoch_score[uid_j]
            wandb_data[ f'Peer weight (norm) uid:{uid_str}' ] = norm_weights[uid_j]
            wandb_data[ f'Peer weight (w/o norm) uid:{uid_str}' ] = validator.peer_weights[uid_j]
            wandb_data[ f'Quested uid: {uid_str}' ]= validator.logs.quested_peers_count[uid_j]
            wandb_data[ f'Responded uid: {uid_str}' ]= validator.logs.responded_peers_count[uid_j]
            wandb_data[ f'Respond rate uid: {uid_str}' ]= respond_rate[uid_j]

        wandb.log( wandb_data )
        
        # --- Save.
        if best_loss > epoch_loss : 
            best_loss = epoch_loss
            torch.save( { 'validator': validator.state_dict() }, "{}/validator.torch".format( save_path ))
        epoch += 1


if __name__ == "__main__":
    main( config() )