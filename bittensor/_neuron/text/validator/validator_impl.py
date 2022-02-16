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
""" The bittensor base validator

Example:
    $ python miners/text/validator.py --logging.debug

"""
import sys
import bittensor
import torch
import wandb
import math
import pandas
import traceback
from rich import print
from rich.console import Console
from rich.traceback import install

from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from loguru import logger
logger = logger.opt( colors=True )
console = Console()
install(show_locals=True)

class Neuron:

    def __init__( self, 
        config,
        wallet,
        subtensor,
        metagraph,
        dendrite,
        dataset,
        nucleus,
        device,
    ):
        self.config = config
        self.wallet = wallet
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.dendrite = dendrite
        self.dataset = dataset
        self.nucleus = nucleus  
        self.device = device
        self.global_step = 0
        self.epoch = 0

    def __exit__ ( self, exc_type, exc_value, exc_traceback ):
        r""" Close down neuron.
        """
        print(exc_type, exc_value, exc_traceback)

    def __enter__(self):
        r""" Sanity checks and begin validator.
        """
        # === Wallet ===
        # Checks that the validator has a valid uid (is registered on the network.)
        # If the wallet has not been registered. sys.exit().
        # If the network is mocked, we register.
        if self.subtensor.network != 'mock':
            if not self.wallet.is_registered( subtensor = self.subtensor ):
                logger.critical( "You must register the validator's wallet before running, use: btcli register --wallet.name {} --wallet.hotkey {}", self.wallet.name, self.wallet.hotkey_str)
                sys.exit(0)
        else:
            self.wallet.register( subtensor = self.subtensor )

        # === UID ===
        # Get our uid from the chain. 
        # At this point we should have a uid because we are already registered.
        self.uid = self.wallet.get_uid( subtensor = self.subtensor )    

        # === Monitoring ===
        # Optionally set up wandb logging.
        if self.config.wandb.api_key != 'default':
            bittensor.wandb(
                config = self.config,
                cold_pubkey = self.wallet.coldkeypub.ss58_address,
                hot_pubkey = self.wallet.hotkey.ss58_address,
                root_dir = self.config.neuron.full_path
            )
        
    def run ( self ):
        r""" Run the validator and terminate on Keyboard interrupt.
        """
        with self:
            while True:
                try:
                    self.run_epoch()
                    self.epoch += 1
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print( 'Unknown exception: {} with traceback {}', e, )
                    console.print_exception(show_locals=True)
                    if not self.config.neuron.restart_on_failure:
                        break

    def run_epoch( self ):
        r""" Runs a validator epoch. We apply batches until the epoch length is exhausted.
            Occasionally the validator nucleus is completely reset to ensure we dont converge to far.
            At the end of the epoch we set weights on the chain and optionally log to wandb.
        """
            
        # === Setup Epoch ===
        # Reset epoch scores history.
        # Reset the validator weights ever x epochs.
        self.metagraph.sync().save()
        self.epoch_scores = torch.zeros( ( self.metagraph.n.item() ) )
        if self.epoch % self.config.neuron.epochs_until_reset == 0:
            self.nucleus.reset_weights()
            self.optimizer = torch.optim.SGD ( self.nucleus.parameters(), lr = self.config.neuron.learning_rate, momentum = self.config.neuron.momentum )

        # === Run Epoch ===
        # Each block length lasts blocks_per_epoch blocks.
        # This gives us a consistent network wide timer.
        # Here we run until blocks_per_epochs have progressed.
        start_block = self.subtensor.block
        while self.subtensor.block < start_block + self.config.neuron.blocks_per_epoch:
            # === Forward ===
            loss, scores = self.nucleus( next( self.dataset ), self.metagraph, self.dendrite )

            # === Backward ===
            loss.backward()

            # === Apply gradients ===
            clip_grad_norm_(self.nucleus.parameters(), self.config.neuron.clip_gradients)
            self.optimizer.step()
            self.optimizer.zero_grad()      

            # === Logs + state update ===
            self.global_step += 1
            self.epoch_scores += scores.sum( 0 )
            print( '\n\t epoch:', self.epoch, '\t step:', self.global_step, '\t blocks:', self.subtensor.block - start_block, '/', self.config.neuron.blocks_per_epoch )
            print( 'scores:\n', pandas.DataFrame( scores.detach().sum(0) ).describe() ) 

        # === Set weights ===
        # Find the n_topk_peer_weights peers to set weights to.
        # We use the mean of the epoch weights.
        self.epoch_scores = self.epoch_scores / self.epoch_scores.sum()
        topk_scores, topk_uids = bittensor.unbiased_topk( self.epoch_scores, k = min(self.config.neuron.n_topk_peer_weights, self.metagraph.n.item())  )
        print( '\t Setting weights:\n\t', list( zip(  topk_uids.tolist(), topk_scores.tolist() ) ) )
        self.subtensor.set_weights(
            uids = topk_uids.detach().to('cpu'),
            weights = topk_scores.detach().to('cpu'),
            wallet = self.wallet,
        )

        # === Wandb Logs ===
        # Optionally send validator logs to wandb.
        if self.config.wandb.api_key != 'default':
            wandb_data = { 'stake': self.metagraph.S[ self.uid ].item(), 'dividends': self.metagraph.D[ self.uid ].item() } 
            df = pandas.concat( [
                bittensor.utils.indexed_values_to_dataframe( prefix = 'weights', index = topk_uids, values = self.epoch_scores ),
                self.dendrite.to_dataframe( metagraph = self.metagraph )
            ], axis = 1); df['uid'] = df.index
            wandb_data_dend = self.dendrite.to_wandb()
            wandb.log( { **wandb_data, **wandb_data_dend }, step = self.subtensor.block )
            wandb.log( { 'stats': wandb.Table( dataframe = df ) }, step = self.subtensor.block )


class Nucleus( torch.nn.Module ):

    def __init__(self, config, device):
        super(Nucleus, self).__init__()
        self.embedding = torch.nn.Embedding( bittensor.__vocab_size__,  bittensor.__network_dim__ )
        self.layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout, batch_first=True)
        self.encoder = TransformerEncoder( self.layers, config.nucleus.nlayers )
        self.c_layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout, batch_first=True)
        self.local_encoder = TransformerEncoder(self.c_layers, 1)
        self.decoder = torch.nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__ , bias=False)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.config = config
        self.device = device
        self.gates = {}
        for uid in range(2000):
            self.gates[uid] = torch.nn.Linear( bittensor.__network_dim__, 1, bias=True).to( self.device )

    def reset_weights ( self ):
        r""" Resets the validator weights.
        """
        torch.nn.init.xavier_uniform_ ( self.embedding.weight )
        torch.nn.init.xavier_uniform_ ( self.decoder.weight )
        def init_xavier( component ):
            try:
                torch.nn.init.xavier_uniform_( component.weight )
            except: pass
        self.encoder.apply( init_xavier )
        self.local_encoder.apply( init_xavier )
        for uid in range(2000):
            torch.nn.init.xavier_uniform_( self.gates[uid].weight )

    def forward ( 
        self, 
        inputs: torch.FloatTensor,
        metagraph: 'bittensor.Metagraph',
        dendrite: 'bittensor.Dendrite',
    ):
        r""" Forward validator pass. Selects peer to query, joins results and computes scoring.
            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, *-1*)`, `required`): 
                    Tensor inputs to distribute to neurons using query context.
                metagraph (TODO):
                dendrite (TODO):
            Returns:
                total_loss (TODO):
                decoded_targets (TODO):
                query_uids (TODO):
                scores (TODO):
        """
        # === Apply model ===
        active_uids = torch.where( metagraph.active > 0)[0]
        query_hidden, _, weights = self.query( inputs, metagraph, dendrite )
        encoded_hidden = self.encoder( query_hidden )
        decoded_targets = self.decoder ( encoded_hidden )

        # === Create variance loss ===
        importance_loss = self.config.nucleus.importance  * ( torch.std(self.total_weights[active_uids])/torch.mean(self.total_weights[active_uids]) )**2

        # === Compute loss targets ===
        shift_logits = decoded_targets[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()     
        loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) ) 
        total_loss = loss + importance_loss
        return total_loss, weights

    def query ( 
        self, 
        inputs: torch.FloatTensor,
        metagraph: 'bittensor.Metagraph',
        dendrite: 'bittensor.Dendrite',
    ):
        r""" Query into the network and get joined results.
                Args:
                    inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, *-1*)`, `required`): 
                        Tensor inputs to distribute to neurons using query context.
                    metagraph (TODO):
                    dendrite (TODO):
                Returns:
                    query_hidden (TODO): 
                    query_uids (TODO):

            """
        # === Create the local context ===
        local_context = self.local_encoder( self.embedding( inputs ) )* math.sqrt( bittensor.__network_dim__ )

        # === Query network ===
        endpoints, topk_weights, query_uids, weights = self.route( inputs, local_context, metagraph, dendrite )
        responses, return_ops, _ = dendrite.forward_text ( 
            endpoints = endpoints, 
            inputs = inputs
        )

        # === Join based on weights ===
        joining_uids = torch.where(return_ops== bittensor.proto.ReturnCode.Success)[0]
        joining_weights = F.softmax( topk_weights[(return_ops == bittensor.proto.ReturnCode.Success)], dim = 0 )
        output = torch.zeros( (inputs.shape[0], inputs.shape[1], bittensor.__network_dim__)).to( self.device )
        for index, joining_weight in enumerate( joining_weights ): 
            output += responses[joining_uids[index]].to( self.device ) * joining_weight

        return output, query_uids[joining_uids], weights

    def route(
            self, 
            inputs: torch.FloatTensor, 
            context: torch.FloatTensor,
            metagraph: 'bittensor.Metagraph',
            dendrite: 'bittensor.Dendrite',
        ):
            r""" Routes inputs using context and metagraph state.
                Args:
                    inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, *-1*)`, `required`): 
                        Tensor inputs to distribute to neurons using query context.
                    context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, *-1*)`, `required`): 
                        Tensor context of the inputs generated by the local_encoder 
                Returns:
                    filtered_endpoints (:obj:`List[bittensor.Endpoint]` of shape :obj:`(config.neuron.topk)`, `required`)
                        Joined responses from each queried neuron.
                    topk_weights (:obj:`torch.FloatTensor` of shape :obj:`(config.neuron.topk)`, `required`): 
                        Batchwise weights for each of the top k peers.
            """
            # === Get weights for uids ===
            # weights: (torch.FloatTensor): weights for each filtered_uid
            # weights.shape = [ batch size, n_filtered ]
            # gates use the last token for context 
            weights = torch.cat( [ self.gates[ uid ](context[:,-1,:]) for uid in metagraph.uids.tolist() ], axis = 1)

            # === Normalize weights across batch dimension ===
            # filtered_mean_weights: (torch.FloatTensor): normalized weights across batch dimension. 
            # filtered_mean_weights.shape = [ n_filtered ]
            filtered_mean_weights = torch.mean(weights, axis = 0)
            noise = torch.normal( 0, torch.std(filtered_mean_weights).item(), size=( filtered_mean_weights.size())).to( self.config.neuron.device )

            self.total_weights = torch.zeros( metagraph.n.item(), device = self.device)
            for i in range(len(metagraph.uids.tolist())):
                self.total_weights[metagraph.uids[i]] = filtered_mean_weights[i] + noise[i]

            # === Get indices and values for uids with highest scores ===
            # topk_weights: (torch.float64): scores of uids with highest scores.
            # topk_weights.shape = [ real_topk ]
            # topk_indices: (torch.LongTensor): indicies of uids with highest scores.
            # topk_indices.shape = [ real_topk ]
            real_topk = min( len(metagraph.uids.tolist()), self.config.neuron.topk )
            topk_weights, topk_indices = torch.topk(self.total_weights[metagraph.uids], real_topk, dim=0)

            # === Get endpoint information for the highest scoring uids ===
            # filtered_endpoints: List[bittensor.endpoints]: endpoint information for filtered uids.
            # len(neurons) == real_topk
            filtered_endpoints = []
            for uid in metagraph.uids[topk_indices]:
                filtered_endpoints.append( metagraph.endpoints[ uid ] )
            return filtered_endpoints, topk_weights, metagraph.uids[topk_indices], weights