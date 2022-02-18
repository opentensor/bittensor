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
import time
import wandb
import math
import pandas
import traceback
from rich import print
from rich.console import Console
from rich.traceback import install
from ..neuron_utilities import joining_context, partial_contexts
import torch.nn as nn

from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from loguru import logger
logger = logger.opt( colors=True )
console = Console()
install(show_locals=True)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Neuron:
    """ Neuron class which drives the training of the validator.
    
    """
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
        self.using_wandb = self.config.wandb.api_key != 'default'

    def __exit__ ( self, exc_type, exc_value, exc_traceback ):
        r""" Close down neuron.
        """
        print(exc_type, exc_value, exc_traceback)
        self.dataset.close()
        self.dendrite.__del__()

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
        if self.using_wandb:
            bittensor.wandb(
                config = self.config,
                cold_pubkey = self.wallet.coldkeypub.ss58_address,
                hot_pubkey = self.wallet.hotkey.ss58_address,
                root_dir = self.config.neuron.full_path
            )
        
    def run ( self ):
        r""" Run the validator and terminate on Keyboard interrupt.
        """
         
        # === Setup ===
        # Checks wallet and starts monitoring with wandb.
        with self:

            # === Run ===
            # Iterates through epochs.
            while True:
                try:

                    # === Epoch ===
                    # Each epoch runs for blocks_per_epoch and resets
                    # the model every epochs_until_reset.
                    self.run_epoch()
                    self.epoch += 1

                # === Stops on interrupt otherwise restarts ===
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print_exception(show_locals=False)
                    print( traceback.format_exc() )
                    print( 'Unknown exception: {}', e )
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
        epoch_steps = 0
        score_history = []
        moving_avg_history = []
        if self.epoch % self.config.neuron.epochs_until_reset == 0:
            # Resetting the weights here.
            self.nucleus.reset_weights()
            self.optimizer = torch.optim.SGD ( 
                self.nucleus.parameters(), lr = self.config.neuron.learning_rate, momentum = self.config.neuron.momentum 
            )

        # === Run Epoch ===
        # Each block length lasts blocks_per_epoch blocks.
        # This gives us a consistent network wide timer.
        # Here we run until blocks_per_epochs have progressed.
        start_block = self.subtensor.block
        current_block = start_block
        while self.subtensor.block < start_block + self.config.neuron.blocks_per_epoch:
            start_time = time.time()

            # === Forward ===
            # Forwards inputs through the network and returns the loss
            # and endpoint scores using shapely approximation of salience.
            loss, scores = self.nucleus( next( self.dataset ), self.metagraph, self.dendrite )

            # === Backward ===
            # Backwards gradients through model to train gating and remote endpoints.
            loss.backward()

            # === Apply gradients ===
            # Applies local gradients to parameters.
            clip_grad_norm_(self.nucleus.parameters(), self.config.neuron.clip_gradients)
            self.optimizer.step()
            self.optimizer.zero_grad()    

            # === Normalize scores ===
            # Updates moving averages and history.
            scores = torch.nn.functional.normalize ( scores , p=2, dim = 0 ) # Normalized step scores.
            score_history.append( scores ) # Save score history.
            moving_avg_scores = torch.stack( score_history ).mean(0) # Average history.
            moving_avg_scores = moving_avg_scores / moving_avg_scores.sum() # Normalize moving average.
            moving_avg_history.append( moving_avg_scores ) # Save score history.
        
            # === Logs + state update ===
            # Prints step logs to screen.
            epoch_steps += 1
            self.global_step += 1
            current_block = self.subtensor.block
            zipped_scores = list( zip( self.metagraph.uids[ moving_avg_scores > 0.0 ].tolist() , moving_avg_scores [moving_avg_scores > 0.0 ].tolist() ) ) 
            sorted_mvg_scores = sorted(zipped_scores, key=lambda x: x[1])
            print( '\n\t epoch:', self.epoch, '\t step:', self.global_step, '\t blocks:', current_block - start_block, '/', self.config.neuron.blocks_per_epoch )
            print( 'scores:\n', sorted_mvg_scores)
            if self.using_wandb:
                step_stats = pandas.DataFrame( moving_avg_scores ).describe()
                wandb.log(
                    {  
                        'loss': loss.item(),
                        'time': time.time() - start_time,
                        'global_step': self.global_step,
                        'scount': step_stats[0]['count'], 
                        'smax': step_stats[0]['max'], 
                        'smean': step_stats[0]['mean'], 
                        'sstd': step_stats[0]['std'], 
                        'smin': step_stats[0]['min'], 
                        's25%': step_stats[0]['25%'], 
                        's50%': step_stats[0]['50%'], 
                        's75%': step_stats[0]['75%'], 
                    }, 
                    step = current_block
                )


        # === Set weights ===
        # Find the n_topk_peer_weights peers to set weights to.
        # We use the mean of the epoch weights.
        print (moving_avg_scores.tolist())
        topk_scores, topk_uids = bittensor.unbiased_topk( moving_avg_scores, k = min(self.config.neuron.n_topk_peer_weights, self.metagraph.n.item())  )
        self.subtensor.set_weights(
            uids = topk_uids.detach().to('cpu'),
            weights = topk_scores.detach().to('cpu'),
            wallet = self.wallet,
        )

        # === Wandb Logs ===
        # Optionally send validator logs to wandb.
        if self.using_wandb:
            # Logging history to wandb.
            df = pandas.concat( [
                bittensor.utils.indexed_values_to_dataframe( prefix = 'weights', index = topk_uids, values = moving_avg_scores ),
                self.dendrite.to_dataframe( metagraph = self.metagraph )
            ], axis = 1); df['uid'] = df.index
            wandb_data_dend = self.dendrite.to_wandb()
            wandb_data = { 'stake': self.metagraph.S[ self.uid ].item(), 'dividends': self.metagraph.D[ self.uid ].item() } 
            wandb.log( { 'stats': wandb.Table( dataframe = df ) }, step = current_block )
            wandb.log( { **wandb_data, **wandb_data_dend }, step = current_block )
            wandb.log( {'scores:': wandb.plot.line_series (
                xs = list( range ( len ( moving_avg_history ) ) ), 
                ys = torch.transpose( torch.stack( moving_avg_history ), 0, 1).tolist(),
                keys = self.metagraph.uids.tolist(),
                title = "Epoch scores",
                xname = "steps"
            )}, step = current_block)


class Nucleus( torch.nn.Module ):
    """ Nucleus class which holds the validator model.
    """
    def __init__(self, config, device):
        super(Nucleus, self).__init__()
        self.config = config
        self.device = device

        # Token embeddings project int64 tokens onto representations.
        self.token_embedding = torch.nn.Embedding( bittensor.__vocab_size__,  bittensor.__network_dim__ )
        
        # Routing encoder, projects token embeddings onto context for routing inputs.
        self.routing_encoder_layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout, batch_first=True)
        self.routing_encoder = TransformerEncoder(self.routing_encoder_layers, 1)

        # Encoder projects response representations onto hidden units.
        self.encoder_layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout, batch_first=True)
        self.encoder = TransformerEncoder( self.encoder_layers, config.nucleus.nlayers )

        # Decoder which projects hidden unit representations on to the token dimension.
        self.decoder = torch.nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__ , bias=False)

        # Positional Encoding
        self.local_pos_encoder = PositionalEncoding(bittensor.__network_dim__, self.config.nucleus.dropout)

        # Crosss entropy loss for NTP.    
        self.loss_fct = torch.nn.CrossEntropyLoss()
    
        # SGMOE Gates: Instantiating the gates per expert.
        self.gates = {}
        for uid in range(2000):
            self.gates[uid] = torch.nn.Linear( bittensor.__network_dim__, 1, bias=True).to( self.device )

    def reset_weights ( self ):
        r""" Resets the validator weights.
        """
        # === Resets all the weights using xavier initialization. ===
        torch.nn.init.xavier_uniform_ ( self.token_embedding.weight )
        torch.nn.init.xavier_uniform_ ( self.decoder.weight )
        def init_xavier( component ):
            try:
                torch.nn.init.xavier_uniform_( component.weight )
            except: pass
        self.routing_encoder.apply( init_xavier )
        self.encoder.apply( init_xavier )
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
                metagraph (bittensor.Metagraph):
                    Metagraph object used to query network information.
                dendrite (bittensor.Dendrite):
                    Dendrite RPC client used to make network queries.
            Returns:
                global_loss (torch.FloatTensor, [1] ):
                    Loss for training validator nucleus.
                scores (torch.FloatTensor, [ metagraph.n ]):
                    Scores per endpoint for this batch.
        """        
        # === Create the local context used to select endpoints ===
        # The context tensor returns a hidden unit representation for the text inputs
        # this context can be used as input to the gates in the next step.
        # embedding: retrieve learned representation vectors for input vocabulary tokens.
        # inputs.shape = [batch_size, sequence_len]
        # embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        embedding =  self.token_embedding( inputs )* math.sqrt( bittensor.__network_dim__ )

        # src_mask: attention mask adds -inf to positions not allowed to attend, preventing forward-looking when
        #           predicting each token in the sequence.
        # src_mask.shape = [sequence_len, sequence_len]
        src_mask = torch.triu(torch.ones(embedding.size(1), embedding.size(1)) * float('-inf'), diagonal=1)
        src_mask = src_mask.to(self.config.neuron.device)

        # pos_embedding: adding positional encoding to embedding.
        # pos_embedding.shape = [sequence_len, batch_size, bittensor.__network_dim__]
        pos_embedding = self.local_pos_encoder(embedding)

        # routing_context: (torch.FloatTensor): context tensor which is used to select endpoints.
        # routing_context.shape = [ batch size, __network_dim__ ]
        routing_context = self.routing_encoder( pos_embedding, mask = src_mask )

        # === Get weights for uids. ===
        # We iterate over each of the network uids and compute a querying score for each
        # using the gating function. This returns a score per endpoint per example.
        # routing_weights: (torch.FloatTensor): score per example, per endpoint.
        # routing_weights.shape = [ batch size, __network_n__ ]
        # The gates act over the last embedding of the routing_context.
        active_uids = torch.where(metagraph.active > 0)[0]
        active_routing_weights = torch.cat( [ self.gates[ uid ](routing_context[:,-1,:]) for uid in active_uids.tolist() ], axis = 1)

        # === Normalize routing_weights across batch dimension and add noise. ===
        # We are summing across the batch dimension to create a per-batch score per endpoint.
        # The resulting routing_weights tensor is a score per expert.
        # routing_weights: (torch.FloatTensor): normalized weights across batch dimension with noise.
        # routing_weights.shape = [ n_filtered ]
        batchwise_routing_weights = torch.mean(active_routing_weights, axis = 0)
        noisy_routing_weights = torch.normal( 0, torch.std(batchwise_routing_weights).item(), size=( batchwise_routing_weights.size())).to( self.config.neuron.device )
        noisy_routing_weights =  batchwise_routing_weights + noisy_routing_weights

        # === Get indices and values for uids with highest scores ===
        # We are taking the topk routing weights and returning their uids.
        # First we ensure topk is smaller than the network size then use the torch.topk.
        # topk_routing_weights: (torch.float64): scores of uids with highest scores.
        # topk_routing_weights.shape = [ real_topk ]
        # topk_routing_uids: (torch.LongTensor): uids with highest scores.
        # topk_routing_uids.shape = [ real_topk ]
        real_topk = min( len( active_uids ), self.config.nucleus.topk )
        top_k_routing_weights, routing_uids = torch.topk( noisy_routing_weights, real_topk, dim=0)

        # === Get endpoint information for the highest scoring uids ===
        # We index into the metagraph's endpoints and return a list of the filtered set of endpoints we wish to query.
        # routing_endpoints: List[bittensor.endpoints]: endpoint information for filtered uids.
        # len(neurons) == real_topk
        routing_endpoints = [ metagraph.endpoints[ uid ] for uid in active_uids[routing_uids] ]

        # === Query the endpoints ===
        # Makes the dendrite call into the network returning the representations 
        # for each of the endpoints. The return ops can be used to filter weights and outputs.
        # query_responses: (List[torch.float64]): responses from each endpoint.
        # query_responses.shape = real_topk * [ batch_size, sequence_len, __network_dim__ ]
        # return_ops: (torch.int64): Return ops.
        # return_ops.shape = [ real_topk ]
        query_responses, return_ops, times = dendrite.forward_text ( 
            endpoints = routing_endpoints, 
            inputs = inputs
        )
        # Send responses to device. This is required to ensure we move the responses
        # Onto the correct device.
        for response in query_responses:
            response.to( self.device )

        # === Compute loss given joined responses ===
        # This function computes target loss for next token prediction given 
        # the joined responses as a hidden unit input.
        # target_loss: (torch.float64): loss after decoding responses to targets.
        # target_loss.shape = [ 1 ]
        def get_target_loss ( hidden, targets ):
            # hidden: (torch.float64): [ batch_size, sequence_len, __network_dim__ ]
            #   Hidden units which are encoded and decoded onto targets for loss computation.
            # targets: (torch.float64): [n]
            #   Token targets,
            src_mask = torch.triu(torch.ones(hidden.size(1), hidden.size(1)) * float('-inf'), diagonal=1)
            src_mask = src_mask.to(self.config.neuron.device)
            encoded_hidden = self.encoder( hidden, mask = src_mask )
            decoded_targets = self.decoder( encoded_hidden )
            shift_logits = decoded_targets[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            return self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) )

        # === Compute global loss ===
        # Computes the global training loss for the nucleus by decoding all the responses
        # onto the targets.
        # target_loss: (torch.float64): loss after decoding all responses and a variance loss.
        # target_loss.shape = [ 1 ]
        responses_hidden, _ = joining_context( return_ops, batchwise_routing_weights[routing_uids], query_responses) 
        target_loss = get_target_loss ( responses_hidden, inputs )
        print ('Loss\t|\t{}'.format( target_loss.item() ))

        importance_loss = self.config.nucleus.importance  * (torch.std(batchwise_routing_weights)/torch.mean(batchwise_routing_weights))**2
        target_loss =  target_loss +  importance_loss

        # === Compute shapely scores ===
        # Computes shapely scores for each endpoint by masking the response and
        # computing the change in loss induced.
        # shapely_scores: (torch.float32): shapely scores per query_response
        # shapely_scores.shape = [ metagraph.n ]
        masked_contexts = partial_contexts(return_ops, active_uids[routing_uids], top_k_routing_weights,  query_responses)
        shapely_scores = torch.zeros( (metagraph.n.item()) )
        # Turn off gradient computation for shapely scores.
        with torch.no_grad():
            self.eval()
            unmasked_loss = get_target_loss(responses_hidden, inputs)
            # Iterate over all responses creating a masked context.
            for i,uid in enumerate(masked_contexts):
                # Create mask by zeroing out the response at index.              
                masked_loss = get_target_loss ( masked_contexts[uid], inputs )
                shapely_score = unmasked_loss - masked_loss
                print ('Shapely\t|\tuid: {}\tweight: {}\tscore: {}'.format( uid, batchwise_routing_weights[routing_uids][i], -shapely_score.item() ))
                shapely_scores[ uid ] = -shapely_score

        
        # === Done ===
        return target_loss, shapely_scores
