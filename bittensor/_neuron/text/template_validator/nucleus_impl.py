import bittensor
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

class Validator( torch.nn.Module ):

        def __init__(self, config, metagraph, dendrite, device):
            super(Validator, self).__init__()
            self.layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout )
            self.encoder = TransformerEncoder( self.layers, config.nucleus.nlayers )
            self.decoder = torch.nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__ , bias=False)
            self.loss_fct = torch.nn.CrossEntropyLoss()
            self.peer_weights = torch.nn.Parameter(torch.ones( [ metagraph.n.item() ] , requires_grad=True))
            self.noise_offset = 0.0000001
            self.metagraph = metagraph
            self.dendrite = dendrite
            self.config = config
            self.device = device


        def forward ( self, inputs ):
            # Apply model.
            query_hidden = self.query( inputs.to( self.device ) )
            encoded_hidden = self.encoder( query_hidden )
            decoded_targets = self.decoder ( encoded_hidden )

            # Compute loss.
            shift_logits = decoded_targets[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()     
            self.loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) )
            return self.loss, decoded_targets

        def scores ( self ):
            """Computes salience scores for each peer in the network w.r.t the loss. 
            We use a simplified fishers information score. score_i = hessian_ii * peer_weight_i^2
            """
            peer_weights_d1 = torch.autograd.grad(self.loss, self.peer_weights, create_graph=True, retain_graph=True, allow_unused=True)[0]
            if peer_weights_d1 == None: return torch.ones_like( self.peer_weights ) * (1 / self.metagraph.n.item()) # None if no grad w.r.t the chain weights.
            peer_weights_d2 = torch.autograd.grad(peer_weights_d1.sum(), self.peer_weights, retain_graph=True, allow_unused=True )[0]
            validator_scores =  peer_weights_d2 * (self.peer_weights**2)/2  
            return validator_scores

        def query ( self, inputs ):

            # ---- Get active peers and their weights ---- 
            active_uids = torch.where(self.metagraph.active > 0)[0]
            active_peer_weights = self.peer_weights[active_uids]

            # ---- Topk Weights ---- (TODO: check if the gaussians are enough disrupt the chain weights)
            real_topk = min( self.config.nucleus.topk, self.metagraph.n.item(), len(active_uids))
            noise = torch.normal( 0, torch.std(active_peer_weights).item()+self.noise_offset, size=( active_peer_weights.size())).to( self.config.neuron.device )
            topk_weights, topk_idx = torch.topk(active_peer_weights + noise , real_topk, dim=0)
            topk_uids = active_uids[topk_idx]

            # ---- Query network ----
            responses, return_ops, query_times = self.dendrite.forward_text ( 
                endpoints = self.metagraph.endpoints[ topk_uids ], 
                inputs = inputs
            )

            # ---- Join based on weights ----
            joining_uids = torch.where(return_ops== bittensor.proto.ReturnCode.Success)[0]
            joining_weights = F.softmax( topk_weights[(return_ops == bittensor.proto.ReturnCode.Success)], dim = 0 )
            output = torch.zeros( (inputs.shape[0], inputs.shape[1], bittensor.__network_dim__)).to( self.device )
            for index, joining_weight in enumerate( joining_weights ): 
                output += responses[joining_uids[index]].to( self.device ) * joining_weight

            # ---- Punish peers with non-successful return ops ----
            with torch.no_grad():
                self.peer_weights[topk_uids[(return_ops != bittensor.proto.ReturnCode.Success)]] -= self.config.nucleus.punishment
                self.peer_weights[ self.peer_weights < -1 ] = -1 # lower bound for chain weights 

            return output