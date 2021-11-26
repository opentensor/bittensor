import bittensor
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math
class Validator( torch.nn.Module ):

        def __init__(self, config, metagraph, dendrite, device):
            super(Validator, self).__init__()

            self.embedding = torch.nn.Embedding( bittensor.__vocab_size__,  bittensor.__network_dim__ )
            self.layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout )
            self.encoder = TransformerEncoder( self.layers, config.nucleus.nlayers )
            self.c_layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout )
            self.local_encoder = TransformerEncoder(self.c_layers, 1)
            self.decoder = torch.nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__ , bias=False)
            self.loss_fct = torch.nn.CrossEntropyLoss()
            self.peer_weights = torch.nn.Parameter(torch.ones( [ metagraph().n.item() ] , requires_grad=True, device = device))
            self.metagraph = metagraph
            self.dendrite = dendrite
            self.config = config
            self.device = device
            self.gates = {}
            self.sync_with_chain_state()


        def forward ( self, inputs ):
            # Apply model.
            active_uids = torch.where(self.metagraph().active > 0)[0]

            query_hidden = self.query( inputs)
            encoded_hidden = self.encoder( query_hidden )
            decoded_targets = self.decoder ( encoded_hidden )


            # regularization of importance
            importance_loss = self.config.nucleus.importance  * (torch.std(self.total_weights[active_uids])/torch.mean(self.total_weights[active_uids]))**2


            # Compute loss.
            shift_logits = decoded_targets[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()     
            self.loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) ) 
            self.total_loss = self.loss + importance_loss
            return self.total_loss, decoded_targets

        def scores ( self ):
            """Computes salience scores for each peer in the network w.r.t the loss. 
            We use a simplified fishers information score. score_i = hessian_ii * peer_weight_i^2
            """
            peer_weights_d1 = torch.autograd.grad(self.loss, self.total_weights, create_graph=True, retain_graph=True, allow_unused=True)[0]
            if peer_weights_d1 == None: return torch.ones_like( self.total_weights ) * (1 / self.metagraph().n.item()) # None if no grad w.r.t the chain weights.
            peer_weights_d2 = torch.autograd.grad(peer_weights_d1.sum(), self.total_weights, retain_graph=True, allow_unused=True )[0]
            validator_scores =  peer_weights_d2 * (self.total_weights**2)/2  
            return validator_scores

        def query ( self, inputs ):

            # ---- Get active peers and their weights ---- 
            active_uids = torch.where(self.metagraph().active > 0)[0]
            active_peer_weights = self.peer_weights[active_uids]

            # --- Create the local context ---
            local_context = self.local_encoder( self.embedding( inputs ) )* math.sqrt(bittensor.__network_dim__)

            endpoints, topk_weights = self.route(inputs,local_context)
            # ---- Query network ----
            responses, return_ops, query_times = self.dendrite.forward_text ( 
                endpoints = endpoints, 
                inputs = inputs
            )

            # ---- Join based on weights ----
            joining_uids = torch.where(return_ops== bittensor.proto.ReturnCode.Success)[0]
            joining_weights = F.softmax( topk_weights[(return_ops == bittensor.proto.ReturnCode.Success)], dim = 0 )

            output = torch.zeros( (inputs.shape[0], inputs.shape[1], bittensor.__network_dim__)).to( self.device )
            for index, joining_weight in enumerate( joining_weights ): 
                output += responses[joining_uids[index]].to( self.device ) * joining_weight

            return output

        def sync_with_chain_state( self ):
            r""" Creates new parameters based on metagraph size.
                Args:
                    metagraph (:obj: `bittensor.Metagraph'`, `required`):
                        bittensor metagraph object.
            """
            # Add new gates for each uid.
            for uid in self.metagraph().uids.tolist():
                if uid in self.gates.keys():
                    pass
                else:
                    self.gates[uid] = torch.nn.Linear( bittensor.__network_dim__, 1, bias=True).to(self.device)


        def route(
            self, 
            inputs: torch.FloatTensor, 
            context: torch.FloatTensor,
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
            # For ease of use.
            batch_size = inputs.shape[0]
            metagraph = self.metagraph()
            # Active uids in the metagraph
            active_uids = torch.where(metagraph.active > 0)[0]

            # Get weights for uids.
            # weights: (torch.FloatTensor): weights for each filtered_uid
            # weights.shape = [batch size , n_filtered]
            # gates use the last token for context 
            weights = torch.cat( [ self.gates[ uid ](context[:,-1,:]) for uid in active_uids.tolist() ], axis = 1)

            # Normalize weights across batch dimension. 
            # filtered_weights_mean: (torch.FloatTensor): normalized weights across batch dimension. 
            # filtered_weights_mean.shape = [ n_filtered ]
            filtered_mean_weights = torch.mean(weights, axis = 0)
            noise = torch.normal( 0, torch.std(filtered_mean_weights).item(), size=( filtered_mean_weights.size())).to( self.config.neuron.device )

            self.total_weights = torch.zeros(self.metagraph().n.item(), device = self.device)
            for i in range(len(active_uids)):
                self.total_weights[active_uids[i]] = filtered_mean_weights[i] + noise[i]

            # Get indices and values for uids with highest scores.
            # topk_weights: (torch.float64): scores of uids with highest scores.
            # topk_weights.shape = [ real_topk ]
            # topk_indices: (torch.LongTensor): indicies of uids with highest scores.
            # topk_indices.shape = [ real_topk ]
            real_topk = min( len(active_uids), self.config.neuron.topk )
            topk_weights, topk_indices = torch.topk(self.total_weights[active_uids], real_topk, dim=0)

            # Get the real uids with the top scores.
            # real_filtered_topk_uids: (torch.LongTensor): uids with highest scores.
            # real_filtered_topk_uids.shape = [ real_topk ]
            
            # Get endpoint information for the highest scoring uids.
            # neurons: List[bittensor.proto.Neuron]: endpoint information for filtered uids.
            # len(neurons) == real_topk
            filtered_endpoints = []
            for uid in active_uids[topk_indices]:
                filtered_endpoints.append( metagraph.endpoints[ uid ] )
            return filtered_endpoints, topk_weights

