import bittensor
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from ..neuron_utilities import joining_context, jacobian, partial_contexts


class Validator( torch.nn.Module ):

        def __init__(self, config, metagraph, dendrite, device):
            super(Validator, self).__init__()
            self.layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout, batch_first=True)
            self.encoder = TransformerEncoder( self.layers, config.nucleus.nlayers )
            self.decoder = torch.nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__ , bias=False)
            self.loss_fct = torch.nn.CrossEntropyLoss()
            self.peer_weights = torch.nn.Parameter(torch.ones( [ metagraph().n.item() ] , requires_grad=True, device = device))
            self.noise_offset = 0.0000001
            self.metagraph = metagraph
            self.dendrite = dendrite
            self.config = config
            self.device = device


        def forward ( self, inputs ):
            self.train()
            # Apply model.
            query_hidden = self.query( inputs )
            encoded_hidden = self.encoder( query_hidden )
            decoded_targets = self.decoder ( encoded_hidden )

            # Compute loss.
            shift_logits = decoded_targets[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()     
            self.loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) )
            return self.loss, decoded_targets

        def scores ( self , loss, inputs  ):
            """Computes a mixture of greedy saliency and shapley scores for each peer in the network w.r.t the loss. 
            """
            validator_scores = torch.zeros(self.peer_weights.size())
            with torch.no_grad():
                self.eval()
                estimate_loss = self.decode_remote( self.output, inputs )

                for uid in self.partial_context:
                    partial_remote_target_loss = self.decode_remote( self.partial_context[uid],inputs )
                    validator_scores[uid] =  (partial_remote_target_loss - estimate_loss)/estimate_loss 
                           
            peer_weights_d1 = jacobian(loss, self.peer_weights)
            first_order = (peer_weights_d1.detach()* -self.peer_weights.detach())
            validator_scores= F.normalize(validator_scores, p = 2,dim=0)*(0.5) + F.normalize(first_order, p = 2,dim=0)*(0.5)
            return validator_scores

        def decode_remote(self, context, inputs):
            remote_hidden = self.encoder( context)
            remote_target = self.decoder(remote_hidden)  
            shift_logits = remote_target[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            partial_remote_target_loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) ).item()

            return partial_remote_target_loss

        def query ( self, inputs ):

            # ---- Get active peers and their weights ---- 
            active_uids = torch.where(self.metagraph().active > 0)[0]
            active_peer_weights = self.peer_weights[active_uids]

            # ---- Topk Weights ---- (TODO: check if the gaussians are enough disrupt the chain weights)
            real_topk = min( self.config.nucleus.topk, self.metagraph().n.item(), len(active_uids))
            noise = torch.normal( 0, torch.std(active_peer_weights).item()+self.noise_offset, size=( active_peer_weights.size())).to( self.config.neuron.device )
            topk_weights, topk_idx = bittensor.unbiased_topk(active_peer_weights + noise , real_topk, dim=0)
            topk_uids = active_uids[topk_idx]

            # ---- Query network ----
            responses, return_ops, query_times = self.dendrite.forward_text ( 
                endpoints = self.metagraph().endpoints[ topk_uids ], 
                inputs = inputs
            )

            # ---- Join based on weights ----
            joining_uids = torch.where(return_ops== bittensor.proto.ReturnCode.Success)[0]
            joining_weights = F.softmax( topk_weights[(return_ops == bittensor.proto.ReturnCode.Success)], dim = 0 )
            output = torch.zeros( (inputs.shape[0], inputs.shape[1], bittensor.__network_dim__)).to( self.device )
            for index, joining_weight in enumerate( joining_weights ): 
                output += responses[joining_uids[index]].to( self.device ) * joining_weight

            self.output = output.detach()
            # ---- Calculate masked peers ----
            self.partial_context = partial_contexts(return_ops, topk_uids, topk_weights, responses)

            return output