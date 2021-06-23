'''
The MIT License (MIT)
Copyright © 2021 Opentensor.ai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
'''

import argparse
import random
import torch
import torch.nn.functional as F

from transformers import GPT2Config, GPT2Model
from torch import nn
from collections.abc import Callable
from types import SimpleNamespace

import bittensor

def nextbatch(data, batch_size, tokenizer):
    """ Returns a random batch of sentences from text dataset.

        Args:
            data: (List[dict{'text': str}]): Dataset of text inputs.
            batch_size: size of batch to create.
        
        Returns:
            batch_inputs torch.Tensor (batch_size, sequence_length): List of tokenized sentences.
    """
    batch_text = []
    for _ in range(batch_size):
        batch_text.append(data[random.randint(0, len(data))]['sentence'])
    batch_inputs = tokenizer(batch_text, return_tensors='pt', padding=True, truncation=True)['input_ids']
    return batch_inputs

class GPT2Pooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class GPT2LMNucleus(torch.nn.Module):
    """ A Bittensor Nucleus training GPT2 with Causal Language Modelling (CLM)
    """
    def __init__(self, routing_callback, config: 'bittensor.Config' = None, **kwargs):
        r""" Init a new GPT2 nucleus module.

            Args:
                config (:obj:`bittensor.Config`, `required`): 
                    munched config class.
        """
        super(GPT2LMNucleus, self).__init__()
        if config == None:
            config = GPT2LMNucleus.config()
        GPT2LMNucleus.check_config(config)
        self.config = config

        # To be set.
        self.routing_callback = None

        # Build hugging face config.
        huggingface_config = GPT2Config(
                vocab_size=bittensor.__vocab_size__, 
                n_embd=bittensor.__network_dim__,
                n_layer=config.nucleus.n_layer,
                n_head=config.nucleus.n_head, 
                n_inner=config.nucleus.n_inner, 
                activation_function=config.nucleus.activation_function, 
                resid_pdrop=config.nucleus.resid_pdrop, 
                embd_pdrop=config.nucleus.embd_pdrop, 
                attn_pdrop=config.nucleus.attn_pdrop, 
                layer_norm_epsilon=config.nucleus.layer_norm_epsilon, 
                initializer_range=config.nucleus.initializer_range, 
                summary_type=config.nucleus.summary_type, 
                summary_use_proj=config.nucleus.summary_use_proj, 
                summary_activation=config.nucleus.summary_activation, 
                summary_proj_to_labels=config.nucleus.summary_proj_to_labels, 
                summary_first_dropout=config.nucleus.summary_first_dropout, 
        )

        # encoder_layer: encodes tokenized sequences to network dim.
        # [batch_size, sequence_len] -> [batch_size, sequence_len, bittensor.__network_dim__]
        self.transformer = GPT2Model(huggingface_config)

        # pooler_layer: pools the hidden units for use by the pkm dendrite rpc query.
        # [batch_size, bittensor.__network_dim__, sequence_len] -> [batch_size, bittensor.__network_dim__]
        self.pooler = GPT2Pooler(huggingface_config)

        # hidden_layer: transforms context and encoding to network_dim hidden units.
        # [batch_size, sequence_dim, 2 * bittensor.__network_dim__] -> [batch_size, sequence_len, bittensor.__network_dim__]
        self.hidden_layer = nn.Linear( bittensor.__network_dim__, bittensor.__network_dim__ )

        # target_layer: maps from hidden layer to vocab dimension for each token. Used by MLM loss.
        # [batch_size, sequence_len, bittensor.__network_dim__] -> [batch_size, sequence_len, bittensor.__vocab_size__]
        self.target_layer = nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__, bias=False )
        
        # Loss function: MLM cross-entropy loss.
        # predicted: [batch_size, sequence_len, 1], targets: [batch_size, sequence_len, 1] -> [1]
        self.loss_fct = nn.CrossEntropyLoss()

        self.to(self.device)

    @staticmethod   
    def config() -> SimpleNamespace:
        parser = argparse.ArgumentParser(); 
        GPT2LMNucleus.add_args(parser) 
        config = bittensor.config( parser ); 
        return config

    @staticmethod
    def add_args( config: 'SimpleNamespace' ):    
        r""" Add custom params to the parser.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--n_head', default=1, type=int, 
                            help='Number of attention heads for each attention layer in the Transformer encoder.', namesp)
        parser.add_argument('--n_layer', default=2, type=int, 
                            help='Number of hidden layers in the Transformer encoder.')
        parser.add_argument('--n_inner', default=8, type=int, 
                            help='The dimensionality of the inner feed-forward layers. :obj:`None` will set it to 4 times n_embd')
        parser.add_argument('--activation_function', default='gelu_new', type=str, 
                            help='Activation function, to be selected in the list :obj:`["relu", "silu", "gelu", "tanh", "gelu_new"]')
        parser.add_argument('--resid_pdrop', default=0.1, type=float, 
                            help='GPT residual dropout probabilit.')
        parser.add_argument('--embd_pdrop', default=0.1, type=float, 
                            help='GPT embedding dropout probability.')
        parser.add_argument('--attn_pdrop', default=0.1, type=float, 
                            help='GPT attention dropout probability.')
        parser.add_argument('--layer_norm_epsilon', default=1e-05, type=float, 
                            help='GPT the epsilon to use in the layer normalization layers')
        parser.add_argument('--summary_type', default='cls_index', type=str, 
                            help='Supply a Tensor of classification token position (like GPT/GPT-2).')
        parser.add_argument('--initializer_range', default=0.02, type=float, 
                            help='The standard deviation of the truncated_normal_initializer for initializing all weight matrices.')
        parser.add_argument('--summary_use_proj', default=True, type=bool, 
                            help='Whether or not to add a projection after the vector extraction.')
        parser.add_argument('--summary_activation', type=str, 
                            help='Pass "tanh" for a tanh activation to the output, any other value will result in no activation.')
        parser.add_argument('--summary_proj_to_labels', default=True, type=bool, 
                            help='Whether the projection outputs should have config.num_labels or config.hidden_size classes.')
        parser.add_argument('--summary_first_dropout', default=0.1, type=float, 
                            help='The dropout ratio to be used after the projection and activation.')
        parser.add_argument('--n_block_filter', default=100, type=int, 
                            help='Stale neurons are filtered after this many blocks.')
        parser.add_argument('--gradient_checkpointing', default=True, type=bool, 
                            help='Stale neurons are filtered after this many blocks.')
        return parser.parse_known_args( namespace = namespace )

    @staticmethod
    def check_config(config: 'bittensor.Config'):
        pass

    def attach_routing_callback(self, routing_callback: Callable[ [torch.Tensor, torch.Tensor], torch.Tensor ] ):
        """ Assigns the routing_callback call to this neuron.

            Returns:
                routing_callback (:callabl:`Callable[ [torch.Tensor, torch.Tensor], torch.Tensor `, `required`): 
                    Routing function to call on self.route()
        """
        self.routing_callback = routing_callback

    @property
    def route( self, inputs: torch.Tensor, query: torch.Tensor ):
        """ Calls this nucleus's subscribed routing function. self.routing_callback must be set before this call is made.

        Args:
            inputs (:obj:`torch.LongTensor` of shape :obj:`( batch_size, sequence_len )`, `required`): 
                    Batch_size length list of tokenized sentences.

            query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, query_dimension)`, `required`): 
                    Context tensor used to select which neurons to query for each example.
            
         Returns:
            remote_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                joined responses from network call.
        """
        if self.routing_callback == None:
            raise RuntimeError('The routing function must be set on this nucleus before a remote_forward call can execute.')
        else:
            return self.routing_callback( inputs = inputs, query = query )

    def forward_text(self, inputs: torch.LongTensor):
        """ Local forward inputs through the MLM GPT Nucleus.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of tokenized sentences.
            
            Returns:
                hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Hidden layer representation produced using the local_context.
        """
        hidden = self.local_forward(inputs=inputs.to(self.device), training = False).local_hidden
        return hidden

    def local_forward(self, inputs: torch.LongTensor, training: bool = True) -> SimpleNamespace:
        r""" Forward pass through GPT nucleus.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.

                training (:obj:`bool')`, `optional`, defaults to True):
                    Switch to True if this forward pass computes an MLM loss.

            SimpleNamespace {
                    local_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer context.

                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer encoding produced using local_context.

                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
                        GPT MLM Target predictions produced using local_context. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        GPT MLM loss using local_context.
                }
        """
        inputs = torch.clamp(inputs, 0, bittensor.__vocab_size__) # Filter out of range tokens.

        # Return vars to be filled.
        output = SimpleNamespace()
        
        # local_context: distilled version of remote_context.
        # local_context.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.local_context = self.transformer(input_ids=inputs, return_dict=True).last_hidden_state

        # local_hidden: hidden layer encoding of sequence with local_context.
        # local_hidden.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.local_hidden = self.hidden_layer(output.local_context)

        if training:
            # local_target: projection of local_hidden onto target dimension.
            # local_target.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
            output.local_target = self.target_layer(output.local_hidden)

            # local_target_loss: MLM loss between local_target and passed targets.
            # local_target_loss.shape = [1]
            shift_logits = output.local_target[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            output.local_target_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                   
        return output

    def remote_forward(self, inputs: torch.LongTensor, training: bool) -> SimpleNamespace:
        """ Forward pass inputs and labels through the GPT2 module.


        Args:
            inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.

            training (:obj:`bool')`, `optional`, defaults to True):
                Switch to True if this forward pass computes an MLM loss.

        Returns:
            self.local_forward() + SimpleNamespace ( 

                    remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `optional`): 
                        Hidden layer encoding produced using the remote_context.

                    remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,  bittensor.__vocab_size__)`, `optional`):
                        GPT MLM Target predictions using the remote_context.

                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                        GPT MLM loss using the remote_context.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss between local_context and remote_context.

                    router (:obj:`SimpleNamespace`, `required`): 
                        Outputs from the pkm dendrite.
            )
        """
        inputs = torch.clamp(inputs, 0, bittensor.__vocab_size__) # Filter out of range tokens.

        # Run the local model.
        # output = SimpleNamespace
        output = self.local_forward(inputs, training)

        # pooled: pooled hidden layer from local run, used as our query context.
        # pooled.shape = [batch_size, bittensor.__network_dim__]
        pooled = self.pooler(output.local_hidden.detach())

        # remote_context: joined responses from a dendrite.forward_text call.
        # remote_context.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.remote_context = self.route( inputs = inputs.to(self.device), query = pooled )
        output.remote_context = output.remote_context

        # distillation_loss: distillation loss between local_context and remote_context
        # distillation_loss.shape = [1]
        output.distillation_loss = F.mse_loss(output.local_context, output.remote_context.detach())

        # remote_hidden: hidden layer encoding using remote_context.
        # remote_hidden.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.remote_hidden = self.hidden_layer(output.remote_context)

        if training:
            # remote_target: projection of remote_hidden onto target dimension.
            # remote_target.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
            output.remote_target = self.target_layer(output.remote_hidden)

            # remote_target_loss: MLM loss between remote_target and passed targets.
            # remote_target_loss.shape = [1]
            shift_logits = output.remote_target[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            output.remote_target_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return output


